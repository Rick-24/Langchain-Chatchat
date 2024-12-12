import asyncio
import json
from typing import AsyncIterable, List

from fastapi.concurrency import run_in_threadpool
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts.chat import ChatPromptTemplate
from py2neo import Graph
from starlette.websockets import WebSocket

from chatchat.server.chat.utils import History
from chatchat.server.knowledge_base.kb_doc_api import search_docs
from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory
from chatchat.server.knowledge_base.utils import format_professor_reference
from chatchat.server.utils import (wrap_done, get_ChatOpenAI, get_default_llm,
                                   BaseResponse, get_prompt_template, api_address
                                   )
from chatchat.settings import Settings

graph = Graph('http://82.157.118.175:7474/', auth=('neo4j', 'neo4jpassword'))


async def LLM(query: str,
              history: List[History] = [],
              model: str = "",
              prompt_name: str = "NER",
              professor_context: str = "",
              information: str = "",
              context: str = "",
              stream: bool = False
              ) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    callbacks = [callback]

    # Enable langchain-chatchat to support langfuse
    import os
    langfuse_secret_key = os.environ.get('LANGFUSE_SECRET_KEY')
    langfuse_public_key = os.environ.get('LANGFUSE_PUBLIC_KEY')
    langfuse_host = os.environ.get('LANGFUSE_HOST')
    if langfuse_secret_key and langfuse_public_key and langfuse_host:
        from langfuse.callback import CallbackHandler
        langfuse_handler = CallbackHandler()
        callbacks.append(langfuse_handler)

    max_tokens = Settings.model_settings.MAX_TOKENS
    temperature = Settings.model_settings.TEMPERATURE
    llm = get_ChatOpenAI(
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=callbacks,
    )

    prompt_template = get_prompt_template("llm_chat", prompt_name)
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    chat_prompt = ChatPromptTemplate.from_messages(
        [i.to_msg_template() for i in history] + [input_msg])
    chain = chat_prompt | llm

    # Begin a task that runs in the background.
    if prompt_name == "NER":
        task = asyncio.create_task(wrap_done(
            chain.ainvoke({"question": query}),
            callback.done),
        )
    else:
        task = asyncio.create_task(wrap_done(
            chain.ainvoke({"question": query, "information": information, "context": context,
                           "professor_context": professor_context}),
            callback.done),
        )

    answer = ""
    if stream:
        async for token in callback.aiter():
            yield token
            answer += token
    else:
        async for token in callback.aiter():
            answer += token
        yield answer


def get_prompt(context: str, professor_context: str, information: str):
    if len(professor_context) != 0:
        return "professor"
    state = 0
    state |= (0 if len(context) == 0 else 1)
    state |= (0 if len(information) == 0 else 1) << 1
    if state == 1:
        # 只有上下文信息
        return "context"
    elif state == 2:
        # 只有三元组信息
        return "reasoning"
    elif state == 3:
        # 有上下文和专家信息
        return "reasoning_context"


async def kg_information(query: str):
    kg = {}
    try:
        NER_entities_str = ""
        async for token in LLM(query, prompt_name="NER"):
            NER_entities_str += token
        # print(NER_entities_str)
        NER_entities = NER_entities_str.split('[')[1].split(']')[0].split(',')
        REAL_entities = []
        for entity in NER_entities:
            for doc in search_docs(entity, "entity", 2, 0.8):
                REAL_entities.append(doc.page_content)
        kg['REAL_entities'] = REAL_entities

        cypher = """
                   WITH {} AS names
                   UNWIND range (0, size (names) - 1) AS i
                   UNWIND range (0, size (names) - 1) AS j
                   WITH names[i] AS from, names[j] AS to
                   MATCH path = (startNode{{name: from}})-[*]->(endNode{{name: to}})
                   WITH nodes (path) AS nodesList, relationships (path) AS relsList, length (path) AS pathLength
                   UNWIND range (0, pathLength - 1) AS idx
                   RETURN DISTINCT id(nodesList[idx]) as sourceId, nodesList[idx]. name AS sourceName, TYPE (relsList[idx]) AS relation, nodesList[idx + 1]. name AS targetName, id(nodesList[idx + 1]) as targetId

                   UNION 

                   WITH {} AS querys
                   UNWIND querys AS query
                   MATCH path = (startNode{{name: query}})-[*0.. 2]->(endNode)
                   WITH nodes (path) AS nodesList, relationships (path) AS relsList, length (path) AS pathLength
                   UNWIND range (0, pathLength - 1) AS idx
                   RETURN DISTINCT id(nodesList[idx]) as sourceId, nodesList[idx]. name AS sourceName, TYPE (relsList[idx]) AS relation, nodesList[idx + 1]. name AS targetName, id(nodesList[idx + 1]) as targetId
               """.format(REAL_entities, REAL_entities)

        graphResult = graph.run(cypher)
        triplets = graphResult.data()
        source_node_properties = ['sourceId', 'sourceName']
        target_node_properties = ['targetId', 'targetName']
        # 构建一个字典集合，用于去重

        kg['entities'] = [{'id': node[0], 'name': node[1]} for node in set([node_tuple for sublist in [
            (tuple(item[key] for key in source_node_properties), tuple(item[key] for key in target_node_properties))
            for item in triplets] for node_tuple in sublist]) if node[0] is not None]

        kg['relations'] = [{'source': item['sourceId'], 'relation': item['relation'], 'target': item['targetId']} for
                           item
                           in
                           triplets if item['sourceId'] is not None and item['targetId'] is not None]

        kg['information'] = ','.join(
            list(map(lambda x: "<{},{},{}>".format(x['sourceId'], x['relation'], x['targetId']), triplets)))
    except Exception as e:
        # print(e)
        kg['information'] = ""
        kg['entities'] = []
        kg['relations'] = []
        kg['REAL_entities'] = []
    return kg


async def kb_graph_chat(websocket: WebSocket):
    try:
        await websocket.accept()
        turn = 1
        while True:
            input_json = await websocket.receive_json()
            query, history, knowledge_base_name = input_json["query"], input_json["history"], input_json[
                "knowledge_base_name"]
            if 'model_name' not in input_json:
                model_name = get_default_llm()
            else:
                model_name = input_json['model_name']

            kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
            if kb is None:
                await websocket.send_json(BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}").dict())
                await websocket.close()
                return
            ok, msg = kb.check_embed_model()
            if not ok:
                raise ValueError(msg)
            history = [History.from_data(h) for h in history]
            await websocket.send_json({"query": query, "turn": turn, "flag": "start"})

            docs = await run_in_threadpool(search_docs,
                                           query=query,
                                           knowledge_base_name=knowledge_base_name,
                                           top_k=Settings.kb_settings.VECTOR_SEARCH_TOP_K,
                                           score_threshold=Settings.kb_settings.SCORE_THRESHOLD,
                                           file_name="",
                                           metadata={})
            if len(docs) == 0:  ## 如果没有找到相关文档，直接返回
                await websocket.send_text("")
                await websocket.send_json({"query": query, "turn": turn, "flag": "none"})
                continue
            source_documents, context, professor_context = format_professor_reference(knowledge_base_name, docs,
                                                                                      api_address(is_public=True))

            kg = {}
            if len(professor_context) != 0:
                kg['information'] = ""
                kg['entities'] = []
                kg['relations'] = []
                kg['REAL_entities'] = []
            else:
                kg = await kg_information(query)

            prompt_name = get_prompt(context, professor_context, kg['information'])
            async for token in LLM(query, prompt_name=prompt_name, professor_context=professor_context,
                                   model=model_name,
                                   history=history,
                                   information=kg['information'], context=context, stream=True):
                await websocket.send_text(token)
            await websocket.send_text("")
            await websocket.send_json(
                json.dumps({"query": query, "turn": turn, "flag": "end", "docs": source_documents,
                            "kg": {'nodes': kg['entities'], 'links': kg['relations']}, "NER": kg['REAL_entities']},
                           ensure_ascii=False))
    finally:
        await websocket.close()
