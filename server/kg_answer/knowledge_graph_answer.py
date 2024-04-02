import asyncio
import json
from pathlib import Path
from typing import AsyncIterable, List
from urllib.parse import urlencode

from py2neo import Graph
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from starlette.websockets import WebSocket, WebSocketDisconnect

from configs import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE, LLM_MODELS
from server.chat.utils import History
from server.db.repository import add_chat_history_to_db
from server.knowledge_base.kb_doc_api import search_docs
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.utils import get_doc_path
from server.utils import get_ChatOpenAI, get_prompt_template, wrap_done, BaseResponse

graph = Graph('http://82.157.118.175:7474/', auth=('neo4j', 'neo4jpassword'))


def knowledge_graph_answer(query_entities: List[str]):
    kb = KBServiceFactory.get_service_by_name("entity")
    for query in query_entities:
        real_entities = search_docs(query, "entity", 3, 0.5)
        # print(real_entities)


async def LLM(query: str,
              history: List[History] = [],
              model_name: str = LLM_MODELS[0],
              prompt_name: str = "NER",
              professor_context: str = "",
              information: str = "",
              context: str = "",
              stream: bool = False,
              ) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    temperature = TEMPERATURE
    max_tokens = None
    model = get_ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=[callback],
    )

    prompt_template = get_prompt_template("llm_chat", prompt_name)
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    chat_prompt = ChatPromptTemplate.from_messages(
        [i.to_msg_template() for i in history] + [input_msg])
    chain = LLMChain(prompt=chat_prompt, llm=model)

    # Begin a task that runs in the background.
    if prompt_name == "NER":
        task = asyncio.create_task(wrap_done(
            chain.acall({"question": query}),
            callback.done),
        )
    else:
        task = asyncio.create_task(wrap_done(
            chain.acall({"question": query, "information": information, "context": context,
                         "professor_context": professor_context}),
            callback.done),
        )

    answer = ""
    chat_history_id = add_chat_history_to_db(chat_type="llm_chat", query=query)

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


async def kg_answer(websocket: WebSocket):
    try:
        await websocket.accept()
        turn = 1
        while True:
            input_json = await websocket.receive_json()
            query, history, knowledge_base_name = input_json["query"], input_json["history"], input_json[
                "knowledge_base_name"]
            if 'model_name' not in input_json:
                model_name = LLM_MODELS[0]
            else:
                model_name = input_json['model_name']
            kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
            if kb is None:
                await websocket.send_json(BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}").dict())
                await websocket.close()
                return
            history = [History.from_data(h) for h in history]
            await websocket.send_json({"query": query, "turn": turn, "flag": "start"})

            top_k = VECTOR_SEARCH_TOP_K
            score_threshold = SCORE_THRESHOLD
            docs = search_docs(query, knowledge_base_name, top_k, score_threshold)
            if len(docs) == 0:  ## 如果没有找到相关文档,直接返回
                await websocket.send_text("")
                await websocket.send_json({"query": query, "turn": turn, "flag": "none"})
                continue

            doc_path = get_doc_path(knowledge_base_name)
            source_documents = []
            professor_knowledges = []
            context_knowledges = []
            for inum, doc in enumerate(docs):
                filename = Path(doc.metadata["source"]).resolve().relative_to(doc_path)
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"knowledge_base/download_doc?" + parameters
                text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
                if doc.page_content.startswith('question:'):
                    professor_knowledges.append(doc.page_content.split('answer:')[1])
                else:
                    context_knowledges.append(doc.page_content)
                source_documents.append(text)

            context = "\n".join(context_knowledges)
            professor_context = "\n".join(professor_knowledges)
            # print(context)
            # print(professor_context)
            kg = {}
            if len(professor_context) != 0:
                kg['information'] = ""
                kg['entities'] = []
                kg['relations'] = []
                kg['REAL_entities'] = []
            else:
                kg = await kg_information(query)

            prompt_name = get_prompt(context, professor_context, kg['information'])
            async for token in LLM(query, prompt_name=prompt_name, professor_context=professor_context, model_name= model_name,
                                   information=kg['information'], context=context, stream=True):
                await websocket.send_text(token)
            await websocket.send_text("")
            await websocket.send_json(
                json.dumps({"query": query, "turn": turn, "flag": "end", "docs": source_documents,
                            "kg": {'nodes': kg['entities'], 'links': kg['relations']}, "NER": kg['REAL_entities']},
                           ensure_ascii=False))
    except Exception as e:
        await websocket.close()


if __name__ == '__main__':
    graph = Graph('http://82.157.118.175:7474/', auth=('neo4j', 'neo4jpassword'))

    result = graph.run("MATCH (n) return count(n)")
    print(result)
