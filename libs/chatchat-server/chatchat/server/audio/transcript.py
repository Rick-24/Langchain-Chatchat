from fastapi import UploadFile, File, Form
from xinference.client import Client

from chatchat.server.utils import BaseResponse

client = Client("http://localhost:9997")
# model_uid = client.launch_model(model_name="Belle-whisper-large-v3-turbo-zh", model_type="audio")
# model = client.get_model(model_uid)
DEFAULT_MODEL_NAME = "Belle-whisper-large-v3-zh"
models = {DEFAULT_MODEL_NAME: client.get_model(DEFAULT_MODEL_NAME)}


async def transcript_audio(
        file: UploadFile = File(..., description="上传语音， 支持格式：wav, mp3, flac"),
        model_name: str = Form("Belle-whisper-large-v3-zh", description="模型名称"),
):
    if models[model_name] is None:
        try:
            models[model_name] = client.get_model(model_name)
        except Exception as e:
            return "模型不存在"
    file_content = await file.read()

    result = models[model_name].transcriptions(file_content)
    return BaseResponse(
        code=200, msg="语音转换成功", data={"text": result["text"]}
    )


