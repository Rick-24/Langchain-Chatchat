from fastapi import APIRouter, Request

from chatchat.server.audio.transcript import transcript_audio
from chatchat.server.utils import BaseResponse

audio_router = APIRouter(prefix="/audio", tags=["Audio"])

audio_router.post(
    "/transcript", response_model= BaseResponse, summary="语音转文本"
)(transcript_audio)