import wave

import soundfile as sf
import numpy as np
import scipy.io.wavfile
import pyaudio as pa
from fastapi import APIRouter, File, UploadFile
from typing import List
from logging import getLogger
import time, math, wave
# from src.ml.prediction import classifier,Data
from scipy.io import wavfile

from elastic import make_chatdata_nk
from src.app.audio import byte_to_wav, stt
from src.ml.chatbot_kakao import aichatbot
from src.ml.filter import abuse_filtering
from src.ml.training_kakao import pretreatment, make_model_input_form, embedding
import io
import soundfile as sf

logging = getLogger(__name__)
router = APIRouter()


# 문자 챗봇
@router.get("/chat_bot")
def chatbot(memberId: float = 0, weId: float = 0, chatRequest: str = ''):
    print("request,", chatRequest)
    start = time.time()
    math.factorial(100000)
    # 욕설방지 필터링
    filter = abuse_filtering(chatRequest, 0)

    if filter is not None:
        return filter

    fil = time.time()
    print('욕설 방지 시간', fil - start)
    # chatbot
    chat_response = aichatbot(memberId, weId, chatRequest)
    print(chat_response)
    chat = time.time()
    print('chat', chat - fil)  # chatbot 시간 체크
    print('chat 시간', chat - start)
    return {"response": chat_response}


# 챗봇 카카오톡 데이터 입력시 학습시키기
@router.post("/chat_bot_train")
async def chatbot_train(files: List[UploadFile] = File(...), memberId: float = 0, weId: float = 0):
    # 카카오톡 파일 전처리
    my_katalk_df = pretreatment(files)
    # 한글 전처리, input 데이터프레임으로 변형
    result_dataframe = make_model_input_form(my_katalk_df)

    # 임베딩
    embedding_result_csv_name = embedding(result_dataframe, memberId, weId)

    # es 데이터 insert
    make_chatdata_nk(embedding_result_csv_name, memberId, weId)
    return {"message": "success!"}


# 음성 챗봇
@router.post("/voice_chat_bot_inference")
def voice_chat_bot_inference(user_id: float = 0, we_id: float = 0, voice: bytes = File()):
    start = time.time()
    # 목소리 byte to wav (TODO: 입력 데이터 확장자 확인 필요)
    voice_wav = byte_to_wav(voice)

    # wav stt
    voice_to_text = stt()

    # 욕설 방지
    filter_abuse = abuse_filtering(voice_to_text, 1)
    fil = time.time()

    if filter_abuse is not None:
        return filter_abuse

    print(voice_to_text)

    # chatbot
    chat_response = aichatbot(user_id, we_id, voice_to_text)
    chat = time.time()
    print('욕설 방지 시간', fil - start)

    ## tts



    return {"response": 'hi'}

# 서버 실행시
# uvicorn main:app --reload --host=0.0.0.0 --port=8001
# uvicorn main:app --reload

# http://127.0.0.1:8001/docs
