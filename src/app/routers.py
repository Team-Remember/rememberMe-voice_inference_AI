import wave

from fastapi import APIRouter, File, UploadFile, Request
from typing import List
from logging import getLogger
import time, math, wave

from elastic import insert_chatdata_es, loadchat
from src.app.audio import byte_to_wav, stt
from src.ml.filter import abuse_filtering
from src.ml.training_kakao import pretreatment, make_model_input_form, embedding
from src.ml.voice_infrence import tts

logging = getLogger(__name__)
router = APIRouter()


# 챗봇 카카오톡 데이터 입력시 학습시키기
@router.post("/chat_bot_train_kakao")
async def chatbot_train(files: List[UploadFile] = File(...), memberId: float = 0, weId: float = 0):
    start = time.time()
    # 카카오톡 파일 전처리
    my_katalk_df = pretreatment(files)
    # 한글 전처리, input 데이터프레임으로 변형
    result_dataframe = make_model_input_form(my_katalk_df)

    # 임베딩
    embedding_result_csv_name = embedding(result_dataframe, memberId, weId)

    # es 데이터 insert
    insert_chatdata_es(embedding_result_csv_name, memberId, weId)
    result = time.time()
    print('training 시간', result - start)
    return {"message": "success!"}


# 챗봇 데이터 베이스 데이터 입력시 학습시키기
@router.post("/chat_bot_train_db")
async def chatbot_database_train(request: Request):
    request_body = await request.json()
    print(request_body)
    return "성공!"


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
    chat_response = loadchat(memberId, weId, chatRequest)
    print(chat_response)
    chat = time.time()
    print('chat', chat - fil)  # chatbot 시간 체크
    print('chat 시간', chat - start)
    return {"response": chat_response}


# 음성 챗봇
@router.post("/voice_chat_bot_inference")
def voice_chat_bot_inference(user_id: float = 0, we_id: float = 0, voice: bytes = File()):
    start = time.time()
    # 목소리 byte to wav
    byte_to_wav(voice, user_id, we_id)

    # wav stt
    voice_to_text = stt(user_id, we_id)

    # 욕설 방지
    filter_abuse = abuse_filtering(voice_to_text, 1)
    fil = time.time()

    if filter_abuse is not None:
        return filter_abuse

    print(voice_to_text)

    # chatbot
    chat_response = loadchat(user_id, we_id, voice_to_text)
    chat = time.time()
    print('chat_response', chat_response)
    print('욕설 방지 시간', fil - start)

    # tts
    tts_wav = tts(user_id, we_id, chat_response)

    # wav > byte TODO: wave > byte

    return "hi"


