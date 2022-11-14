import math
import os
import time
from logging import getLogger
from typing import List

from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTasks

from elastic import insert_chatdata_es, loadchat
from fastspeech.synthesize import synthesize_voice
from src.app.audio import byte_to_wav, stt
from src.ml.filter import abuse_filtering
from src.ml.training_chatbot import pretreatment_kakao_file, make_model_input_form, embedding_csv, \
    make_model_input_form_from_db
from src.ml.voice_infrence import tts

logging = getLogger(__name__)
router = APIRouter()


# 챗봇 카카오톡 데이터 입력시 학습시키기
@router.post("/chat_bot_train_kakao")
async def chatbot_train(memberId: str, weId: str, files: List[UploadFile] = File(...)):
    start = time.time()
    # 카카오톡 파일 전처리
    my_katalk_df = pretreatment_kakao_file(files)

    # input 데이터프레임으로 변형
    result_dataframe = make_model_input_form(my_katalk_df)

    # 임베딩
    embedding_result_csv_name = embedding_csv(result_dataframe, memberId, weId)

    # es 데이터 insert
    insert_chatdata_es(embedding_result_csv_name, memberId, weId)
    result = time.time()
    print('training 시간', result - start)

    # es 데이터 insert 후 csv 삭제
    os.remove(embedding_result_csv_name)
    return {"message": "success!"}


# 챗봇 데이터 베이스 데이터 입력시 학습시키기
@router.post("/chat_bot_train_db")
async def chatbot_database_train(request: Request):
    request_list = await request.json()
    # 전처리
    print(request_list)
    memberId = request_list[0]['memberId']
    opponentId = request_list[0]['opponentId']
    print(memberId, opponentId)
    # format
    embedding_result_df = make_model_input_form_from_db(request_list)

    # 임베딩
    embedding_result_csv_name = embedding_csv(embedding_result_df, memberId, opponentId)

    # es 데이터 insert
    insert_chatdata_es(embedding_result_csv_name, memberId, opponentId)

    # es 데이터 insert 후 csv 삭제
    os.remove(embedding_result_csv_name)
    return "성공!"


# 문자 챗봇
@router.get("/chat_bot")
def chatbot(memberId: int, weId: int, chatRequest: str = ''):
    print("request,", chatRequest)
    print('memberId', memberId, 'weId', weId)
    start = time.time()
    math.factorial(100000)
    # 욕설방지 필터링
    abuse_filter = abuse_filtering(chatRequest, 0)
    print("None 이야?", abuse_filter)
    if abuse_filter is not None:
        return {"response": abuse_filter}

    fil = time.time()
    print('욕설 방지 시간', fil - start)
    # chatbot
    chat_response = loadchat(memberId, weId, chatRequest)
    print(chat_response)
    chat = time.time()
    print('chat', chat - fil)  # chatbot 시간 체크
    print('chat 시간', chat - start)
    return {"response": chat_response}


def remove_file(path: str) -> None:
    os.unlink(path)


# 음성 챗봇
@router.post("/voice_chat_bot_inference")
async def voice_chat_bot_inference(request: Request, background_tasks: BackgroundTasks):
    request_list = await request.form()

    voice = request_list['voice'].file.read()
    memberId = request_list['userId']
    weId = request_list['weId']
    start = time.time()

    print('voice', voice)
    print('memberId', memberId)
    print('weId', weId)

    # 목소리 byte to wav
    byte_to_wav(voice, memberId, weId)

    # wav stt
    voice_to_text = stt(memberId, weId)

    # 욕설 방지
    filter_abuse = abuse_filtering(voice_to_text, 1)
    fil = time.time()

    print(voice_to_text)

    # chatbot
    chat_response = loadchat(memberId, weId, voice_to_text)
    chat = time.time()

    print('욕설 방지 시간', fil - start)

    tts_wav = ''
    # 욕설일 때 성우 tts
    if filter_abuse is not None:
        chat_response = filter_abuse
        tts_wav = tts(memberId, weId, chat_response)
    elif chat_response == '챗봇의 데이터가 충분하지 않습니다. 카카오톡 데이터를 넣어주세요!' or chat_response == '미안해요.. 당신의 말을 이해하지 못했어요..':
        # 위의 안내 문구일 경우 성우 tts
        tts_wav = tts(memberId, weId, chat_response)
    else:
        # 욕설이 아닐 때 회은이 목소리
        tts_wav = synthesize_voice(chat_response)
    final = time.time()
    print('chat_response', chat_response)
    print("최종 시간", final-start )
    # background_tasks.add_task(remove_file, tts_wav)
    return FileResponse(tts_wav, media_type='audio/wav')
