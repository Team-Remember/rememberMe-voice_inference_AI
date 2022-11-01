import time
import logging
from typing import List

from fastapi import FastAPI, File, UploadFile, Request
from pydantic import BaseModel
import pandas as pd

from chatbot_kakao import aichatbot
from training_kakao import pretreatment, make_model_input_form, embedding

app = FastAPI()


# 챗봇
@app.get("/chat_bot")
def chatbot(user_id: float = 0, we_id: float = 0, chat_request: str = ''):
    chat_response = aichatbot(user_id, we_id, chat_request)
    return {"response": chat_response}


# 챗봇 카카오톡 데이터 입력시 학습시키기
@app.post("/chat_train")
async def chatbot_train(files: List[UploadFile] = File(...)):
    # 카카오톡 파일 전처리
    my_katalk_df = pretreatment(files)
    # 한글 전처리, input 데이터프레임으로 변형
    result_dataframe = make_model_input_form(my_katalk_df)

    # 임베딩
    embedding_result = embedding(result_dataframe)
    return {"message": "success!"}


# 서버 실행시
# uvicorn main:app --reload --host=0.0.0.0 --port=8001
# uvicorn main:app --reload

# http://127.0.0.1:8001/docs