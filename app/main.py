import os
from logging import getLogger
import time

import requests
from fastapi import Request, BackgroundTasks, FastAPI
from fastapi.responses import FileResponse

from dl.inference import byte_to_wav
import json

from dl.model.fastspeech.synthesize import synthesize_voice
# from dl.voice_infrence import tts

logging = getLogger(__name__)
app = FastAPI()


# uvicorn app.main:app --reload --host=0.0.0.0 --port=8002
# stt
@app.post("/stt")
async def voice_chat_bot_inference(request: Request):
    request_list = await request.form()

    voice = request_list['voice'].file.read()
    user_id = request_list['userId']
    we_id = request_list['weId']
    start = time.time()

    print('voice', voice, ' memberId', user_id, 'we_id', we_id)

    # 목소리 byte to wav
    byte_to_wav(voice, user_id, we_id)

    # wav stt
    voice_to_text = stt(user_id, we_id)
    print('voice_to_text', voice_to_text)
    return {"stt": voice_to_text}
    #
    # # nl inference AI api
    # URL = "http://127.0.0.1:8001/chat_bot"
    # response = requests.get(URL, params={"chatRequest": voice_to_text, "memberId": user_id, "weId": we_id},
    #                         verify=False)
    # json_object = json.loads(response.text)
    # response_text = json_object['response']
    # filtering = json_object['filter']
    # print('response_text', response_text, 'filtering', filtering)
    #
    # tts_wav = ''
    # # 욕설 이나 안내문구일 때 성우 tts
    # if filtering == 1:
    #     tts_wav = tts(user_id, we_id, response_text)
    # else:
    #     # 욕설이 아닐 때 회은이 목소리
    #     tts_wav = synthesize_voice(response_text, user_id, we_id)
    # final = time.time()
    # print('chat_response', response_text, "최종 시간", final - start)
    # # background_tasks.add_task(os.remove, tts_wav)
    # return FileResponse('./results/{}_{}.wav'.format(user_id, we_id), media_type='audio/wav',
    #                     background=background_tasks)


# tts
@app.post("/tts")
async def voice_chat_bot_inference(request: Request, background_tasks: BackgroundTasks):
    request_form = await request.form()
    user_id = request_form['userId']
    we_id = request_form['weId']
    filtering = request_form['filtering']
    text = request_form['text']

    start = time.time()

    tts_wav = ''
    # 욕설 이나 안내문구일 때 성우 tts
    if filtering == 1:
        tts_wav = tts(user_id, we_id, text)
    else:
        # 욕설이 아닐 때 회은이 목소리
        tts_wav = synthesize_voice(text, user_id, we_id)
    final = time.time()
    print('chat_response', text, "최종 시간", final - start)
    # background_tasks.add_task(os.remove, tts_wav)
    return FileResponse('./results/{}_{}.wav'.format(user_id, we_id), media_type='audio/wav',
                        background=background_tasks)


def remove_file(path: str) -> None:
    os.unlink(path)
