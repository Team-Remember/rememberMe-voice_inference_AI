import os
import time
from logging import getLogger

from fastapi import Request, BackgroundTasks, FastAPI
from fastapi.responses import FileResponse

from app.stt import byte_to_wav, stt
from model.fastspeech.synthesize import synthesize_voice
from app.voice_infrence import tts

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


# tts
@app.post("/tts")
def voice_chat_bot_inference(userId: int, weId: int, filtering: int, text: str,
                                   background_tasks: BackgroundTasks):
    user_id = userId
    we_id = weId
    start = time.time()
    text = text + '어'
    print(user_id, we_id, filtering, text)

    tts_wav = ''
    # 욕설 이나 안내문구일 때 성우 tts
    if filtering == 1:
        tts_wav = tts(user_id, weId, text)
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
