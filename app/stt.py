import os

import speech_recognition as sr


def byte_to_wav(byte_sound, user_id, we_id):
    with open(f'wav{user_id}_{we_id}.wav', mode='bx') as f:
        f.write(byte_sound)


def stt(user_id, we_id):
    AUDIO_FILE = f'wav{user_id}_{we_id}.wav'
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)
    try:
        result_sound = r.recognize_google(audio, language='ko-KR')
        print("Google Speech Recognition thinks you said : " + result_sound)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    # 파일 삭제
    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)
    return result_sound