import pyaudio as pa
import wave
import speech_recognition as sr
from google.cloud import speech
# import speech_recognition as sr
# import playsound


def byte_to_wav(byte_sound):
    file_name = 'sample.wav'
    CHUNK = 1024
    CHANNELS = 1
    FORMAT = pa.paInt16
    RATE = 44100
    # byte -> byte array
    # byte_list = bytearray(byte_sound)

    wf = wave.open(file_name, 'wb')
    wf.setnchannels(CHANNELS)  # 1, 2
    wf.setnframes(CHUNK)  # 1024
    wf.setsampwidth(pa.get_sample_size(FORMAT))  # FORMAT = pyaudio.paInt32
    wf.setframerate(RATE)  # 44100
    wf.writeframes(byte_sound)  # audio가 바로 byte array
    wf.close()


# def stt():
#     client = speech.SpeechClient()
#     gcs_uri = "gs://cloud-samples-data/speech/brooklyn_bridge.raw"
#
#     audio = speech.RecognitionAudio(uri=gcs_uri)
#
#     config = speech.RecognitionConfig(
#         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=16000,
#         language_code="ko-KR",
#     )
#
#     # Detects speech in the audio file
#     response = client.recognize(config=config, audio=audio)
#
#     for result in response.results:
#         print("Transcript: {}".format(result.alternatives[0].transcript))


def stt():
    AUDIO_FILE = 'sample.wav'
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
    return result_sound
