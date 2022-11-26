from model.fastspeech.synthesize import synthesize_voice


def tts_voice(text, user_id, we_id):
    voice_url = synthesize_voice(text, user_id, we_id)
    return voice_url
