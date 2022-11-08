import re


def after_inference(before_inference_text):

    # 샵검색  포함시 샵검색 제외후 해당하는 이미지 전달
    if '샵검색' in before_inference_text:
        text = re.sub('샵검색', '', before_inference_text)
        text = text.strip() # 공백 제거
        # 이미지 전달하는 코드 작성
        return "이미지? 이미지 url?"
    elif 'ㅋㅋ' in before_inference_text:
        return "ㅋㅋㅋ와 관련된 이미지"
    elif 'ㅎㅎ' in before_inference_text:
        return 'ㅎㅎㅎ와 관련된 이미지'


    # return after_infrence_text
