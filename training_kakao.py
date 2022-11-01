import pandas as pd
from sentence_transformers import SentenceTransformer

# 한국어 bert 모델 불러오기
from tqdm import tqdm

model = SentenceTransformer('jhgan/ko-sroberta-multitask')


def pretreatment(files):
    pretreatment_result = list()
    kakao_before_contents = ''
    for file in files:
        # 파일 읽기
        kakao_before_contents += file.file.read().decode('utf8')

    # 전처리
    chat_room = katalk_msg_parse(kakao_before_contents)
    # 전처리 결과 append
    pretreatment_result.append(chat_room)

    pretreatment_result_df = pd.concat(pretreatment_result, ignore_index=True)

    return pretreatment_result_df


# 카카오톡 데이터 전처리
def katalk_msg_parse(kakao_before_contents):
    my_katalk_data = list()
    katalk_msg_pattern = "[0-9]{4}[년.] [0-9]{1,2}[월.] [0-9]{1,2}[일.] 오\S [0-9]{1,2}:[0-9]{1,2},.*:"
    date_info = "[0-9]{4}년 [0-9]{1,2}월 [0-9]{1,2}일 \S요일"
    in_out_info = "[0-9]{4}[년.] [0-9]{1,2}[월.] [0-9]{1,2}[일.] 오\S [0-9]{1,2}:[0-9]{1,2}:.*"
    file_info = "Talk_[0-9]{4}.[0-9]{1,2}.[0-9]{1,2} [0-9]{1,2}:[0-9]{1,2}-[0-9].txt"
    save_file_info = "저장한 날짜 : [0-9]{4}[.] [0-9]{1,2}[.] [0-9]{1,2}[.] 오\S [0-9]{1,2}:[0-9]{1,2}"

    for line in kakao_before_contents.split('\n'):
        if re.match(date_info, line) or re.match(in_out_info, line) or re.match(file_info, line) or re.match(
                save_file_info, line):
            continue
        elif line == '\n':
            continue
        elif re.match(katalk_msg_pattern, line):
            line = line.split(",")
            #             date_time = line[0]
            user_text = line[1].split(" : ", maxsplit=1)
            user_name = user_text[0].strip()
            text = user_text[1].strip().replace("\n", "")
            text = pretreatment_line(text) # 전처리

            my_katalk_data.append({
                'user_name': user_name,
                'text': text
            })

        else:
            if len(my_katalk_data) > 0:
                text_2 = line.strip()
                text_2 = pretreatment_line(text_2)
                my_katalk_data[-1]['text'] += "\n" + text_2

    my_katalk_df = pd.DataFrame(my_katalk_data)
    print(my_katalk_df)
    return my_katalk_df


def pretreatment_line(before):
    # 전처리
    only_BMP_pattern = re.compile("["u"\U00010000-\U0010FFFF""]+", flags=re.UNICODE)  # 이모티콘
    # url = re.compile(r'[- ]*http+[s]*://+[a-zA-Z0-9-_.?=/&]*') # url
    text = re.sub('이모티콘', '', before)
    text = re.sub('\n', ' ', text)
    text = re.sub(only_BMP_pattern, ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub('사진', '', text)
    text = re.sub('음성메시지', '', text)
    text = re.sub('샵검색 : #', '', text) + ' 이거 검색해 봐'
    text = re.sub('ㅋ', '', text)
    text = re.sub('삭제된 메시지입니다.', '', text)
    text = re.sub(r'파일: [a-zA-Z0-9.?/&=:가-힣]*' + '.' + r'[a-zA-Z0-9.?/&=:가-힣]*', '', text)
    text = re.sub(r'[?/&=#\\:-{}]', '', text)  # 특수문자 제거
    text = re.sub(r'[0-9,]*원을 보냈어요. 송금 받기 전까지 내역 상세화면에서 취소할 수 있어요.', '', text)
    text = re.sub(r'[0-9,]*원을 보냈어요 송금 받기 전까지 내역 상세화면에서 취소할 수 있어요', '', text)
    after = text.replace('[네이버 지도]', '\n[네이버 지도]')

    return after


def make_model_input_form(my_katalk_df):
    df = my_katalk_df
    df['counter'] = 0

    counter = 0

    ## 현재 화자와 다음 화자가 다른 경우, counter를 증가시킨다.
    for i in range(1, len(df)):
        current_person = df.iloc[i]['user_name']
        next_person = df.iloc[i - 1]['user_name']
        if current_person != next_person:
            counter = counter + 1
        df.at[i, 'counter'] = counter

    # 화자와 counter를 더해 message_idx를 만든다.
    df['message_idx'] = df.user_name.astype(str) + "-" + df.counter.astype(str)

    ## counter와 message_idx로 그룹바이하여 문장을 리스트로 묶는다.
    parsed = pd.DataFrame(df.groupby(['counter', 'message_idx'])['text'].apply(list))

    ## 묶은 라인을 하나의 스트링으로 합치고 줄바꿈을 마지막에 붙인다.
    parsed['line2'] = parsed['text'].map(lambda x: " ".join(x).strip())
    pared_line2_list = parsed['line2'].to_list()
    parsing_list = list(filter(None, pared_line2_list))  # 빈칸 제거
    Q_list = list()
    A_list = list()

    # Q, A 분류
    for index, sentence in enumerate(parsing_list):
        if index % 2 == 0:
            Q_list.append(sentence.strip())
        else:
            A_list.append(sentence.strip())

    # 길이 맞추기
    if len(Q_list) > len(A_list):
        A_length = len(A_list)
        Q_list = Q_list[:A_length]
    elif len(A_list) > len(Q_list):
        Q_length = len(Q_list)
        A_list = A_list[:Q_length]

    # 반환할 데이터 프레임 만들기
    result_dataframe = pd.DataFrame({'Q': Q_list, 'A': A_list})

    result_dataframe.to_csv('result_dataframe.csv', index=None)
    return result_dataframe


def embedding(dataframe):
    embeding_list = []
    for temp in tqdm(dataframe['A']):
        embed_temp = model.encode(temp)
        embeding_list.append(embed_temp)

    df_embeding = pd.DataFrame(embeding_list)
    print(df_embeding)
    df_embeding.to_csv('embeding2.csv', header=None, index=None)


import re


def ck_url(x):
    # url에 대한 정규식
    re_equ = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0- 9.\-]+[.][az]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\() [^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>])+\ )))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”'']))"
    ck_url = re.findall(re_equ, x)
    if ck_url:
        return "문자열의 URL: ", [i[0] for i in ck_url]
    else:
        return "URL이 없습니다!"
        # ip_str = input("문자열을 입력하세요: ") print(ck_url(ip_str) ))
# kakao_file = 'kakaolist1'
#
# df = pd.read_csv(kakao_file+'.csv', encoding="utf-8")
#
# Date = 'Date'
# User = 'user_name'
# Message = 'text'
#
# df = df[df[Message] != '(이모티콘) ']
#
# pre_user = ''
#
# messages = []
# user_message = ''
#
# for index, user in enumerate(df[User]):
#
#     try:
#         if pre_user !=user:
#             pre_user = user
#             messages.append(user_message)
#             user_message = ''
#         user_message += df[Message][index]
#     except:
#         pass
#
# fp = open('converted.txt', 'a', encoding="utf-8")
#
# for message in messages:
#     if message:
#         fp.write(message.replace(':', '').replace(',', '') + '\n')
