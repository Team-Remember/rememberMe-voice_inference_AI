import pandas as pd
from sentence_transformers import SentenceTransformer

# 한국어 bert 모델 불러오기
from tqdm import tqdm
import re

model = SentenceTransformer('jhgan/ko-sroberta-multitask')


# 카카오톡 파일 전처리
def pretreatment_kakao_file(files):
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
            text = pretreatment_line(text)  # 전처리

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
    text = re.sub('이모티콘', '', before)
    text = re.sub('\n', ' ', text)
    text = re.sub(only_BMP_pattern, ' ', text)
    text = re.sub(r'http\S+', '', text)  # url
    text = re.sub('사진', '', text)
    text = re.sub('음성메시지', '', text)
    # text = re.sub('샵검색 : #', '', text) + ' 이거 검색해 봐'
    text = re.sub('ㅋ', '', text)
    text = re.sub('삭제된 메시지입니다.', '', text)
    text = re.sub(r'파일: [a-zA-Z0-9.?/&=:가-힣]*' + '.' + r'[a-zA-Z0-9.?/&=:가-힣]*', '', text)
    text = re.sub(r'[?/&=#\\:-{}]', '', text)  # 특수문자 제거
    text = re.sub(r'[0-9,]*원을 보냈어요. 송금 받기 전까지 내역 상세화면에서 취소할 수 있어요.', '', text)
    text = re.sub(r'[0-9,]*원을 보냈어요 송금 받기 전까지 내역 상세화면에서 취소할 수 있어요', '', text)
    after = text.replace('[네이버 지도]', '\n[네이버 지도]')

    return after


# input 데이터프레임으로 변형
def make_model_input_form(my_katalk_df):
    df = my_katalk_df
    df['counter'] = 0

    counter = 0

    # 현재 화자와 다음 화자가 다른 경우, counter를 증가시킨다.
    for i in range(1, len(df)):
        current_person = df.iloc[i]['user_name']
        next_person = df.iloc[i - 1]['user_name']
        if current_person != next_person:
            counter = counter + 1
        df.at[i, 'counter'] = counter

    # 화자와 counter를 더해 message_idx를 만든다.
    df['message_idx'] = df.user_name.astype(str) + "-" + df.counter.astype(str)

    # counter와 message_idx로 그룹바이하여 문장을 리스트로 묶는다.
    parsed = pd.DataFrame(df.groupby(['counter', 'message_idx'])['text'].apply(list))

    # 묶은 라인을 하나의 스트링으로 합치고 줄바꿈을 마지막에 붙인다.
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

    # q에 a, a에 q append
    tmp_list = A_list.copy()
    A_list.extend(Q_list)
    Q_list.extend(tmp_list)

    # 반환할 데이터 프레임 만들기
    result_dataframe = pd.DataFrame({'Q': Q_list, 'A': A_list})

    return result_dataframe


def embedding_csv(dataframe, member_id, we_id):
    embeding_list = []
    for temp in tqdm(dataframe['Q']):
        embed_temp = model.encode(temp)
        embeding_list.append(embed_temp)

    # df_embeding = pd.DataFrame(embeding_list)
    dataframe['chatvector'] = embeding_list
    dataframe.to_csv(f'embeding_result_{member_id}_{we_id}.csv', index=None)
    return f'embeding_result_{member_id}_{we_id}.csv'


# input 데이터프레임으로 변형
def make_model_input_form_from_db(db_data):
    db_list = []

    for x in db_data:
        for y in range(1, len(x['data'])):
            current_person = x['data'][y]['nickName']
            next_person = x['data'][y - 1]['nickName']

            if y == 1 and current_person == next_person:
                db_list.append(x['data'][y - 1]['chatText'] + ' ' + x['data'][y]['chatText'])
            elif y == 1 and current_person != next_person:
                db_list.append(x['data'][y - 1]['chatText'])
                db_list.append(x['data'][y]['chatText'])
            elif y > 1 and current_person == next_person:
                before_text = db_list.pop()
                db_list.append(before_text + ' ' + x['data'][y]['chatText'])
            elif y > 1 and current_person != next_person:
                db_list.append(x['data'][y]['chatText'])

    print(db_list)

    db_list = list(filter(None, db_list))  # 빈칸 제거
    Q_list = list()
    A_list = list()

    # Q, A 분류
    for index, sentence in enumerate(db_list):
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

    # q에 a, a에 q append
    tmp_list = A_list.copy()
    A_list.extend(Q_list)
    Q_list.extend(tmp_list)

    # 반환할 데이터 프레임 만들기
    result_dataframe = pd.DataFrame({'Q': Q_list, 'A': A_list})

    return result_dataframe

# if __name__ == '__main__':
#     db_data = [
#             {
#                 "id": 10,
#                 "data": [
#                     {
#                         "nickName": "dd",
#                         "chatText": "안녕녕"
#                     },
#                     {
#                         "nickName": "dd",
#                         "chatText": "안녕녕"
#                     },
#                     {
#                         "nickName": "dd",
#                         "chatText": "안녕녕"
#                     },
#                     {
#                         "nickName": "dd",
#                         "chatText": "안녕녕"
#                     },
#                     {
#                         "nickName": "dd",
#                         "chatText": "화이팅"
#                     }
#                 ],
#                 "memberId": 1,
#                 "opponentId": 2
#             },
#             {
#                 "id": 17,
#                 "data": [
#                     {
#                         "nickName": "남경",
#                         "chatText": "11"
#                     },
#                     {
#                         "nickName": "회은",
#                         "chatText": "2"
#                     },
#                     {
#                         "nickName": "남경",
#                         "chatText": "3"
#                     },
#                     {
#                         "nickName": "남경",
#                         "chatText": "43"
#                     },
#                     {
#                         "nickName": "남경",
#                         "chatText": "5"
#                     },
#                     {
#                         "nickName": "남경",
#                         "chatText": "6"
#                     }
#                 ],
#                 "memberId": 1,
#                 "opponentId": 3
#             }
#         ]
#     make_model_input_form_from_db(db_data)
