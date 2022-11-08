import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('jhgan/ko-sroberta-multitask')


def aichatbot(user_id, we_id, chat_request):
    em_result = model.encode(chat_request)
    df_embeding = pd.read_csv("embeding2.csv", header=None)
    df = pd.read_csv("result_dataframe1.csv")
    co_result = []

    for temp in range(len(df_embeding)):
        data = df_embeding.iloc[temp]
        co_result.append(cosine_similarity([em_result], [data])[0][0])

    df['cos'] = co_result
    df_result = df.sort_values('cos', ascending=False)
    print(df_result)

    if df_result.iloc[0]['cos'] >= 0.79:
        return_sentence = df_result.iloc[0]['Q']
    else:
        return_sentence = '미안해요.. 당신의 말을 이해하지 못했어요..'
    return return_sentence
