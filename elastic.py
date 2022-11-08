from datetime import datetime

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 지수로그 없애기 위해 소수점 6자리까지만
np.set_printoptions(precision=6, suppress=True)

# url = 'http://ec2-3-19-14-184.us-east-2.compute.amazonaws.com:9200/'
url = 'http://localhost:9200/'


def make_mappings():
    es = Elasticsearch(hosts=[url], basic_auth=('elastic', 'rlagksgh'), )
    index_name = 'chat_bot'

    mappings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 2
        },
        "mappings": {
            "dynamic": "true",
            "_source": {
                "enabled": "true"
            },
            "properties": {
                "member_id": {
                    "type": "integer"
                },
                "we_id": {
                    "type": "integer"
                },
                "Q": {
                    "type": "text"
                },
                "A": {
                    "type": "text"
                },
                "chatvector": {
                    "type": "dense_vector",
                    "dims": 512
                },
                "@timestamp": {
                    "type": "date"
                }
            }
        }
    }

    es.indices.create(index=index_name, body=mappings)


def delete_mapping():
    es = Elasticsearch(hosts=[url], basic_auth=('elastic', 'rlagksgh'), )
    index_name = 'chat_bot'

    es.options(ignore_status=[400, 404]).indices.delete(index=index_name)


def insert_chatdata_es(embedding_result_csv_name, member_id, we_id):
    es = Elasticsearch(hosts=[url], basic_auth=('elastic', 'rlagksgh'), )
    df = pd.read_csv(embedding_result_csv_name)
    index = "chat_bot"
    count = 0
    for temp1, temp, temp2 in zip(df['A'], df['Q'], df['chatvector']):
        # chatvector 에 값을 넣기 위해서 str > replace > list > float 으로 변환.
        list_of_string = temp2.replace('[', '').replace(']', '').split()[0:512]
        list_of_float = list(map(float, list_of_string))
        doc = {
            "member_id": member_id,
            "we_id": we_id,
            "Q": temp,
            "A": temp1,
            "chatvector": list_of_float,
            "@timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')
        }

        count = count + 1

        es.index(index=index, body=doc)


def loadchat(member_id, we_id, textdata):
    es = Elasticsearch(hosts=[url], basic_auth=('elastic', 'rlagksgh'), )
    index = "chat_bot"
    textembeding = model.encode(textdata)
    s_body = {
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match_phrase": {
                                    "member_id": member_id
                                }
                            },
                            {
                                "match_phrase": {
                                    "we_id":  we_id

                                }
                            }
                        ]
                    },
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'chatvector') + 1.0",
                    "params": {"query_vector": textembeding[0:512]}
                }
            }
        }
    }

    res = es.search(index=index, body=s_body)

    if len(res['hits']['hits']) == 0:
        return "챗봇의 데이터가 충분하지 않습니다. 카카오톡 데이터를 넣어주세요!"

    if res['hits']['hits'][0]['_score'] >= 1.7:
        return_sentence = res['hits']['hits'][0]['_source']['A']
    else:
        return_sentence = '미안해요.. 당신의 말을 이해하지 못했어요..'
    return return_sentence


if __name__ == "__main__":
    import time

    start = time.time()  # 시작 시간 저장
    # make_mappings()  # 1번 테이블생성
    # make_chatdata()   #2번 챗봇데이터 테이블에 입력
    # insert_chatdata_es('embeding_result_1.0_2.0.csv', 2, 3)
    # delete_mapping() # 테이블 삭제
    result = loadchat(3,4,'지렸다.')
    print(result)
    print("time :", time.time() - start)
