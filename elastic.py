from datetime import datetime
from elasticsearch import Elasticsearch, helpers
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import csv
import numpy as np

# model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
df = pd.read_csv('embeding_result_1.0_2.0.csv')
# 지수로그 없애기 위해 소수점 6자리까지만
np.set_printoptions(precision=6, suppress=True)

# url = 'http://ec2-3-19-14-184.us-east-2.compute.amazonaws.com:9200/'
url = 'http://localhost:9200/'


def make_mappings():
    es = Elasticsearch(hosts=[url], basic_auth=('elastic', 'rlagksgh'), )
    index_name = 'chatdata1'

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
                    "type": "text"
                },
                "we_id": {
                    "type": "text"
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
    index_name = 'chatdata1'

    es.options(ignore_status=[400, 404]).indices.delete(index=index_name)


def make_chatdata():
    es = Elasticsearch(hosts=[url], basic_auth=('elastic', 'rlagksgh'), )

    index = "chatdata1"
    count = 0
    for temp1, temp in zip(df['A'], df['Q']):
        t = model.encode(temp)

        doc = {
            "member_id": 1,
            "we_id": 2,
            "Q": temp,
            "A": temp1,
            "chatvector": t[0:512],
            "@timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')
        }

        count = count + 1

        es.index(index=index, body=doc)


def make_chatdata_nk(embedding_result_csv_name, memberId, weId):
    es = Elasticsearch(hosts=[url], basic_auth=('elastic', 'rlagksgh'))

    df = pd.read_csv(embedding_result_csv_name)
    index = "chatdata1"
    count = 0
    for temp1, temp, temp2 in zip(df['A'], df['Q'], df['chatvector']):
        doc = {
            "member_id": 1,
            "we_id": 2,
            "Q": temp,
            "A": temp1,
            "chatvector": temp2[0:512],
            "@timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')
        }

        count = count + 1

        es.index(index=index, body=doc)


def loadchat(textdata):
    es = Elasticsearch(hosts=[url], basic_auth=('elastic', 'rlagksgh'), )
    index = "chatdata1"
    textembeding = model.encode(textdata)
    s_body = {
        "query": {
            "script_score": {

                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'chatvector') + 1.0",
                    "params": {"query_vector": textembeding[0:512]}
                }
            }
        }
    }

    res = es.search(index=index, body=s_body)
    return res


if __name__ == "__main__":
    import time

    start = time.time()  # 시작 시간 저장
    # make_mappings()  # 1번 테이블생성
    # make_chatdata()   #2번 챗봇데이터 테이블에 입력
    make_chatdata_nk('embeding_result_1.0_2.0.csv', 1.0, 2.0)
    # delete_mapping() # 테이블 삭제
    # result = loadchat('안녕')
    # print(result)
    # print(result['hits']['hits'][random.randint(0,len(result['hits']['hits']))]['_source']['A'])

    print("time :", time.time() - start)
