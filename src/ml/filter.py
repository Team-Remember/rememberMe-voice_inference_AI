from transformers import TextClassificationPipeline, BertTokenizerFast, TFBertForSequenceClassification
import os, time

MODEL_NAME = 'abuse-filtering-model'
MODEL_SAVE_PATH = os.path.join("src/model", MODEL_NAME)

# Load Fine-tuning model
loaded_tokenizer = BertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
loaded_model = TFBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH, from_pt=True)

text_classifier = TextClassificationPipeline(
    tokenizer=loaded_tokenizer,
    model=loaded_model,
    framework='tf',
    return_all_scores=True
)


def abuse_filtering(text, text_or_voice):
    # predict
    preds_list = text_classifier(text)[0]
    print(preds_list)
    for x in preds_list[:9]:
        if x['score'] > 0.8 and text_or_voice == 0:
            return x['label'] + '에 대한 내용이 담겨있습니다. 다른 문장을 입력해주세요!'
        elif x['score'] > 0.8 and text_or_voice == 1:
            return x['label'] + '에 대한 내용이 담겨있습니다. 다른 문장을 말씀해주세요!'
    return None


# from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
# import time
#
# model_name = 'smilegate-ai/kor_unsmile'
#
# model = BertForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# pipe = TextClassificationPipeline(
#     model=model,
#     tokenizer=tokenizer,
#     device=0,  # cpu: -1, gpu: gpu number
#     top_k=None,
#     function_to_apply='sigmoid'
# )
#
#
# #
# def abuse_filtering(text, text_or_voice):
#     pipe_list = pipe(text)[0][0]  # 욕설 필터링 결과
#     print(pipe_list)
#     text = None
#     if pipe_list['label'] != 'clean' and pipe_list['score'] > 0.7:
#         text = pipe_list['label']
#     return_text = None
#     if text is not None and text_or_voice == 0:
#         return_text = text + '에 대한 내용이 담겨있습니다. 다른 문장을 입력해주세요!'
#     elif text is not None and text_or_voice == 1:
#         return_text = text + '에 대한 내용이 담겨있습니다. 다른 문장을 말씀해주세요!'
#     return return_text


if __name__ == '__main__':
    start = time.time()
    print(abuse_filtering("바보 멍청이", 1))
    print(time.time() - start)
