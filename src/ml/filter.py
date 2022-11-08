from transformers import TextClassificationPipeline, BertTokenizerFast, TFBertForSequenceClassification
import os

MODEL_NAME = 'abuse_filtering_model'
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


# text_or_voice 문자 : 0, 음성 : 1
def abuse_filtering(text, text_or_voice):
    # predict
    preds_list = text_classifier(text)[0]
    for x in preds_list[:9]:
        if x['score'] > 0.8 and text_or_voice == 0:
            return x['label'] + '에 대한 내용이 담겨있습니다. 다른 문장을 입력해주세요!'
        elif x['score'] > 0.8 and text_or_voice == 1:
            return x['label'] + '에 대한 내용이 담겨있습니다. 다른 문장을 말씀해주세요!'
    return None

# if __name__ == '__main__':
#     start = time.time()
#     abuse_filtering("바보 멍청이")
#     print(time.time() - start)
