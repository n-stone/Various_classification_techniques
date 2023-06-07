import config
from ethics_inference import SentimentPredictor

model_path = config.MODEL_PATH
tokenizer_path = config.TOKENIZER_PATH

predictor = SentimentPredictor(model_path, tokenizer_path)

while 1:
    sentence = input("입력 : ")
    predictor.predict_sentiment(sentence)
