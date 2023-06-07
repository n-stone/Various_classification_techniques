import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt

class SentimentPredictor:
    def __init__(self, model_path, tokenizer_path):
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.model = load_model(model_path)
        self.stopwords = ['의', '에', '에서', '을', '를', '이', '가', '에게', '한테', '와', '과', '과 같은', '은', '는', '라는', '들의', '조차', '따위의', '도', '만', '까지', '부터', '까지만', '마저', '조차', '든지', '나', '니', '다가', '든지', '이라도', '이나', '이든지', '이라고', '이며', '이든가', '이라며', '이든가', '이야말로', '이어서', '인가', '일지라도', '일까', '지말고', '지마', '처럼', '커녕', '한테', '하고', '하면서', '하면서도', '해서', '해도']
        self.max_len = 30
        self.okt = Okt()

    def preprocess(self, sentence):
        pattern = '[^A-Za-z0-9가-힣 ! ?]'
        sentence = re.sub(pattern=pattern, repl='', string=sentence)
        pattern = '<[^>]*>' 
        sentence = re.sub(pattern=pattern, repl='', string=sentence)
        pattern = '!{1,}' 
        sentence = re.sub(pattern=pattern, repl='!', string=sentence)
        pattern = '\\?{1,}'
        sentence = re.sub(pattern=pattern, repl='?', string=sentence)
        pattern = '~{1,}'
        sentence = re.sub(pattern=pattern, repl='~', string=sentence)        
        sentence = self.okt.morphs(sentence, stem=True) 
        sentence = [word for word in sentence if not word in self.stopwords] 
        return sentence

    def predict_sentiment(self, sentence):
        sentence = self.preprocess(sentence)
        encoded = self.tokenizer.texts_to_sequences([sentence]) # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen=self.max_len) # 패딩
        score = float(self.model.predict(pad_new)) # 예측
        if score > 0.5:
            print("{:.2f}% 확률로 욕설입니다.".format(score * 100))
            print(sentence, "\n")
        else:
            print("{:.2f}% 확률로 욕설이 아닙니다.".format((1 - score) * 100))
            print(sentence, "\n")
