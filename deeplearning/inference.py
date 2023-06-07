import re
import pickle
import pandas as pd
import tensorflow as tf
from konlpy.tag import Okt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

okt = Okt()
with open('/home/dmjeong/cls/deeplearnin/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model_path = "/home/dmjeong/cls/deeplearnin/best_model.h5"
model = load_model(model_path)
max_length = 60

stopwords = ['의', '에', '에서', '을', '를', '이', '가', '에게', '한테', '와', '과', '과 같은', '은', '는', '라는', '들의', '조차', '따위의',
              '도', '만', '까지', '부터', '까지만', '마저', '조차', '든지', '나', '니', '다가', '든지', '이라도', '이나', '이든지', '이라고',
              '이며', '이든가', '이라며', '이든가', '이야말로', '이어서', '인가', '일지라도', '일까', '지말고', '지마', '처럼', '커녕', '한테',
              '하고', '하면서', '하면서도', '해서', '해도']

def sentiment_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True) 
  new_sentence = [word for word in new_sentence if not word in stop_words]
  encoded = tokenizer.texts_to_sequences([new_sentence]) 
  pad_new = pad_sequences(encoded, maxlen = max_length) 
  score = float(model.predict(pad_new)) 
  if(score > 0.7):
    print("{:.2f}% 확률로 코딩 리뷰입니다.\n".format(score * 100))
  else:
    print("{:.2f}% 확률로 코딩 외 리뷰입니다.\n".format((1 - score) * 100))
    
while 1:
  text = input("입력 값 : ")
  sentiment_predict(text)