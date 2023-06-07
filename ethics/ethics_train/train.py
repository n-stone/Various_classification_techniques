import config
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

data = pd.read_csv(config.DATA_PATH)

# 데이터셋 분할
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 토큰화
tokenizer = Tokenizer(num_words=10000)  # 가장 빈도가 높은 10,000개의 단어만 사용합니다.
tokenizer.fit_on_texts(train_data['sentence'])

# 시퀀스 변환
X_train = tokenizer.texts_to_sequences(train_data['sentence'])
X_test = tokenizer.texts_to_sequences(test_data['sentence'])

# 시퀀스 패딩
maxlen = 100  # 문장의 최대 길이
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# 레이블 변환
y_train = train_data['label'].values
y_test = test_data['label'].values

# 모델 생성
model = keras.Sequential()
model.add(Embedding(input_dim=10000, output_dim=16, input_length=maxlen))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

mc = ModelCheckpoint(config.MODELCHECKPOINT_PATH, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 훈련
history = model.fit(X_train, y_train, epochs=100, callbacks=[mc], batch_size=32, validation_split=0.2)
