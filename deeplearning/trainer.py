import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from konlpy.tag import Okt

class Trainer:
    def __init__(self):
        self.okt = Okt()
        self.tokenizer = Tokenizer()
        self.max_len = 60
        self.vocab_size = None
        self.model = None

    def preprocess_data(self, data_path):
        data = pd.read_csv(data_path)
        data.drop_duplicates(subset=['sentence'], inplace=True)

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=4097)

        train_data.drop_duplicates(subset=['sentence'], inplace=True)
        train_data['sentence'] = train_data['sentence'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
        train_data['sentence'].replace('', np.nan, inplace=True)
        train_data = train_data.dropna(how='any')

        test_data.drop_duplicates(subset=['sentence'], inplace=True)
        test_data['sentence'] = test_data['sentence'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
        test_data['sentence'].replace('', np.nan, inplace=True)
        test_data = test_data.dropna(how='any')

        stopwords = ['의', '에', '에서', '을', '를', '이', '가', '에게', '한테', '와', '과', '과 같은', '은', '는', '라는', '들의', '조차', '따위의',
                     '도', '만', '까지', '부터', '까지만', '마저', '조차', '든지', '나', '니', '다가', '든지', '이라도', '이나', '이든지', '이라고',
                     '이며', '이든가', '이라며', '이든가', '이야말로', '이어서', '인가', '일지라도', '일까', '지말고', '지마', '처럼', '커녕', '한테',
                     '하고', '하면서', '하면서도', '해서', '해도']

        train_data['tokenized'] = train_data['sentence'].apply(self.okt.morphs)
        train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
        test_data['tokenized'] = test_data['sentence'].apply(self.okt.morphs)
        test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

        self.tokenizer.fit_on_texts(train_data['tokenized'].values)

        threshold = 2
        total_cnt = len(self.tokenizer.word_index)
        rare_cnt = 0
        total_freq = 0
        rare_freq = 0

        for key, value in self.tokenizer.word_counts.items():
            total_freq = total_freq + value
            if value < threshold:
                rare_cnt = rare_cnt + 1
                rare_freq = rare_freq + value

        self.vocab_size = total_cnt - rare_cnt + 2
        self.tokenizer = Tokenizer(self.vocab_size, oov_token='OOV')
        self.tokenizer.fit_on_texts(train_data['tokenized'].values)

        with open('tokenizer.pickle', 'wb') as handle:
          pickle.dump(self.tokenizer, handle)

        X_train = self.tokenizer.texts_to_sequences(train_data['tokenized'].values)
        X_test = self.tokenizer.texts_to_sequences(test_data['tokenized'].values)

        X_train = pad_sequences(X_train, maxlen=self.max_len)
        X_test = pad_sequences(X_test, maxlen=self.max_len)

        y_train = train_data['label'].values
        y_test = test_data['label'].values

        return X_train, y_train, X_test, y_test

    def build_model(self, embedding_dim=100, hidden_units=128):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, embedding_dim))
        self.model.add(Bidirectional(LSTM(hidden_units)))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    def train_model(self, X_train, y_train, epochs=15):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

        self.model.fit(X_train, y_train, epochs=epochs, callbacks=[es, mc], batch_size=256, validation_split=0.2, shuffle=True)

# Example usage:
tr = Trainer()
X_train, y_train, X_test, y_test = tr.preprocess_data("/home/dmjeong/source/regex/test_result.csv")
tr.build_model()
tr.train_model(X_train, y_train, epochs=15)