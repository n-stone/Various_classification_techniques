import json
import torch 
from transformers import AutoTokenizer

Json_Path = "/home/dmjeong/cls/token/output.json"
Token_Path = "klue/roberta-base"

tokenizer = AutoTokenizer.from_pretrained(Token_Path)

remove_set = {0, 2}

with open(Json_Path, 'r') as f:
    json_data = json.load(f)

key_count = 0
for item in json_data:
    if 'token' in item:
        key_count += 1

token_list = []
result_list = []

for i in range(100):
    token = json_data[i]["token"]
    token_list.append(token)

for item in token_list:
    result_list.extend(map(int, item.split(', ')))

test_enc_1 = tokenizer.decode(result_list)
test_1 = tokenizer.encode(test_enc_1)

class TokenizerTask:
    def __init__(self, sentence):
        self._sentence = sentence
        self._enc = tokenizer.encode(self._sentence)
        self._dec = tokenizer.decode(self._enc)
        self._enc_remove_data = [i for i in self._enc if i not in remove_set]
        self.classification = self.classification()

    def classification(self):
        try:  
            if set(self._enc_remove_data) & set(test_1):
                text = "코딩 데이터 입니다."
            else:
                text = "코딩 외 데이터 입니다."
        except IndexError:
            text = "Error"   
        return text

while 1:
    sentence = input("입력 :")
    test = TokenizerTask(sentence)
    sss = test.classification
    print(sss)
    # result = {}
    # result["aaaa"] = sss
    # print(result)