import json
from transformers import AutoTokenizer

class TokenizerTask:
    def __init__(self, sentence, json_path, token_path):
        self._sentence = sentence
        self._tokenizer = AutoTokenizer.from_pretrained(token_path)
        self._remove_set = {0, 2}
        self._json_path = json_path
        self._token_path = token_path
        self._enc = self._tokenizer.encode(self._sentence)
        self._dec = self._tokenizer.decode(self._enc)
        self._enc_remove_data = [i for i in self._enc if i not in self._remove_set]
        self._token = self._get_token()
        self._classification = self.classification()

    def _get_token(self):
        with open(self._json_path, 'r') as f:
            json_data = json.load(f)

        key_count = 0
        for item in json_data:
            if 'token' in item:
                key_count += 1

        token_list = []
        result_list = []

        for i in range(key_count):
            token = json_data[i]["token"]
            token_list.append(token)

        for item in token_list:
            result_list.extend(map(int, item.split(', ')))

        token_enc = self._tokenizer.decode(result_list)
        token = self._tokenizer.encode(token_enc)
        return token

    def classification(self):
        try:  
            if set(self._enc_remove_data) & set(self._token):
                text = "코딩 데이터 입니다."
            else:
                text = "코딩 외 데이터 입니다."
        except IndexError:
            text = "Error"   
        return text
while 1:
    text = input("입력 :")
    json_path = "/home/dmjeong/cls/token/output.json"
    token_path = "klue/roberta-base"

    task = TokenizerTask(text, json_path, token_path)
    result = task.classification()
    print(result)