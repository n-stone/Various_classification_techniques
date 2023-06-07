import torch 
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

remove_set = {0, 2}

# ss = tokenizer.encode("text")
# print(ss)

test_enc_1 = tokenizer.decode([13381, 2603, 23086])
test_1 = tokenizer.encode(test_enc_1)

# print(test_enc_1)
# print(test_1)

class TokenizerTask:
    def __init__(self, sentence):
        self._sentence = sentence
        self._enc = tokenizer.encode(self._sentence)
        self._dec = tokenizer.decode(self._enc)
        self._enc_remove_data = [i for i in self._enc if i not in remove_set]
        self.classification = self.classification()
        # self._fc1 = set(self._enc_remove_data) & set(test_1)
         
    def __str__(self):
        return f'label result > {self.classification}'

    def classification(self):
        try:  
            if set(self._enc_remove_data) & set(test_1):
                text = "코딩 데이터 입니다."

            else:
                text = "코딩 외 데이터 입니다."
        except IndexError:
            text = "Error"   
        return text


tttt = "안녕하세요. 저는 코코블에 관심이 많습니다. 그래서 코딩 블럭을 연습하고 있습니다."
tttt1 = "안녕하세요. 성현범에 관심이 있어요."

sentence = tttt1 
test = TokenizerTask(sentence)
sss = test.classification
print(sss)
result = {}
result["aaaa"] = sss
print(result)
