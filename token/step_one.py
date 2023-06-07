import json
import torch 
import pandas as pd
from transformers import AutoTokenizer

result = []
File_Path = "/home/dmjeong/cls/token/sheet.csv"
Tokenizer_Path = "klue/roberta-base"

data = pd.read_csv(File_Path)
tokenizer = AutoTokenizer.from_pretrained(Tokenizer_Path)

remove_set = {0, 2}

for text in data['용어']: 
    dtx = tokenizer.encode(text)
    dtx_remove_data = [i for i in dtx if i not in remove_set]
    output_str = ', '.join(str(num) for num in dtx_remove_data)
    result.append({'text' : text, 'token' : output_str})

with open('output.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False)