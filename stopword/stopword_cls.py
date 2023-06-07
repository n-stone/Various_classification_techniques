from konlpy.tag import Okt
from collections import Counter
import re

okt = Okt()
stop_words = ['코딩', '코코블']

text = "안녕하세요. 저는 코코블에 관심이 많습니다. 그래서 코딩 블럭을 연습하고 있습니다."
 
morphs = okt.morphs(text)
filtered_morphs = [morph for morph in morphs if morph in stop_words]

print(filtered_morphs)                                                                                                                                                                                                                                     