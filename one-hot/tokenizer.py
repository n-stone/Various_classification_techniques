from konlpy.tag import Okt  

okt = Okt()   

def one_hot_encoding(word, word_to_index):
    one_hot_vector = [0]*(len(word_to_index))
    index = word_to_index[word]
    one_hot_vector[index] = 1
    return one_hot_vector

while 1:
    try:
        text = input("입력 값 : ")
        tokens = okt.morphs(text) 
        print(tokens)
        word_to_index = {word : index for index, word in enumerate(tokens)}
        one_hot = one_hot_encoding("성현범", word_to_index)
        if 1 in one_hot:
            print("데이터가 존재합니다.")
    except KeyError:
        print("데이터가 존재하지 않습니다.")
        