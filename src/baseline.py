

import pickle
obj =  pickle.load(open('hindencorp05.p','rb'))


english_corpus = obj['X']

hindi_corpus = obj['Y']




temp_english_corpus = english_corpus[:5000]
temp_hindi_corpus = hindi_corpus[:5000]




temp_hindi_transliteral_corpus = open('trasnliterate.txt','r',encoding='utf-8').read()
temp_hindi_transliteral_corpus1 = '\n'.join(temp_hindi_transliteral_corpus)
full_vocab = set(temp_hindi_transliteral_corpus1)
char_to_ix2 = {ch: i for i, ch in enumerate(full_vocab)}
ix_to_char2= {i: ch for i, ch in enumerate(full_vocab)}


temp_english_corpus1 = '\n'.join(temp_english_corpus)
full_vocab = set(temp_english_corpus1)
char_to_ix1 = {ch: i for i, ch in enumerate(full_vocab)}
ix_to_char1 = {i: ch for i, ch in enumerate(full_vocab)}


print(char_to_ix1)
print(ix_to_char1)
print('*'*20)

temp_hindi_corpus = 0

temp_hindi_corpus = temp_hindi_transliteral_corpus.split('\n')

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Reshape,LSTM


input_seq = 300 #CHARS
output_seq = 300 #CHARS

input_w2v = 100 #VOCAB SIZE OF ENGLISH
output_w2v = 100 #VOCAB SIZE OF HINDI

import numpy as np

X_train = np.zeros((4500,input_seq,input_w2v))
y_train = np.zeros((4500,output_seq,output_w2v))

i = 0

for each_sentence in temp_english_corpus[:4500]:
    j  = 0

    for each_char in each_sentence:
        try:
            ohe_index = char_to_ix1[each_char]
            X_train[i][j][ohe_index] = 1
            j += 1
        except:
            print(each_char,'not found during training - English')
            j += 1

    i+=1


i=0

error_counter =0

for each_sentence in temp_hindi_corpus[:4500]:
    j = 0
    for each_char in each_sentence:
        try:
           ohe_index = char_to_ix2[each_char]
           y_train[i][j][ohe_index] = 1
           j += 1
        except:
            print(each_char,'not found during training - Hindi')
            error_counter +=1
            j += 1

    i+=1


print(error_counter,'ERROR COUNTER!!')



X_test = np.zeros((1,input_seq,input_w2v))
y_test = np.zeros((1,output_seq,output_w2v))

model = Sequential()
model.add(LSTM(64, input_shape=(input_seq,input_w2v)))
model.add(Dense(output_seq*output_w2v,activation='softmax'))
model.add(Reshape((output_seq, output_w2v)))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])
model.fit(X_train, y_train,
                  batch_size=32, epochs=1,
                  verbose=1,shuffle=False)

test_index = 1500
j = 0


for each_char in each_sentence:
    try:
        ohe_index = char_to_ix1[each_char]
        X_test[0][j][ohe_index] = 1
    except:
        print(each_char,'not found during training - English')
    finally:
        j+=1




myoutput = model.predict(X_test,batch_size =1)

final_answer = ""

for each_vector in myoutput[0]:
    arg_max_index = np.argmax(each_vector)
    final_answer += ix_to_char2[arg_max_index]


print('Predicted Answer',final_answer)
print('Actual Answer',temp_hindi_corpus[test_index])



