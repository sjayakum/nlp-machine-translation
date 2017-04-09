


file_hindi = open('../IITB.en-hi.hi','r',encoding='utf-8')

hindi_corpus = file_hindi.read()

# print(hindi_corpus.split('\n')[1000:1050])


file_english = open('../IITB.en-hi.en','r',encoding='utf-8')

english_corpus = file_english.read()






# print(english_corpus.split('\n')[1000:1050])


temp_english_corpus = english_corpus.split('\n')[:5000]
temp_hindi_corpus = hindi_corpus.split('\n')[:5000]

# print('Temp English Corpus',temp_english_corpus)

import pickle

english_model = pickle.load(open('english_model_new.w2v','rb'))
hindi_model = pickle.load(open('hindi_model_new.w2v','rb'))

# from gensim.models import Word2Vec
#
#
# english_model = Word2Vec.load('english_w2v.p')
# hindi_model = Word2Vec.load('hindi_w2v.p')
#



from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, Reshape, SimpleRNN



input_seq = 10
output_seq = 10

input_w2v = 100
output_w2v = 100

import numpy as np

X_train = np.zeros((4500,input_seq,input_w2v))
y_train = np.zeros((4500,output_seq,output_w2v))

i = 0

for each_sentence in temp_english_corpus[:4500]:
    j = 0
    for each_word in each_sentence.split(' '):
        try:
            X_train[i][j] = english_model.wv[each_word]
        except:
            print(each_word,'not found during training - English')
        finally:
            j+=1

    i+=1


i=0
for each_sentence in temp_hindi_corpus[:4500]:
    j = 0
    for each_word in each_sentence.split(' '):
        try:
            y_train[i][j] = hindi_model.wv[each_word]
        except:
            print(each_word,'not found during training - Hindi')
        finally:
            j+=1

    i+=1



X_test = np.zeros((1,input_seq,input_w2v))
y_test = np.zeros((1,output_seq,output_w2v))

model = Sequential()
model.add(LSTM(64, input_shape=(input_seq,input_w2v),return_sequences=True))
model.add(LSTM(512,unroll=True,go_backwards=True))
model.add(Dense(output_seq*output_w2v,activation='softmax'))
model.add(Reshape((output_seq, output_w2v)))
model.summary()
model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(X_train, y_train,
                  batch_size=5, epochs=5,
                  verbose=1,shuffle=False)

test_index = 4600
j = 0
for each_word in temp_english_corpus[test_index].split(' '):
    try:
        X_test[0][j] = english_model.wv[each_word]
    except:
        print(each_word, 'not found during testing stage -English')
    finally:
        j += 1


myoutput = model.predict(X_test,batch_size =1)

final_answer = ""

for each_vector in myoutput[0]:
    final_answer += ' ' + hindi_model.similar_by_vector(each_vector,1)[0][0]


print('Predicted Answer',final_answer)
print('Actual Answer',temp_hindi_corpus[test_index])





