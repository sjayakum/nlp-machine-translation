


import pickle
obj =  pickle.load(open('hindencorp05.p','rb'))


english_corpus = obj['X']

hindi_corpus = obj['Y']

english_corpus = '\n'.join(english_corpus)
hindi_corpus = '\n'.join(hindi_corpus)


# print(english_corpus.split('\n')[1000:1050])


temp_english_corpus = english_corpus.split('\n')[:5000]
temp_hindi_corpus = hindi_corpus.split('\n')[:5000]

# print('Temp English Corpus',temp_english_corpus)
import gensim.models.word2vec as w2v
english_model = w2v.Word2Vec.load('english_model_full.w2v')
hindi_model = w2v.Word2Vec.load('hindi_model_full.w2v')

# from gensim.models import Word2Vec
#
#
# english_model = Word2Vec.load('english_w2v.p')
# hindi_model = Word2Vec.load('hindi_w2v.p')
#



from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, Reshape, SimpleRNN



input_seq = 25
output_seq = 25

input_w2v = 100
output_w2v = 100

import numpy as np

X_train = np.zeros((4500,input_seq,input_w2v))
y_train = np.zeros((4500,output_seq,output_w2v))

i = 0

error_counter = 0

for each_sentence in temp_english_corpus[:4500]:
    j = 0
    for each_word in each_sentence.split(' '):
        try:
            X_train[i][j] = english_model.wv[each_word]
        except:
            print(each_word,'not found during training - English')
            error_counter+=1
        finally:
            j+=1

    i+=1


print(error_counter,'Error Counter English')




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
model.add(LSTM(128))
model.add(Dense(output_seq*output_w2v,activation='softmax'))
model.add(Reshape((output_seq, output_w2v)))
model.summary()
model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(X_train, y_train,
                  batch_size=25, epochs=5,
                  verbose=1)

test_index  = 2400
j = 0
for each_word in temp_english_corpus[test_index].split(' '):
    try:
        X_test[0][j] = english_model.wv[each_word]
    except:
        print(each_word, 'not found during testing stage -English')
    finally:
        j += 1


myoutput = model.predict(X_test,batch_size =1)

final_answer  = ""

for each_vector in myoutput[0]:
    final_answer+= ' ' + hindi_model.similar_by_vector(each_vector,1)[0][0]


print('Predicted Answer',final_answer)
print('Actual Answer',temp_hindi_corpus[test_index])





