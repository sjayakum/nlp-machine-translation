import codecs
import nltk
import pickle

from nltk.tokenize import sent_tokenize, word_tokenize

english_hindi_dict = dict()
english_hindi_multiword = dict()

IITB_DICTIONARY_FILE_PATH = 'UW-Hindi_Dict-20131003.txt'
HINDENCORP_PATH = 'hindencorp05.p'

with codecs.open(IITB_DICTIONARY_FILE_PATH, 'r', 'utf-8') as dictionary_file:
    for index, line in enumerate(dictionary_file):
        end_index = line.find("]")
        if(end_index == -1):
            print(line)
            continue

        hindi_phrase = line[1:end_index]
        quote_start_index = line.index('"')
        quote_end_index = line.find('"', quote_start_index+1)
        if(quote_end_index == -1):
            print(line)
            continue

        english_phrase_with_meaning = line[quote_start_index+1:quote_end_index]
        meaning_index = english_phrase_with_meaning.find('(')
        if(meaning_index != -1):
            english_phrase = english_phrase_with_meaning[:meaning_index]
        else:
            english_phrase = english_phrase_with_meaning
        # print(hindi_phrase, english_phrase)
        if(english_phrase.find(' ') == -1):
            # Single Word Phrase
            english_hindi_dict[english_phrase] = hindi_phrase
        else:
            english_hindi_multiword[english_phrase] = hindi_phrase

        # if(index%100 == 0):
        #     print(index)


model = pickle.load(open(HINDENCORP_PATH, 'rb'))
english = model["X"]
hindi = model["Y"]
for index, english_phrase in enumerate(english):
    hindi_phrase = hindi[index]
    if(english_phrase.find(' ') == -1):
        english_hindi_dict[english_phrase] = hindi_phrase
    else:
        english_hindi_multiword[english_phrase] = hindi_phrase

print(len(english_hindi_dict))

obj = pickle.load(open("test_data.p", "rb"))

absent_words = set()
for keys in obj.keys():
    sentence = obj[keys]["I"]
    tokenized = word_tokenize(sentence)
    translated_sentence = list()
    for word in tokenized:
        if word in english_hindi_dict:
            translated_sentence.append(english_hindi_dict[word])
        elif not word.isalnum():
            translated_sentence.append(word)
        else:
            absent_words.add(word)

    obj[keys]["X"] = ' '.join(translated_sentence)

# print(absent_words) 
pickle.dump(obj, open('dictionary_test_data.p', 'wb'))




