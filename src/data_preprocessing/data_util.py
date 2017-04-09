import pickle
from collections import Counter
import codecs

def create_dataset(en_sentences, hi_sentences):
    en_vocab_dict = Counter(word.strip(',." ;:)(][?!') for sentence in en_sentences for word in sentence.split())
    hi_vocab_dict = Counter(
        word.strip(',." ;:)(|][?!<>a-zA-Z') for sentence in hi_sentences for word in sentence.split())

    en_vocab = list(map(lambda x: x[0], sorted(en_vocab_dict.items(), key=lambda x: -x[1])))
    hi_vocab = list(map(lambda x: x[0], sorted(hi_vocab_dict.items(), key=lambda x: -x[1])))

    # Limit the vocabulary size. Consider only the top 20,000 and 30,000 words respectively
    en_vocab = en_vocab[:20000]
    hi_vocab = hi_vocab[:30000]

    # Build a Word to Index Dictionary for English
    start_idx = 2
    en_word2idx = dict([(word, idx + start_idx) for idx, word in enumerate(en_vocab)])
    en_word2idx['<ukn>'] = 0  # Unknown words
    en_word2idx['<pad>'] = 1  # Padding word

    # Build an Index to Word Dictionary for English using the already created Word to Index Dictionary
    en_idx2word = dict([(idx, word) for word, idx in en_word2idx.items()])

    # Build a Word to Index Dictionary for Hindi
    start_idx = 4
    hi_word2idx = dict([(word, idx + start_idx) for idx, word in enumerate(hi_vocab)])
    hi_word2idx['<ukn>'] = 0  # Unknown
    hi_word2idx['<go>'] = 1
    hi_word2idx['<eos>'] = 2  # End of sentence
    hi_word2idx['<pad>'] = 3  # Padding

    # Build an Index to Word Dictionary for Hindi using the already created Word to Index Dictionary
    hi_idx2word = dict([(idx, word) for word, idx in hi_word2idx.items()])

    # Encode words in senteces by their index in Vocabulary
    x = [[en_word2idx.get(word.strip(',." ;:)(][?!'), 0) for word in sentence.split()] for sentence in en_sentences]
    y = [[hi_word2idx.get(word.strip(',." ;:)(][?!'), 0) for word in sentence.split()] for sentence in hi_sentences]

    X = []
    Y = []
    for i in range(len(x)):
        n1 = len(x[i])
        n2 = len(y[i])
        n = n1 if n1 < n2 else n2
        if abs(n1 - n2) < 0.3 * n:
            if n1 <= 20 and n2 <= 20:
                X.append(x[i])
                Y.append(y[i])

    return X, Y, en_word2idx, en_idx2word, en_vocab, hi_word2idx, hi_idx2word, hi_vocab

def save_dataset(filepath, obj):
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp, -1)

def read_dataset(filepath):
    with open(filepath, 'rb') as fp:
        return pickle.load(fp)
