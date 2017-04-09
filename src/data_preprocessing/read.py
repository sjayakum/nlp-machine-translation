import pickle
from collections import Counter
import codecs


def read_sentences(filepath):
    X = []
    Y = []
    with codecs.open(filepath, encoding = "utf-8", mode = "r") as fp:
        for sentence in fp:
            splits = sentence.split("\t")
            X.append(splits[3].strip())
            Y.append(splits[4].strip())
    return X, Y

