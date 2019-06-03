#!/usr/bin/env python
# coding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk import word_tokenize
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
import json
import pickle
import numpy as np

import gensim.models as g
import gensim.utils

def import_model():
    model = g.Doc2Vec.load("enwiki_dbow/doc2vec.bin")
    return model

def getRawData():
    print('loading data...')
    with open('data/train.json', 'r') as f:
        datastore = json.load(f)
    with open('data/test.json', 'r') as f:
        datastore_test = json.load(f)
    print('loading success!!!')

    # review_list = []
    # label_lists = []
    # for key, value in datastore.items():
    #     review_list.append(key)
    #     label_lists.append(value)

    X_train = []
    y_train = []
    for key, value in datastore.items():
        X_train.append(key)
        y_train.append(value)

    X_test = []
    y_test = []
    for key, value in datastore_test.items():
        X_test.append(key)
        y_test.append(value)

    # X_train, X_test, y_train, y_test = train_test_split(review_list, label_lists, test_size=0.2)
    print('size of training lists are {}'.format(len(X_train)))
    return X_train, X_test, y_train, y_test


def doc2vec(X_train, X_test, y_train, y_test):
    X_train_vector = []
    X_test_vector = []

    model = import_model()

    print('vectorize data...')

    for line in X_train:
        stemmed = gensim.utils.simple_preprocess(line)
        vec = model.infer_vector(stemmed)
        X_train_vector.append(vec)

    X_train_vector = np.array(X_train_vector)

    for line in X_test:
        stemmed = gensim.utils.simple_preprocess(line)
        vec = model.infer_vector(stemmed)
        X_test_vector.append(vec)

    X_test_vector = np.array(X_test_vector)

    print('the final doc2vec train matrix with shape {}'.format(X_train_vector.shape))
    print('the final doc2vec test matrix with shape {}'.format(X_test_vector.shape))


    with open('data/X_train_doc2vec_rest', 'wb') as fp:
        pickle.dump(X_train_vector, fp)
    with open('data/X_test_doc2vec_rest', 'wb') as fp:
        pickle.dump(X_test_vector, fp)
    with open('data/y_train_doc2vec_rest', 'wb') as fp:
        pickle.dump(y_train, fp)
    with open('data/y_test_doc2vec_rest', 'wb') as fp:
        pickle.dump(y_test, fp)
    print('The test and training data are dumped in X_train_doc2vec_rest, X_test_doc2vec_rest, y_train_doc2vec_rest, y_test_doc2vec_rest')


if __name__ == '__main__':
    # filename = './data/train.txt'
    X_train, X_test, y_train, y_test = getRawData()
    doc2vec(X_train, X_test, y_train, y_test)
