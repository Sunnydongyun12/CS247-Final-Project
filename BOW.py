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

def import_model():
    file = open("model/vectorizer.p",'rb')
    vectorizer = pickle.load(file)
    file = open("model/transformer.p",'rb')
    transformer = pickle.load(file)
    return vectorizer, transformer

def getRawData(filename):
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


def bow(X_train, X_test, y_train, y_test):
    print('stemming data...')
    stemmer = PorterStemmer()
    X_train_stem = []
    X_test_stem = []
    for line in X_train:
        temp = []
        words = word_tokenize(line) 
        for w in words:
            w = stemmer.stem(w)
            temp.append(w)
        X_train_stem.append(" ".join(temp))

    for line in X_test:
        temp = []
        words = word_tokenize(line) 
        for w in words:
            w = stemmer.stem(w)
            temp.append(w)
        X_train_stem.append(" ".join(temp))
    print('stemming success!!!')

    vectorizer, tfidf_transformer = import_model()

    X_train_stem = X_train
    X_test_stem = X_test
    print('vectorize data...')

    # vectorizer = CountVectorizer(stop_words='english', token_pattern=r'\b[^\d\W_]+\b', min_df=3)
    # training_vectors = vectorizer.fit_transform(X_train_stem)

    training_vectors = vectorizer.transform(X_train_stem)
    testing_vectors = vectorizer.transform(X_test_stem)
    # print sample vocabulary
    # vectorizer.get_feature_names()
    print('vectorize success!!!')

    print('transform data into tfidf matrix...')
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(training_vectors)
    X_test_tfidf = tfidf_transformer.transform(testing_vectors)
    print('the final tfidf train matrix with shape {}'.format(X_train_tfidf.shape))
    print('the final tfidf test matrix with shape {}'.format(X_test_tfidf.shape))
    print('yeah! done!')

    with open('data/X_train_tfidf_rest', 'wb') as fp:
        pickle.dump(X_train_tfidf, fp)
    with open('data/X_test_tfidf_rest', 'wb') as fp:
        pickle.dump(X_test_tfidf, fp)
    with open('data/y_train_rest', 'wb') as fp:
        pickle.dump(y_train, fp)
    with open('data/y_test_rest', 'wb') as fp:

        pickle.dump(y_test, fp)
    print('The test and training data are dumped in X_train_tfidf, X_test_tfidf, y_train, y_test')

if __name__ == '__main__':
    filename = './data/train.txt'
    X_train, X_test, y_train, y_test = getRawData(filename)
    bow(X_train, X_test, y_train, y_test)

