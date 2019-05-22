#!/usr/bin/env python
# coding: utf-8
###################################################################################################
#
# preprocess .xml data to generate 5*1 label vector
# each dimension of vector represent food, service, price, ambience and anecdotes/miscellaneous
# the output file: sentence\n label\n ...
#
# example:
#
# from data_preprocessing import data_preprocess
# input_file = 'data/restaurant/train.xml'
# save_fname = 'data/restaurant/train_preprocess.txt'
# data_preprocess(input_file, save_fname)
#
###################################################################################################

import os
import spacy
import xml.etree.ElementTree as ET
nlp = spacy.load("en")

def aspect2num(aspect):
    if aspect == 'food':
        return 0
    elif aspect == 'service':
        return 1
    elif aspect == 'price':
        return 2
    elif aspect == 'ambience':
        return 3
    elif aspect == 'anecdotes/miscellaneous':
        return 4
    
def polarity2num(aspect):
    if aspect == 'positive':
        return 1
    elif aspect == 'neutral':
        return 0
    elif aspect == 'negative':
        return -1

def get_aspect_polarity_vector(aspect, polarity, cur_vector):
    cur_vector[aspect2num(aspect)] = polarity2num(polarity)
    return cur_vector

def data_preprocess(input_file, save_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    sentences, aspects, labels = [], [], []
    with open(save_file, 'w') as f:
        for sentence in root:
            sptoks = nlp(sentence.find('text').text)
            sentences.append(sptoks)
            f.write("%s\n" % sentences[-1])
            if len(sptoks.text.strip()) != 0:
                ids = []
                temp_po = [0, 0, 0, 0, 0]
                for asp_terms in sentence.iter('aspectCategories'):
                    for asp_term in asp_terms.findall('aspectCategory'):
                        if asp_term.get('polarity') == 'conflict':
                            continue
                        aspect = nlp(asp_term.get('category'))
                        polarity = asp_term.get('polarity') 
                        temp_po = get_aspect_polarity_vector(aspect.text, polarity, temp_po)
            labels.append(temp_po)
            f.write("%s\n" % labels[-1])
        print("Read %s sentences from %s" % (len(sentences), input_file))



