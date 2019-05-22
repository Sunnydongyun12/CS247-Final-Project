import os
import ast
import spacy
import numpy as np
import xml.etree.ElementTree as ET
from errno import ENOENT
from collections import Counter

nlp = spacy.load("en")

def read_data(fname, save_fname, pre_processed):
    dic = {'food': 0, 'service': 1, 'price': 2, 'ambience': 3, 'anecdotes/miscellaneous': 4}

    sentences, aspects, polarities = [], [], []
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        lines = open(save_fname, 'r').readlines()
        for i in range(0, len(lines), 5):
            sentences.append(ast.literal_eval(lines[i]))
            aspects.append(ast.literal_eval(lines[i + 1]))
            sentence_lens.append(ast.literal_eval(lines[i + 2]))
            sentence_locs.append(ast.literal_eval(lines[i + 3]))
            labels.append(ast.literal_eval(lines[i + 4]))
    else:
        if not os.path.isfile(fname):
            raise IOError(ENOENT, 'Not a file', fname)

        tree = ET.parse(fname)
        root = tree.getroot()
        with open(save_fname, 'w') as f:
            for sentence in root:
                sptoks = nlp(sentence.find('text').text)
                sentences.append(sptoks)
                if len(sptoks.text.strip()) != 0:
                    asps = []
                    polars = [0]*5
                    for asp_categories in sentence.iter('aspectCategories'):
                        for asp_category in asp_categories.findall('aspectCategory'):
                            if asp_category.get('polarity') == 'conflict':
                                continue
                            t_sptoks = nlp(asp_category.get('category'))
                            asps.append(t_sptoks)
                            # print(type(t_sptoks.text))

                            polarity = asp_category.get('polarity')
                            index = dic.get(t_sptoks.text)
                            if polarity == 'negative':
                                polars[index] = -1
                            elif polarity == 'neutral':
                                polars[index] = 0
                            elif polarity == "positive":
                                polars[index] = 1
                        aspects.append(asps)
                        polarities.append(polars)
                    f.write("%s\n" % sentences[-1])
                    f.write("%s\n" % aspects[-1])
                    f.write("%s\n" % polarities[-1])
