import gensim.models as g
import gensim.utils
import json
import numpy as np
import pickle



class doc_to_vector:
    def __init__(self, reviewfile, model):
        f = open(reviewfile, 'r')
        self.raw_reviews = json.load(f)
        self.model = g.Doc2Vec.load(model)
        self.labels = []
        self.reviews = []
        self.out_vectors = []

    def get_vocabulary(self):
        for key, value in self.raw_reviews.items():
            self.reviews.append(key)
            self.labels.append(value)

        print('size of review lists are {}'.format(len(self.labels)))

        for i, line in enumerate(self.reviews):
            stemmed = gensim.utils.simple_preprocess(line)
            vec = self.model.infer_vector(stemmed)
            self.out_vectors.append(vec)
        self.out_vectors = np.array(self.out_vectors)

        pickle.dump(self.out_vectors, open("data/yelp/doc2vec/doc2vec_reviews.p", "wb"))
        pickle.dump(self.labels, open("data/yelp/doc2vec/doc2vec_labels.p", "wb"))

if __name__=="__main__":
    reviewfile = 'data/yelp/reviews_label.txt'
    model = "model/enwiki_dbow/doc2vec.bin"
    doc2vec = doc_to_vector(reviewfile, model)
    doc2vec.get_vocabulary()
