import gensim.models as g
import gensim.utils
import json
import numpy as np
import pickle


alpha = 0.025
vec_size = 300
window_size = 5
max_epochs = 40

class doc_to_vector:
    def __init__(self, reviewfile, model_path):
        f = open(reviewfile, 'r')
        self.raw_reviews = json.load(f)
        self.model_path = model_path
        self.labels = []
        self.reviews = []
        self.out_vectors = []

    def get_vocabulary(self):
        for key, value in self.raw_reviews.items():
            self.reviews.append(key)
            self.labels.append(value)

        print('size of review lists are {}'.format(len(self.labels)))

        print('proprocess training data ... ')
        train_corpus = [g.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc), [i]) for i, doc in enumerate(self.reviews)]

        print(len(train_corpus))
        model = g.Doc2Vec(size=vec_size,window=window_size, alpha=alpha,min_alpha=0.00025,min_count=1,dm =0)

        model.build_vocab(train_corpus)
        # print(train_corpus[:2])
        print('start training doc2vec ... ')
        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
        # model = g.Doc2Vec(train_corpus, vector_size=300, window=15, min_count=2, workers=4)

        model.save(self.model_path)

        for i, line in enumerate(self.reviews):
            stemmed = gensim.utils.simple_preprocess(line)
            vec = model.infer_vector(stemmed)
            self.out_vectors.append(vec)
        self.out_vectors = np.array(self.out_vectors)

        pickle.dump(self.out_vectors, open("trained_doc2vec_reviews.p", "wb"))
        pickle.dump(self.labels, open("trained_doc2vec_labels.p", "wb"))

if __name__=="__main__":
    reviewfile = 'reviews_label.txt'
    model_path = "doc2vec_trained.bin"
    doc2vec = doc_to_vector(reviewfile, model_path)
    doc2vec.get_vocabulary()
