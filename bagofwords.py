from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import json
import pickle

class bag_of_words:
    def __init__(self, reviewfile):
        f = open(reviewfile, 'r')
        self.raw_reviews = json.load(f)
        self.labels = []
        self.reviews = []
        self.vocabulary = []

    def get_vocabulary(self):

        for key, value in self.raw_reviews.items():
            self.reviews.append(key)
            self.labels.append(value)

        print('size of review lists are {}'.format(len(self.labels)))

        print('stemming data...')
        stemmer = PorterStemmer()
        review_list_stem = []
        for line in self.reviews:
            temp = []
            words = word_tokenize(line)
            for w in words:
                w = stemmer.stem(w)
                temp.append(w)
            review_list_stem.append(" ".join(temp))
        print('stemming success!!!')

        print('vectorize data...')
        lmtz_train_data = review_list_stem
        vectorizer = CountVectorizer(stop_words='english', token_pattern=r'\b[^\d\W_]+\b', min_df=3)
        training_vectors = vectorizer.fit_transform(lmtz_train_data)

        print('vectorize success!!!')

        print('transform data into tfidf matrix...')
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(training_vectors)
        print('the final tfidf matrix with shape {}'.format(X_train_tfidf.shape))
        print(X_train_tfidf.shape)
        print('yeah! done!')

        pickle.dump(X_train_tfidf, open("reviews.p", "wb"))
        pickle.dump(self.labels, open("labels.p", "wb"))


if __name__ == '__main__':
    reviewfile = 'reviews_label.txt'
    bow = bag_of_words(reviewfile)
    bow.get_vocabulary()