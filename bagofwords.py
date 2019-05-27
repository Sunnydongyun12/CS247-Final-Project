import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
import re
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
      ps = PorterStemmer()
      idx = 0
      for review, star in self.raw_reviews.items():
        idx += 1
        if idx%1000 == 0:
            print(idx)
        new_review = []
        for i in range(len(self.vocabulary)):
          new_review.append(0.0)
        self.labels.append(star)
        words = review.split()
        for word in words:
          tokens = word_tokenize(word)
          for token in tokens:
            # further split by non-alpha character
            tokens_ = re.split('[^a-zA-Z\']', token)
            for token_ in tokens_:
              flag = 0
              # check whether the character is a word
              for letter in token_:
                if letter.isalpha():
                  flag = 1
                  break
              # if it is a word, add it in vocabulary
              if flag == 1:
                token_ = ps.stem(token_)
                if not token_ in self.vocabulary:
                  self.vocabulary.append(token_)
                  for review_ in self.reviews:
                    review_.append(0.0)
                  new_review.append(1)
                else:
                  ind = self.vocabulary.index(token_)
                  new_review[ind] += 1
        self.reviews.append(new_review)
      pickle.dump(self.reviews, open("reviews.p", "wb"))
      pickle.dump(self.vocabulary, open("vocabulary.p", "wb"))
      pickle.dump(self.labels, open("labels.p", "wb"))


if __name__ == '__main__':
    reviewfile = 'reviews_label.txt'
    bow = bag_of_words(reviewfile)
    bow.get_vocabulary()