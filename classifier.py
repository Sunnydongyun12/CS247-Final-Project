from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np

class Classifier:
    def __init__(self, data_file, label_file, ratio=0.8):
        with open(data_file, 'rb') as f:
            self.raw_data = pickle.load(f)
            # self.raw_data = np.array(self.raw_data)
        with open(label_file, 'rb') as f:
            labels = pickle.load(f)
        print('data loaded')
        self.labels = []
        for label in labels:
            self.labels.append(int(label)-1)
        self.labels = np.array(self.labels)
        print(type(self.labels))
        train_set = int(self.raw_data.shape[0]*ratio)
        self.train_data = self.raw_data[:train_set,:]
        self.test_data = self.raw_data[train_set:,:]
        self.train_label = self.labels[:train_set]
        self.test_label = self.labels[train_set:]
        print(['train_data shape', self.train_data.shape])

    def naive_bayes(self):
        print('training begin.....')
        model = MultinomialNB()
        for i in range(0,self.train_data.shape[0],1000):
            print(i)
            clf = model.partial_fit(self.train_data[i:i+1000], self.train_label[i:i+1000], classes=[0,1,2,3,4])
        predicted = clf.predict(self.test_data)
        assert predicted.shape == self.test_label.shape
        score = np.mean(predicted == self.test_label)
        print(["accuracy:", score])

    def logistic_regression(self):
        print('Training on logistic regression model......')
        model =  LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial', max_iter=10000, n_jobs=4)
        model.fit(self.train_data, self.train_label)
        predicted = model.predict(self.test_data)
        assert predicted.shape == self.test_label.shape
        score = np.mean(predicted == self.test_label)
        print(["accuracy:", score])

    def mlp(self):
        print('Training on multi-layer perceptron model......')
        model = MLPClassifier(random_state=0, hidden_layer_sizes=(20,), max_iter=10000)
        for i in range(0,self.train_data.shape[0],1000):
            print(i)
            clf = model.partial_fit(self.train_data[i:i+1000], self.train_label[i:i+1000], classes=[0,1,2,3,4])
        predicted = clf.predict(self.test_data)
        assert predicted.shape == self.test_label.shape
        score = np.mean(predicted == self.test_label)
        print(["accuracy:", score])


if __name__=="__main__":
    classifier = Classifier('doc2vec_reviews.p','doc2vec_labels.p')
    # classifier.naive_bayes()
    classifier.logistic_regression()
    classifier.mlp()
