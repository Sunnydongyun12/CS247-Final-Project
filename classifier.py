from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import torch
import torch.nn as nn
from models import aspectNet
from sklearn.neural_network._base import ACTIVATIONS

class Classifier:
    def __init__(self, data_file, label_file, ratio=0.8):
        with open(data_file, 'rb') as f:
            self.raw_data = pickle.load(f)

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
        model =  LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial', max_iter=10000, verbose=1, n_jobs=4)
        model.fit(self.train_data, self.train_label)
        predicted = model.predict(self.test_data)
        assert predicted.shape == self.test_label.shape
        score = np.mean(predicted == self.test_label)
        print(["accuracy:", score])

    def mlp(self):
        print('Training on multi-layer perceptron model......')
        model = MLPClassifier(random_state=0, hidden_layer_sizes=(2000,2000), max_iter=10000, verbose=True)
        for i in range(0,self.train_data.shape[0],1000):
            print(i)
            clf = model.partial_fit(self.train_data[i:i+1000], self.train_label[i:i+1000], classes=[0,1,2,3,4])
        predicted = clf.predict(self.test_data)
        assert predicted.shape == self.test_label.shape
        score = np.mean(predicted == self.test_label)
        print(["accuracy:", score])

    def test(self, net, batch_size):
        correct = 0
        total = 0
        net.eval()
        for batch_num in range(0, self.test_data.shape[0], batch_size):
            total += 1
            inputs, actual_val = self.test_data[batch_num:batch_num+batch_size,:], self.test_label[batch_num:batch_num+batch_size]
            # perform classification
            inputs = torch.from_numpy(inputs)
            actual_val = torch.from_numpy(actual_val)
            predicted_val = net(torch.autograd.Variable(inputs))
            # convert 'predicted_val' GPU tensor to CPU tensor and extract the column with max_score
            predicted_val = predicted_val.data
            max_score, idx = torch.max(predicted_val, 1)
            # compare it with actual value and estimate accuracy
            correct += (idx == actual_val).sum()

        print("Classifier Accuracy: ", correct / total * 100)

    def get_hidden_feature(self, data, layer):
        for i in range(layer):
            output = ACTIVATIONS['relu'](np.matmul(data, self.aspect_model.coefs_[0]) + self.aspect_model.intercepts_[0])
            data = output
        return output

    def aspect_mlp(self, aspectfile, numEpochs, batch_size):
        with open(aspectfile, 'rb') as f:
            self.aspect_model = pickle.load(f)
        print('model loaded')
        print('training on aspect.....')

        # set up loss function -- 'SVM Loss' a.k.a ''Cross-Entropy Loss
        loss_func = nn.CrossEntropyLoss()
        net = aspectNet(41652)
        # SGD used for optimization, momentum update used as parameter update
        optimization = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(numEpochs):

            # training set -- perform model training
            epoch_training_loss = 0.0
            num_batches = 0
            for batch_num in range(0, self.train_data.shape[0], batch_size):  # 'enumerate' is a super helpful function
                # split training data into inputs and labels
                inputs_, labels_ = self.train_data[batch_num:batch_num+batch_size,:], self.train_label[batch_num:batch_num+batch_size]  # 'training_batch' is a list
                inputs = torch.from_numpy(inputs_.toarray()).float()
                labels = torch.from_numpy(labels_)
                # wrap data in 'Variable'
                inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
                # Make gradients zero for parameters 'W', 'b'
                optimization.zero_grad()
                hidden_feature = self.get_hidden_feature(inputs_.toarray(), 1)
                hidden_feature = torch.from_numpy(hidden_feature).float()
                # forward, backward pass with parameter update
                forward_output = net(inputs, hidden_feature)
                loss = loss_func(forward_output, labels)
                loss.backward()
                optimization.step()
                # calculating loss
                epoch_training_loss += loss.data.item()
                num_batches += 1
                print(loss.data.item())
            print("epoch: ", epoch, ", loss: ", epoch_training_loss / num_batches)
            # self.test(net, batch_size)

if __name__=="__main__":
    classifier = Classifier('reviews.p','labels.p')
    # classifier.naive_bayes()
    # classifier.logistic_regression()
    # classifier.mlp()
    classifier.aspect_mlp('feature_regression_mlp.p', 1, 256)
