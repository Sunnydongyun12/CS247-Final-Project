#!/usr/bin/env python
# coding: utf-8
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import pickle


def load_data():
    file = open("data/X_train_doc2vec_rest",'rb')
    X_train_doc2vec_rest = pickle.load(file)
    file.close()

    file = open("data/y_train_doc2vec_rest",'rb')
    y_train_doc2vec_rest = pickle.load(file)
    file.close()

    file = open("data/X_test_doc2vec_rest",'rb')
    X_test_doc2vec_rest = pickle.load(file)
    file.close()

    file = open("data/y_test_doc2vec_rest",'rb')
    y_test_doc2vec_rest = pickle.load(file)
    file.close()
    print('data loaded successfully')
    return X_train_doc2vec_rest, y_train_doc2vec_rest, X_test_doc2vec_rest, y_test_doc2vec_rest


def model(X_train_doc2vec_rest, y_train_doc2vec_rest):
    print('start training model...')

    nn_1000 = MLPClassifier(
        hidden_layer_sizes=(205), alpha=0.001,
        learning_rate_init=0.01,max_iter=1000, random_state=9, tol=0.0001)
    nn_1000 = nn_1000.fit(X_train_doc2vec_rest, y_train_doc2vec_rest)

    return nn_1000

def train(X_train_doc2vec_rest, y_train_doc2vec_rest):
    model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
    model.fit(X_train_doc2vec_rest, y_train_doc2vec_rest)
    return model;

def get_activations(clf, X):
        hidden_layer_sizes = clf.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)
        layer_units = [X.shape[1]] + hidden_layer_sizes + [clf.n_outputs_]
        activations = [X]
        for i in range(clf.n_layers_ - 1):
            activations.append(np.empty((X.shape[0],
                                         layer_units[i + 1])))
        clf._forward_pass(activations)
        return activations


def main():
    for i in range(5):
        X_train_doc2vec_rest, y_train_doc2vec_rest, X_test_doc2vec_rest, y_test_doc2vec_rest = load_data()
        y_train_doc2vec_rest = np.array(y_train_doc2vec_rest)
        y_train_doc2vec_rest = y_train_doc2vec_rest[:,i]
        y_test_doc2vec_rest = np.array(y_test_doc2vec_rest)
        y_test_doc2vec_rest = y_test_doc2vec_rest[:,i]
        print(y_train_doc2vec_rest.shape)
        mlp = model(X_train_doc2vec_rest, y_train_doc2vec_rest)

        with open('model/feature_regression_mlp_doc2vec_{}.p'.format(i), 'wb') as fp:
            pickle.dump(mlp, fp)
        print('model is save to model/feature_regression_mlp_doc2vec_{}.p'.format(i))

        y_true = np.asarray(y_test_doc2vec_rest)
        y_pred_test = mlp.predict(X_test_doc2vec_rest)
        y_pred_train = mlp.predict(X_train_doc2vec_rest)

        print(y_true[:20])
        print(y_pred_test[:20])
        print(".............")
        print(y_train_doc2vec_rest[:20])
        print(y_pred_train[:20])


        mse_test = mean_squared_error(y_true, y_pred_test)
        mse_train = mean_squared_error(y_train_doc2vec_rest, y_pred_train)
        print("mse_test: {}, mse_train: {}".format(mse_test, mse_train))

        mae_test = mean_absolute_error(y_true, y_pred_test)
        mae_train = mean_absolute_error(y_train_doc2vec_rest, y_pred_train)
        print("mae_test: {}, mae_train: {}".format(mae_test, mae_train))


        acc_test = accuracy_score(y_true, y_pred_test)
        acc_train = accuracy_score(y_train_doc2vec_rest, y_pred_train)
        print("acc_test: {}, acc_train: {}".format(acc_test, acc_train))


if __name__ == '__main__':
    main()
