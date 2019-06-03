#!/usr/bin/env python
# coding: utf-8
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle


def load_data():
    file = open("data/X_train_tfidf_rest",'rb')
    X_train_tfidf_rest = pickle.load(file)
    file.close()

    file = open("data/y_train_rest",'rb')
    y_train_rest = pickle.load(file)
    file.close()

    file = open("data/X_test_tfidf_rest",'rb')
    X_test_tfidf_rest = pickle.load(file)
    file.close()

    file = open("data/y_test_rest",'rb')
    y_test_rest = pickle.load(file)
    file.close()
    print('data loaded successfully')
    return X_train_tfidf_rest, y_train_rest, X_test_tfidf_rest, y_test_rest


def model(X_train_tfidf_rest, y_train_rest):
    print('start training model...')
    # nn_1000_10_2 = MLPRegressor(
    #     hidden_layer_sizes=(1000, 10, 2),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    #     learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    #     random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    #     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # nn_1000_10_2 = nn_1000_10_2.fit(X_train_tfidf_rest, y_train_rest)


    # nn_1000_2 = MLPRegressor(
    #     hidden_layer_sizes=(1000, 2),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    #     learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    #     random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    #     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # nn_1000_2 = nn_1000_2.fit(X_train_tfidf_rest, y_train_rest)


    nn_1000 = MLPRegressor(
        hidden_layer_sizes=(1000, ),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    nn_1000 = nn_1000.fit(X_train_tfidf_rest, y_train_rest)

    return nn_1000


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
    X_train_tfidf_rest, y_train_rest, X_test_tfidf_rest, y_test_rest = load_data()
    mlp = model(X_train_tfidf_rest, y_train_rest)
    with open('model/feature_regression_mlp.p', 'wb') as fp:
        pickle.dump(mlp, fp)
    print('model is save to model/feature_regression_mlp.p')

    y_true = np.asarray(y_test_rest)
    y_pred_test = mlp.predict(X_test_tfidf_rest)
    y_pred_train = mlp.predict(X_train_tfidf_rest)


    print(y_pred_test)
    print(y_true)

    mse_test = mean_squared_error(y_true, y_pred_test)
    mse_train = mean_squared_error(y_train_rest, y_pred_train)
    print("mse_test: {}, mse_train: {}".format(mse_test, mse_train))

    mae_test = mean_absolute_error(y_true, y_pred_test)
    mae_train = mean_absolute_error(y_train_rest, y_pred_train)
    print("mae_test: {}, mae_train: {}".format(mae_test, mae_train))

if __name__ == '__main__':
    main()
