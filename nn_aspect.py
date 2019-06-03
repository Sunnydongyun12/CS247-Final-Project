import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import pickle


def import_model(i):
    file = open("model/feature_regression_mlp_doc2vec_{}.p".format(i),'rb')
    aspect_mlp = pickle.load(file)
    return aspect_mlp

def load_data():
    with open('doc2vec_reviews.p', 'rb') as f:
        train_data = pickle.load(f)
    with open('doc2vec_labels.p', 'rb') as f:
        labels = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def main():


    X_train, X_test, y_train, y_test = load_data()
    X_train = np.asarray(X_train)

    total_aspect_vector = np.zeros((X_train.shape[0], 5))

    for i in range(5):
        aspect_mlp = import_model(i)
        aspect_vector = aspect_mlp.predict(X_train)
        for j in range(aspect_vector.shape[0]):
            total_aspect_vector[j][i] = aspect_vector[j]

    print(total_aspect_vector)

    X_input = np.concatenate((X_train, total_aspect_vector), axis=1)
    print(X_input.shape)

    nn_predict = MLPClassifier(hidden_layer_sizes = (205), activation='relu', solver='adam', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=500, shuffle=True,
        random_state=9, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #
    for i in range(0,X_input.shape[0],1000):
        print(i)
        clf = nn_predict.partial_fit(X_input[i:i+1000], y_train[i:i+1000], classes=[1,2,3,4,5])
    # clf = nn_predict.fit(X_input, y_train)
    #
    y_true = np.asarray(y_test)
    y_pred_train = clf.predict(X_input)

    test_total_aspect_vector = np.zeros((X_test.shape[0], 5))

    for i in range(5):
        aspect_mlp = import_model(i)
        aspect_vector = aspect_mlp.predict(X_test)
        for j in range(aspect_vector.shape[0]):
            test_total_aspect_vector[j][i] = aspect_vector[j]

    X_test_input = np.concatenate((X_test, test_total_aspect_vector), axis=1)
    print(X_test_input.shape)

    y_pred_test = clf.predict(X_test_input)

    mse_test = mean_squared_error(y_true, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    print("mse_test: {}, mse_train: {}".format(mse_test, mse_train))

    mae_test = mean_absolute_error(y_true, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    print("mae_test: {}, mae_train: {}".format(mae_test, mae_train))

    acc_test = accuracy_score(y_true, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    print("acc_test: {}, acc_train: {}".format(acc_test, acc_train))



if __name__ == '__main__':
    main()
