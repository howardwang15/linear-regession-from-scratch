from sklearn.datasets import load_boston
import numpy as np
from random import randint

n_epochs = 500

def preprocess(boston_data, percentage):
    x = boston_data.data
    print(x.shape)
    y = boston_data.target
    n_data = x.shape[0]
    n_training = int(percentage * n_data)
    x_train = np.empty(shape=(n_training, 13))
    y_train = np.empty(shape=(n_training, 1))
    print(x_train.shape)
    for i in range(n_training):
        index = randint(0, n_data - 1)
        x_train[i] = x[index]
        a = x[index]
        y_train[i] = y[index]
        x = np.delete(x, index, axis=0)
        y = np.delete(y, index, axis=0)
        n_data -= 1
    return x_train, x, y_train, y, n_training


def init_weights_and_bias(training_size):
    weights = np.random.rand(13, 1)
    bias = 0.1
    return weights, bias


def mean_square_error(predicted, actual):
    sum = 0
    print(prediction.size)
    for i in range(predicted.size):
        error = np.square(actual[i] - predicted[i])
        sum += error
    return sum


if __name__ == '__main__':
    boston = load_boston()
    test = boston.keys()
    x_train, x_test, y_train, y_test, n_training = preprocess(boston, 0.67)
    weights, bias = init_weights_and_bias(n_training)
    for i in range(n_epochs):
        prediction = np.matmul(x_train, weights) + bias
        sum = mean_square_error(prediction, y_train)
        