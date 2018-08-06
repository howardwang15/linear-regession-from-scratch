from sklearn.datasets import load_boston
import numpy as np
from random import randint
import pandas as pd


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
    return x_train, x, y_train, y




def init_weights_and_bias():
    weights = np.array([1, 2])


if __name__ == '__main__':
    boston = load_boston()
    test = boston.keys()
    x_train, x_test, y_train, y_test = preprocess(boston, 0.67)