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
    print(x_train.shape)
    for i in range(n_training):
        index = randint(0, n_data - 1)
        x_train[i] = x[index]
        x = np.delete(x, index)
        n_data -= 1
    print(x_train)
    print(x)



def init_weights_and_bias():
    weights = np.array([1, 2])


if __name__ == '__main__':
    boston = load_boston()
    preprocess(boston, 0.67)