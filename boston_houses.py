from sklearn.datasets import load_boston
import numpy as np
from random import randint

n_epochs = 500
batch_size = 100
learning_rate = 0.00001

def preprocess(boston_data, percentage):
    x = boston_data.data
    y = boston_data.target
    n_data = x.shape[0]
    n_training = int(percentage * n_data)
    x_train = np.empty(shape=(n_training, 13))
    y_train = np.empty(shape=(n_training, 1))
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


def mean_square_error_prime(prediction, actual, input):
    diff = prediction - actual
    return diff * input


def mean_square_error(predicted, actual):
    error = sum(data ** 2 for data in (actual - predicted))/predicted.shape[0]
    print(error)
    return error


def get_gradients(inputs, predictions, actuals, weights):
    weights_gradient = -(2/float(inputs.shape[0])) * sum((actuals - predictions) * inputs)
    weights_gradient = weights_gradient.reshape((13, 1))
    weights = weights - (learning_rate * weights_gradient)
    return weights

def get_batch(inputs, outputs):
    batch_data = np.empty(shape=(batch_size, 13))
    batch_outputs = np.empty(shape=(batch_size, 1))
    for i in range(batch_size):
        index = randint(0, inputs.shape[0] - 1)
        batch_data[i] = inputs[index]
        batch_outputs[i] = outputs[index]
        np.delete(inputs, index, axis=0)
        np.delete(outputs, index, axis=0)
    return batch_data, batch_outputs




if __name__ == '__main__':
    boston = load_boston()
    test = boston.keys()
    x_train, x_test, y_train, y_test, n_training = preprocess(boston, 0.67)
    weights, bias = init_weights_and_bias(n_training)
    for i in range(n_epochs):
        batch_inputs, batch_outputs = get_batch(x_train, y_train)
        prediction = np.matmul(batch_inputs, weights) + bias
        error = mean_square_error(prediction, batch_outputs)
        weights = get_gradients(batch_inputs, prediction, batch_outputs, weights)
        if i % 100 == 0:
            print('current error: {}'.format(error))

