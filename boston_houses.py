from sklearn.datasets import load_boston #import dataset
import numpy as np #data processing
from random import randint #generate random numbers


n_epochs = 7000 #number of training iterations
batch_size = 100 #number of samples every iteration
learning_rate = 0.000002 #how much the model updates its weights every training iteration

def preprocess(boston_data, split_ratio): 
    """preprocess the dataset by splitting data for training and testing

    Args:
        boston_data (class object): collection of data from the sklearn dataset
        split_ratio (float): percentage of data from dataset used for training. Percentage
            of data used for testing is 1 - split_ratio
    
    Returns:
        tuple: contains traning and testing inputs and outputs
    """
    x = boston_data.data
    y = boston_data.target
    n_data = x.shape[0]
    n_training = int(split_ratio * n_data)
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
    return x_train, x, y_train, y


def init_weights_and_bias(training_size):
    """initialize weights and bias matrices for training the linear regression model
    
    Args: 
        training_size (int): number of training samples
    
    Returns:
        tuple: contains the initial weights and biases
    """
    weights = np.random.rand(13, 1)
    bias = 0.1
    return weights, bias


def mean_square_error(predicted, actual):
    """Cost function used to evaluate how accurate the model is during training

    Args:
        predicted (numpy array of floats): predicted house prices during training
        actual (numpy array of floats): real-world prices of the inputs from the dataset

    Returns:
        float: the mean squared error between the predicted and actual house prices
    """
    error = sum(data ** 2 for data in (actual - predicted))/predicted.shape[0]
    return error


def gradient_descent(inputs, predictions, actuals, weights, biases):
    """Perform the stochastic gradient descent algorithm to update the weights during each training epoch

    Args:
        inputs (numpy array of floats): array of features used during the training epoch
        predictions (numpy array of floats): predicted house prices during training
        actuals (numpy array of floats): real-world prices of the inputs from the dataset
        weights (numpy array of floats): the weights used for training the model
        biases (numpy array of floats): the biases used for training the model

    Returns:
        tuple: contains the updated weights and biases
    """
    weights_gradient = -(2/float(inputs.shape[0])) * sum((actuals - predictions) * inputs)
    weights_gradient = weights_gradient.reshape((13, 1))
    bias_gradient = -2/(float(inputs.shape[0])) * sum(actuals - predictions)
    weights -= learning_rate * weights_gradient
    biases -= learning_rate * bias_gradient
    return weights, biases


def get_batch(inputs, outputs):
    """Randomly select samples from the dataset for training

    Args:
        inputs (numpy array of floats): array of all features from dataset
        outputs (numpy array of floats): array of all house prices from dataset

    Returns:
        tuple: contains the randomly sampled features and their associated house prices
    """
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
    """Preprocess, train, and test the linear regression model
    """
    boston = load_boston()
    x_train, x_test, y_train, y_test = preprocess(boston, 0.67)
    n_training = x_train.shape[0]
    n_testing = x_test.shape[0]
    weights, bias = init_weights_and_bias(n_training)
    for i in range(n_epochs):
        batch_inputs, batch_outputs = get_batch(x_train, y_train)
        prediction = np.matmul(batch_inputs, weights) + bias
        error = mean_square_error(prediction, batch_outputs)
        weights, bias = gradient_descent(batch_inputs, prediction, batch_outputs, weights, bias)
        if i % 100 == 0:
            print('current error: {}'.format(error))

    for i in range(n_testing):
        actual = y_test[i]
        predicted = np.matmul(x_test[i], weights) + bias
        print('actual: {0}, predicted: {1}'.format(actual, predicted[0]))

