import numpy as np
from numpy import genfromtxt, savetxt

STUDENT_NAME = 'Okpala Walter , '
STUDENT_ID = '20910035'


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # initialize the list of weights matrices, then store the
        # network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

    def __repr__(self):
        # construct and return a string that represents the network
        # architecture
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # compute and return the sigmoid activation value for a
        # given input value
        return 1.0 / (1 + np.exp(-x))

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        # check to see if the bias column should be added
        if addBias:
            # insert a column of 1's as the last entry in the feature
            # matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]

        # loop over our layers in the network

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        # return the predicted value
        return p

    def load(self):
        self.W = np.load('weights.npy', allow_pickle=True)
        pass


def test_mlp(data_file):
    # Load the test set
    # START
    X_test = genfromtxt(data_file, delimiter=',')
    # END

    # Load your network
    # START
    nn = NeuralNetwork([X_test.shape[1], 50, 4])
    nn.load()
    # END

    # Predict test set - one-hot encoded

    # make a prediction on the data point
    pred = nn.predict(X_test)
    predictions = pred.argmax(axis=1)
    y_pred = np.eye(4)[predictions]

    return y_pred


'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''
