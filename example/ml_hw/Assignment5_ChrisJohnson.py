"""
Chris Johnson
Assignment #5  3-6-2 neural network
4/22/20
CS3120-001 Machine Learning

Class Resources-
- 8_NN_2layer.py
    @author: Madhuri Suthar, UCLA
- 11_NN.pdf
- 12_BackdropRegularizationLossFunc.pdf
"""

import numpy as np

###################################################################################
# global variables

X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
y = np.array(([0, 1], [1, 0], [1, 0], [0, 1]), dtype=float)


####################################################################################
# helper functions


def sigmoid(t):
    # Activation function
    return 1/(1+np.exp(-t))


def sigmoid_derivative(p):
    # Derivative of Sigmoid
    return p * (1-p)

####################################################################################


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        # 5 dimensions of weight matrix
        self.weights1 = np.random.rand(self.input.shape[1], 6)  # 6 nodes in hidden layer
        self.weights2 = np.random.rand(6, 2)
        print(self.weights1.shape)
        print(self.weights2.shape)
        self.y = y
        self.output = np.zeros(y.shape)

    def feed_forward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def back_prop(self):
        d_weights2 = np.dot(self.layer1.T, 2*(self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) *
                                                 sigmoid_derivative(self.output), self.weights2.T) *
                            sigmoid_derivative(self.layer1))
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feed_forward()
        self.back_prop()

####################################################################################


NN = NeuralNetwork(X, y)
# monitor loss and train
for i in range(1500):  
    if i % 300 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input: \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feed_forward()))
        print("Loss: \n" + str(np.mean(np.square(y - NN.feed_forward()))))  # mean sum squared loss
        print("\n")
    NN.train(X, y)

# 6 test sample 1 , 2
X_test = np.array(([0, 0, 0], [1, 1, 1]), dtype=float)
hidden_layer = sigmoid(np.dot(X_test, NN.weights1))
y_prediction = sigmoid(np.dot(hidden_layer, NN.weights2))
print("Prediction Output: \n" + str(y_prediction))
