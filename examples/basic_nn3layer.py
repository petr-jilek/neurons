import numpy as np
import models.activation_functions.act_func as af
import neural_networks.nn3layers as nn

w1Matrix = np.array([[0.3, 0.7, 0.8],
                     [0.4, 0.8, 0.4],
                     [0.2, 0.3, 0.6],
                     [0.1, 0.7, 0.12]])
w2Matrix = np.array([[0.5, 0.3, 0.4, 0.1],
                     [0.2, 0.3, 0.8, 0.9],
                     [0.8, 0.4, 0.7, 0.6],
                     [0.8, 0.4, 0.7, 0.6]])

biases1 = np.array([0.3, 0.7, 0.5, 0.21])
biases2 = np.array([0.2, 0.4, 0.9, 0.8])

input = np.array([1.5, -4, 8])
desiredOutput = np.array([0.5, 0.4, 0.7, 0.1])

nn = nn.NeutralNetwork(3, 4, 4, w1Matrix, w2Matrix, biases1, biases2, af.sigmoid, af.dsigmoid)


def run():
    print(nn.forward(input))
    print(nn.error(input, desiredOutput))

    for _ in range(1000):
        nn.backpropagate(input, desiredOutput, 10)

    print(nn.forward(input))
    print(nn.error(input, desiredOutput))
