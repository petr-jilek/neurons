import numpy as np

"""
Neural network with 2 inputs and 2 outputs.
Biases.
Activation functions.
"""


class NeutralNetwork(object):
    NUM_INPUTS = 2
    NUM_OUTPUTS = 2

    def __init__(self, wMatrix, biases, actFunc, dActFunc):
        self.wMatrix = wMatrix
        self.biases = biases
        self.actFunc = actFunc
        self.dActFunc = dActFunc

    def error(self, input, desiredOutput):
        output = self.forward(input)

        error = 0
        for i in range(self.NUM_OUTPUTS):
            error += (1 / 2) * (output[i] - desiredOutput[i]) ** 2

        return error

    def forwardz(self, input):
        return np.dot(self.wMatrix, input) + self.biases

    def forward(self, input):
        z = self.forwardz(input)
        for i in range(self.NUM_OUTPUTS):
            z[i] = self.actFunc(z[i])

        return z

    def backpropagate(self, input, desiredOutput, learningRate):
        z = self.forwardz(input)
        output = self.forward(input)

        for i in range(self.NUM_OUTPUTS):
            dCdb = (output[i] - desiredOutput[i]) * self.dActFunc(z[i])
            self.biases[i] = self.biases[i] - learningRate * dCdb

            for j in range(self.NUM_INPUTS):
                dCdw = dCdb * input[j]
                self.wMatrix[i, j] = self.wMatrix[i, j] - learningRate * dCdw
