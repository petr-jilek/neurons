import numpy as np

"""
Neural network with numInputs inputs and numOutputs outputs.
Biases.
Activation functions.
"""


class NeutralNetwork(object):
    def __init__(self, numInputs, numOutputs, wMatrix, biases, actFunc, dActFunc):
        self.numInput = numInputs
        self.numOutputs = numOutputs
        self.wMatrix = wMatrix
        self.biases = biases
        self.actFunc = actFunc
        self.dActFunc = dActFunc

    def error(self, input, desiredOutput):
        output = self.forward(input)

        error = 0
        for i in range(self.numOutputs):
            error += (1 / self.numOutputs) * (output[i] - desiredOutput[i]) ** 2

        return error

    def forwardz(self, input):
        return np.dot(self.wMatrix, input) + self.biases

    def forward(self, input):
        z = self.forwardz(input)
        for i in range(self.numOutputs):
            z[i] = self.actFunc(z[i])

        return z

    def backpropagate(self, input, desiredOutput, learningRate):
        z = self.forwardz(input)
        output = self.forward(input)

        for i in range(self.numOutputs):
            dCdb = (output[i] - desiredOutput[i]) * self.dActFunc(z[i])
            self.biases[i] = self.biases[i] - learningRate * dCdb

            for j in range(self.numInput):
                dCdw = dCdb * input[j]
                self.wMatrix[i, j] = self.wMatrix[i, j] - learningRate * dCdw
