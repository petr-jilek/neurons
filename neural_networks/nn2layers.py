import numpy as np

"""
Neural network 2 layers.
1 input layer, 1 output layer
Biases.
Activation functions.
"""


class NeutralNetwork(object):
    def __init__(self, numInputs, numOutputs, actFunc=None, dActFunc=None, wMatrix=None, biases=None):
        self.numInput = numInputs
        self.numOutputs = numOutputs

        if actFunc is None or dActFunc is None:
            self.actFunc = lambda x: x
            self.dActFunc = lambda x: x
        else:
            self.actFunc = actFunc
            self.dActFunc = dActFunc

        if wMatrix is None:
            self.wMatrix = np.random.rand(numOutputs, numInputs)
        else:
            self.wMatrix = wMatrix

        if biases is None:
            self.biases = np.random.rand(numOutputs)
        else:
            self.biases = biases

    def error(self, input, desiredOutput):
        output = self.forward(input)
        error = 0
        for i in range(self.numOutputs):
            error += (1 / self.numOutputs) * (output[i] - desiredOutput[i]) ** 2
        return error

    def activationFunction(self, input):
        a = []
        for i in range(len(input)):
            a.append(self.actFunc(input[i]))
        return a

    def forwardz(self, input):
        return np.dot(self.wMatrix, input) + self.biases

    def forward(self, input):
        return self.activationFunction(self.forwardz(input))

    def backpropagate(self, input, desiredOutput, learningRate):
        z = self.forwardz(input)
        a = self.activationFunction(z)

        for i in range(self.numOutputs):
            dCdb = (a[i] - desiredOutput[i]) * self.dActFunc(z[i])
            self.biases[i] = self.biases[i] - learningRate * dCdb

            for j in range(self.numInput):
                dCdw = dCdb * input[j]
                self.wMatrix[i, j] = self.wMatrix[i, j] - learningRate * dCdw
