import numpy as np
import models.helpers as helper

"""
Neural network 3 layers.
1 input layer, 1 hidden layer, 1 output layer
Biases.
Activation functions.
"""


class NeutralNetwork(object):
    def __init__(self, numInputs, numHidden, numOutputs, w1Matrix, w2Matrix, biases1, biases2, actFunc, dActFunc):
        self.numInput = numInputs
        self.numHidden = numHidden
        self.numOutputs = numOutputs
        self.w1Matrix = w1Matrix
        self.w2Matrix = w2Matrix
        self.biases1 = biases1
        self.biases2 = biases2
        self.actFunc = actFunc
        self.dActFunc = dActFunc

    def error(self, input, desiredOutput):
        output = self.forward(input)
        error = 0
        for i in range(self.numOutputs):
            error += (1 / self.numOutputs) * ((output[i] - desiredOutput[i]) ** 2)
        return error

    def activationFunction(self, input):
        out = []
        for i in range(len(input)):
            out.append(self.actFunc(input[i]))
        return out

    def forwardz1(self, input):
        return np.dot(self.w1Matrix, input) + self.biases1

    def forwardz2(self, a1):
        return np.dot(self.w2Matrix, a1) + self.biases2

    def forward(self, input):
        output = self.activationFunction(self.forwardz2(self.activationFunction(self.forwardz1(input))))
        return output

    def backpropagate(self, input, desiredOutput, learningRate):
        z1 = self.forwardz1(input)
        a1 = self.activationFunction(z1)
        z2 = self.forwardz2(a1)
        a2 = self.activationFunction(z2)

        z1V = helper.arrayToColumnVector(z1)
        a1V = helper.arrayToColumnVector(a1)
        a2V = helper.arrayToColumnVector(a2)
        z2V = helper.arrayToColumnVector(z2)
        inputV = helper.arrayToColumnVector(input)

        deltaBiases2 = []
        for i in range(self.numOutputs):
            dCda2i = (2 / self.numOutputs) * (a2[i] - desiredOutput[i])
            da2idz2i = a2[i] * (1 - a2[i])
            dCdz2i = dCda2i * da2idz2i
            deltaBiases2.append(dCdz2i)

        deltaBiases2V = helper.arrayToColumnVector(deltaBiases2)
        deltaWeights2V = np.dot(deltaBiases2V, np.transpose(a1V))

        deltaBiases1V = np.dot(np.transpose(self.w2Matrix), deltaBiases2V)
        for i in range(self.numHidden):
            deltaBiases1V[i][0] = deltaBiases1V[i][0] * a1[i] * (1 - a1[i])

        deltaWeights1V = np.dot(deltaBiases1V, np.transpose(inputV))

        b2 = self.biases2 - np.transpose(learningRate * deltaBiases2V)[0]
        w2 = self.w2Matrix - (learningRate * deltaWeights2V)
        b1 = self.biases1 - np.transpose(learningRate * deltaBiases1V)[0]
        w1 = self.w1Matrix - (learningRate * deltaWeights1V)

        self.biases2 = b2
        self.w2Matrix = w2
        self.biases1 = b1
        self.w1Matrix = w1
