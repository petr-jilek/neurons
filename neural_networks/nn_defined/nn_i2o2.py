import numpy as np

"""
Neural network with 2 inputs and 2 outputs.
No biases.
No activation functions.
"""


class NeutralNetwork(object):
    """
    2 inputs, 2 outputs
    """
    NUM_INPUTS = 2
    NUM_OUTPUTS = 2

    """
    wMatrix: Need to give matrix of weights.
    Example: numpy.array([[0.3, 0.7], [0.4, 0.2]])
    """
    def __init__(self, wMatrix):
        self.wMatrix = wMatrix

    """
        input: Input vector.
        Example: numpy.array([1.5, -4])
        desiredOutput: Vector of desired outputs.
        Example: numpy.array([0.5, 0.5])
        Return: Double value of Error (C)
    """
    def error(self, input, desiredOutput):
        output = self.forward(input)

        error = 0
        for i in range(self.NUM_OUTPUTS):
            error += (1 / 2) * (output[i] - desiredOutput[i]) ** 2

        return error

    """
        input: Input vector.
        Example: numpy.array([1.5, -4])
        Return: Output vector
    """
    def forward(self, input):
        return np.dot(self.wMatrix, input)

    """
        input: Input vector.
        Example: numpy.array([1.5, -4])
        desiredOutput: Vector of desired outputs.
        Example: numpy.array([0.5, 0.5])
        learningRate: Double value for gradient descent
        Example: 0.1
    """
    def backpropagate(self, input, desiredOutput, learningRate):
        output = self.forward(input)

        for i in range(self.NUM_OUTPUTS):
            for j in range(self.NUM_INPUTS):
                dCdw = (output[i] - desiredOutput[i]) * input[j]
                self.wMatrix[i, j] = self.wMatrix[i, j] - learningRate * dCdw
