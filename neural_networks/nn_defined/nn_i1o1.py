"""
Neural network with 1 input and 1 output.
No biases.
No activation functions.
"""


class NeutralNetwork(object):
    # weight: Float number of weight (w).
    # Example: 0.8
    def __init__(self, weight):
        self.weight = weight

    # input: Input float number (i).
    # Example: 1.5
    # Return: Output float number (a)
    def forward(self, input):
        return self.weight * input

    # input: Input float number (i).
    # Example: 1.5
    # desiredOutput: Desired output float number (y)
    # Example: 0.5
    # Return: Float number of Error (C)
    def error(self, input, desiredOutput):
        output = self.forward(input)
        return (output - desiredOutput) ** 2

    # input: Input float number (i).
    # Example: 1.5
    # desiredOutput: Desired output float number (y)
    # Example: 0.5
    # learningRate: Float value for gradient descent (alpha)
    # Example: 0.1
    def backpropagate(self, input, desiredOutput, learningRate):
        output = self.forward(input)

        # Chain rule.
        dCda = 2 * (output - desiredOutput)
        dadw = input
        dCdw = dCda * dadw

        # Updating weight.
        self.weight = self.weight - (learningRate * dCdw)
