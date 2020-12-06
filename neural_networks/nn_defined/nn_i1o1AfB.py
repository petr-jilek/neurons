"""
Neural network with 1 input and 1 output.
Biases.
Activation functions.
"""


class NeutralNetwork(object):
    # weight: Float number of weight (w).
    # Example: 0.8
    # bias: Float number of bias (b).
    # Example: 0.1
    # actFunc: Reference to activation function.
    # dActFunc: Reference to derivative of the activation function.
    def __init__(self, weight, bias, actFunc, dActFunc):
        self.weight = weight
        self.bias = bias
        self.actFunc = actFunc
        self.dActFunc = dActFunc

    # input: Input float number (i).
    # Example: 1.5
    # Return: Output float number before activation function (z)
    def forwardz(self, input):
        return (self.weight * input) + self.bias

    # input: Input float number (i).
    # Example: 1.5
    # Return: Output float number after activation function (a)
    def forward(self, input):
        return self.actFunc(self.forwardz(input))

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
        # Output before activation function.
        z = self.forwardz(input)
        # Output after activation function.
        a = self.forward(input)

        # Chain rule.
        dCda = 2 * (a - desiredOutput)
        dadz = self.dActFunc(z)
        dzdw = input
        dCdw = dCda * dadz * dzdw
        dCdb = dCda * dadz

        # Updating weight and bias.
        self.weight = self.weight - (learningRate * dCdw)
        self.bias = self.bias - (learningRate * dCdb)
