"""
Neural network with 1 input and 1 output.
Biases
Activation function
"""


class NeutralNetwork(object):
    def __init__(self, weight, bias, actFunc, dActFunc):
        self.weight = weight
        self.bias = bias
        self.actFunc = actFunc
        self.dActFunc = dActFunc

    def forwardz(self, input):
        return (self.weight * input) + self.bias

    def forward(self, input):
        return self.actFunc(self.forwardz(input))

    def error(self, input, desiredOutput):
        output = self.forward(input)
        return (output - desiredOutput) ** 2

    def backpropagate(self, input, desiredOutput, learningRate):
        z = self.forwardz(input)
        a = self.forward(input)

        """Chain rule"""
        dCda = 2 * (a - desiredOutput)
        dadz = self.dActFunc(z)
        dzdw = input
        dCdw = dCda * dadz * dzdw
        dCdb = dCda * dadz

        """Updating weight and bias"""
        self.weight = self.weight - (learningRate * dCdw)
        self.bias = self.bias - (learningRate * dCdb)
