import math

"""
Activation function for neurons
"""


# sigmoid activation function f(x) = 1 / (1 + e^(-x))
def sigmoid(input):
    return 1 / (1 + math.exp(-input))


# derivation of sigmoid activation function df(x)/dx = e^x / (1 + e^(x)^2)
def dsigmoid(input):
    return math.exp(input) / ((1 + math.exp(input)) ** 2)


# linear activation function f(x) = x
def linear(input):
    return input


# derivation of linear activation function df(x)/dx = 1
def dlinear(input):
    return 1


# rectified linear activation function f(x) = x for (x > 0) and 0 for (x <= 0)
def rectifiedLinear(input):
    if input > 0:
        return input
    else:
        return 0


# derivation of rectified linear activation function df(x)/dx = 1 for (x > 0) and 0 for (x <= 0)
def drectifiedLinear(input):
    if input > 0:
        return 1
    else:
        return 0


# step activation function f(x) = 1 for (x >= 0) and 0 for (x < 0)
def stepFunction(input):
    if input >= 0:
        return 1
    else:
        return 0


# derivation of step activation function df(x)/dx = 0
# normally derivation is 0, but for learning purpose I have chosen 1
def dstepFunction(input):
    return 1
