import math


def sigmoid(input):
    return 1 / (1 + math.exp(-input))


def dsigmoid(input):
    return math.exp(input) / ((1 + math.exp(input)) ** 2)
