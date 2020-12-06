import math

"""
Separation and boundary function for dataGenerator and separators.
Separation function (x, y): Return either 1 or 0 in which region output is.
Boundary function (x): Return value of f(x) which describing decision boundary for learning neural network.
"""


# Circle separation and boundary functions.
# Circle with center in x = 0.5 and y = 0.5 and radius 0.3
# (x - 0.5)^2 + (y - 0.5)^2 = 0.3^2
def circleSeparationFunc(x, y):
    if (((x - 0.5) ** 2) + ((y - 0.5) ** 2)) < (0.3 ** 2):
        return 1
    else:
        return 0


def circleBoundaryFunction(x):
    a = (0.3 ** 2) - ((x - 0.5) ** 2)
    if a > 0:
        return math.sqrt(a) + 0.5
    else:
        return 0


def circleBoundaryFunction2(x):
    a = (0.3 ** 2) - ((x - 0.5) ** 2)
    if a > 0:
        return -math.sqrt(a) + 0.5
    else:
        return 0


# Linear separation and boundary functions.
# Linear function y = f(x) = x
def linearSeparationFunc(x, y):
    if y > x:
        return 1
    else:
        return 0


def linearBoundaryFunction(x):
    return x
