import math


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


def linearSeparationFunc(x, y):
    if y > x:
        return 1
    else:
        return 0


def linearBoundaryFunction(x):
    return x
