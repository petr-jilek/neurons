import numpy as np

"""
Helper functions, such as array converters and others.
"""


def arrayToRowVector(array):
    vector = np.array([[i for i in array]])
    return vector


def arrayToColumnVector(array):
    vector = np.array([[i for i in array]])
    return np.transpose(vector)


def mult(array, number):
    arr = []
    for i in array:
        arr.append(i * number)
    return arr
