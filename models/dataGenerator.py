import random
import math
import numpy as np


def generate(separationFunc, boundaryFunctions, numTrainingData=1000, numValidationData=1000):
    """Generating training data"""
    trainingX = []
    trainingY = []
    trainingDesiredOutsArray = []
    trainingDesiredOuts = []
    trainingInputs = []
    for _ in range(numTrainingData):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        trainingX.append(x)
        trainingY.append(y)
        trainingInputs.append(np.array([x, y]))

        out = separationFunc(x, y)

        trainingDesiredOutsArray.append(out)
        trainingDesiredOuts.append(np.array([out, (1 - out)]))

    xDesired = []
    yDesired = []
    for i in range(len(boundaryFunctions)):
        x = np.linspace(0, 1, 300)
        y = np.array([boundaryFunctions[i](xi) for xi in x])
        xDesired.append(x)
        yDesired.append(y)

    """Generating validation data"""
    validationX = []
    validationY = []
    validationInputs = []
    for _ in range(numValidationData):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        validationX.append(x)
        validationY.append(y)
        validationInputs.append(np.array([x, y]))

    dic = {
        "trainingX": trainingX,
        "trainingY": trainingY,
        "trainingDesiredOutsArray": trainingDesiredOutsArray,
        "trainingDesiredOuts": trainingDesiredOuts,
        "trainingInputs": trainingInputs,

        "xDesired": xDesired,
        "yDesired": yDesired,

        "validationX": validationX,
        "validationY": validationY,
        "validationInputs": validationInputs
    }

    return dic
