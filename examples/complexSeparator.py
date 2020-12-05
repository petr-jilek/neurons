import random
import math
import numpy as np
import matplotlib.pyplot as plt
import models.activation_functions.act_func as af
import neural_networks.nn3layers as nn

nn = nn.NeutralNetwork(2, 8, 2, af.sigmoid, af.dsigmoid)


def run(numTrainingData=10000, numValidationData=1000, learningRate=10):
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

        """
        if x < 0.5:
            out = 1 if y > x else 0
        else:
            out = 1 if y > (1 - x) else 0
        """
        out = 1 if y > (math.sin(4 * x) / 1.5) else 0

        trainingDesiredOutsArray.append(out)
        trainingDesiredOuts.append(np.array([out, (1 - out)]))

    xDesired = np.linspace(0, 1, 300)
    yDesired = np.array([(math.sin(4 * i) / 1.5) for i in xDesired])

    plt.scatter(xDesired, yDesired, s=2, c='red')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # plt.scatter(trainingX, trainingY, s=20, c=trainingDesiredOutsArray)
    # plt.show()

    """Training neural net"""
    for i in range(numTrainingData):
        nn.backpropagate(trainingInputs[i], trainingDesiredOuts[i], learningRate)

    """Generating validation data"""
    validationX = []
    validationY = []
    validationInputs = []
    validationOuts = []
    for _ in range(numValidationData):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        validationX.append(x)
        validationY.append(y)
        validationInputs.append(np.array([x, y]))

    """Validating neural network"""
    for i in range(numValidationData):
        out = nn.forward(validationInputs[i])
        if out[0] > out[1]:
            validationOuts.append(1)
        else:
            validationOuts.append(0)

    # print(validationX)
    # print(validationY)
    # print(validationOuts)

    plt.scatter(validationX, validationY, s=20, c=validationOuts)
    plt.show()
