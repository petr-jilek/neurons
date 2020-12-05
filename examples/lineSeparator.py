import random
import numpy as np
import matplotlib.pyplot as plt
import models.activation_functions.act_func as af
import neural_networks.nn2layers as nn

matrix = np.array([[0.3, 0.7]])
biases = np.array([0.1])

nn = nn.NeutralNetwork(2, 1, matrix, biases, af.sigmoid, af.dsigmoid)


def run(numTrainingData=1000, numValidationData=1000, learningRate=0.2):
    """Generating training data"""
    trainingX = []
    trainingY = []
    trainingDesiredOuts = []
    trainingInputs = []
    for _ in range(numTrainingData):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        trainingX.append(x)
        trainingY.append(y)
        trainingInputs.append(np.array([x, y]))

        out = 1 if y > x else 0
        trainingDesiredOuts.append(np.array([out]))

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
        if out > 0.5:
            validationOuts.append(1)
        else:
            validationOuts.append(0)

    # print(validationX)
    # print(validationY)
    # print(validationOuts)

    plt.scatter(validationX, validationY, s=20, c=validationOuts)
    plt.show()
