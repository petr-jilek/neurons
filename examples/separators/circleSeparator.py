import math
import matplotlib.pyplot as plt
import models.activation_functions.act_func as af


def separationFunc(x, y):
    if (((x - 0.5) ** 2) + ((y - 0.5) ** 2)) < (0.3 ** 2):
        return 1
    else:
        return 0


def boundaryFunction(x):
    a = (0.3 ** 2) - ((x - 0.5) ** 2)
    if a > 0:
        return math.sqrt(a) + 0.5
    else:
        return -1


def boundaryFunction2(x):
    a = (0.3 ** 2) - ((x - 0.5) ** 2)
    if a > 0:
        return -math.sqrt(a) + 0.5
    else:
        return -1


def run(numTrainingData=20000, numValidationData=1000, hiddenNeurons=8, learningRate=2, showBoundaries=True):
    import neural_networks.nn3layers as nn
    import models.dataGenerator as data

    nn = nn.NeutralNetwork(2, hiddenNeurons, 2, af.sigmoid, af.dsigmoid)

    data = data.generate(separationFunc, [boundaryFunction, boundaryFunction2],
                         numTrainingData=numTrainingData, numValidationData=numValidationData)

    if showBoundaries:
        plt.scatter(data["xDesired"][0], data["yDesired"][0], s=2, c='red')
        plt.scatter(data["xDesired"][1], data["yDesired"][1], s=2, c='red')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    """Training neural net"""
    for i in range(len(data["trainingInputs"])):
        nn.backpropagate(data["trainingInputs"][i], data["trainingDesiredOuts"][i], learningRate)

    """Validating neural network"""
    validationOuts = []
    for i in range(len(data["validationInputs"])):
        out = nn.forward(data["validationInputs"][i])
        if out[0] > out[1]:
            validationOuts.append(1)
        else:
            validationOuts.append(0)

    plt.scatter(data["validationX"], data["validationY"], s=20, c=validationOuts)
    plt.show()
