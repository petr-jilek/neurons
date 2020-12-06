import numpy as np
import matplotlib.pyplot as plt
import models.activationFunctions as af
import neural_networks.nn2layers as nn
import models.separationFunctions as sf
import models.dataGenerator as generator

"""
Points separation by one line example.
"""

# Creating matrix of weights and vector of biases.
# Matrix and biases don't have to be specified. If not neural network will generate random.
matrix = np.array([[0.3, 0.7],
                   [0.8, 0.2]])
biases = np.array([0.1, 0.2])

# Creating neural network with 2 inputs and 2 outputs neurons.
# Using sigmoid activation function.
nn = nn.NeutralNetwork(2, 2, af.sigmoid, af.dsigmoid, matrix, biases)


# Run the example.
def run(numTrainingData=1000, numValidationData=1000, learningRate=0.5, showBoundaries=True):
    # Generating training and validation data.
    data = generator.generate(separationFunc=sf.linearSeparationFunc, boundaryFunctions=[sf.linearBoundaryFunction],
                              numTrainingData=numTrainingData, numValidationData=numValidationData)

    # Training neural net.
    for i in range(numTrainingData):
        nn.backpropagate(data["trainingInputs"][i], data["trainingDesiredOuts"][i], learningRate)

    # Validating neural network.
    validationOuts = []
    correct = 0
    incorrect = 0
    for i in range(numValidationData):
        out = nn.forward(data["validationInputs"][i])
        if out[0] > out[1]:
            validationOuts.append(1)
            if sf.linearSeparationFunc(data["validationInputs"][i][0], data["validationInputs"][i][1]) == 1:
                correct += 1
            else:
                incorrect += 1
        else:
            validationOuts.append(0)
            if sf.linearSeparationFunc(data["validationInputs"][i][0], data["validationInputs"][i][1]) == 0:
                correct += 1
            else:
                incorrect += 1

    # Calculate and print accuracy in (%).
    accuracy = (correct / (correct + incorrect)) * 100
    print("Accuracy: ", accuracy)

    # Plot the results.
    plt.scatter(data["validationX"], data["validationY"], s=20, c=validationOuts)

    if showBoundaries:
        plt.scatter(data["xDesired"][0], data["yDesired"][0], s=2, c='red')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()
