import matplotlib.pyplot as plt
import models.activationFunctions as af
import models.separationFunctions as sf
import models.dataGenerator as generator

"""
Points separation by circle with the center at x=0.5 and y=0.5.
"""


# Run the example.
def run(numTrainingData=20000, numValidationData=1000, hiddenNeurons=8, learningRate=2, showBoundaries=True):
    import neural_networks.nn3layers as nn

    # Creating neural network with 2 inputs, hidden neurons by variable and 2 outputs neurons.
    # Using sigmoid activation function.
    nn = nn.NeutralNetwork(2, hiddenNeurons, 2, af.sigmoid, af.dsigmoid)

    # Generating training and validation data.
    data = generator.generate(sf.circleSeparationFunc, [sf.circleBoundaryFunction, sf.circleBoundaryFunction2],
                              numTrainingData=numTrainingData, numValidationData=numValidationData)

    # Training neural net.
    for i in range(len(data["trainingInputs"])):
        nn.backpropagate(data["trainingInputs"][i], data["trainingDesiredOuts"][i], learningRate)

    # Validating neural network.
    validationOuts = []
    correct = 0
    incorrect = 0
    for i in range(len(data["validationInputs"])):
        out = nn.forward(data["validationInputs"][i])
        if out[0] > out[1]:
            validationOuts.append(1)
            if sf.circleSeparationFunc(data["validationInputs"][i][0], data["validationInputs"][i][1]) == 1:
                correct += 1
            else:
                incorrect += 1
        else:
            validationOuts.append(0)
            if sf.circleSeparationFunc(data["validationInputs"][i][0], data["validationInputs"][i][1]) == 0:
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
        plt.scatter(data["xDesired"][1], data["yDesired"][1], s=2, c='red')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()
