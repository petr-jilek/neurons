import numpy as np
import models.activationFunctions as af
import neural_networks.nn3layers as nn

# Creating matrix of weights and vector of biases.
# Matrix and biases don't have to be specified. If not neural network will generate random.
w1Matrix = np.array([[0.3, 0.7, 0.8],
                     [0.4, 0.8, 0.4],
                     [0.2, 0.3, 0.6],
                     [0.1, 0.7, 0.12]])
w2Matrix = np.array([[0.5, 0.3, 0.4, 0.1],
                     [0.2, 0.3, 0.8, 0.9],
                     [0.8, 0.4, 0.7, 0.6],
                     [0.8, 0.4, 0.7, 0.6]])

biases1 = np.array([0.3, 0.7, 0.5, 0.21])
biases2 = np.array([0.2, 0.4, 0.9, 0.8])

# Creating neural network with 3 inputs, 4 hidden and 4 outputs neurons.
# Using sigmoid activation function.
nn = nn.NeutralNetwork(3, 4, 4, af.sigmoid, af.dsigmoid, w1Matrix, w2Matrix, biases1, biases2)

# Creating input and desired output for traingin neural network.
input = np.array([1.5, -4, 8])
desiredOutput = np.array([0.5, 0.4, 0.7, 0.1])


# Run the example.
def run():
    # Print outputs end error before training
    print("Desired output: ", desiredOutput)
    print("Output: ", nn.forward(input))
    print("Error: ", nn.error(input, desiredOutput))

    # Training neural network.
    print("Training...")
    for _ in range(1000):
        nn.backpropagate(input, desiredOutput, 10)

    # Print outputs and error after training.
    print("Output: ", nn.forward(input))
    print("Error: ", nn.error(input, desiredOutput))
