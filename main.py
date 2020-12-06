import random
import math
import numpy as np
import matplotlib.pyplot as plt
import models.activationFunctions as af
import neural_networks.nn3layers as nn
import examples.basic_nn3layer as basicExample
import examples.separators.lineSeparator as lSep
import examples.separators.circleSeparator as cSep

"""Basic example of training neural network for desired output."""
basicExample.run()


"""Example of linear separation."""
# lSep.run(numTrainingData=10000, numValidationData=10000, learningRate=0.2)


"""Example of complex separation by circle."""
# cSep.run(numTrainingData=30000, numValidationData=10000, hiddenNeurons=30, learningRate=1.5, showBoundaries=True)

