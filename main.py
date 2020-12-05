import random
import math
import numpy as np
import matplotlib.pyplot as plt
import models.activation_functions.act_func as af
import neural_networks.nn3layers as nn
import examples.lineSeparator as ls
import examples.basic_nn3layer as be
import examples.complexSeparator as cs
import models.dataGenerator as data
import examples.separators.circleSeparator as css

css.run(numTrainingData=20000, numValidationData=1000, hiddenNeurons=8, learningRate=2, showBoundaries=True)
