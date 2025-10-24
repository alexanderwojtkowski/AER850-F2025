""" AER850: Project 2 """
# Name: Alexander Wojtkowski
# Student #: 501168859

# Due Date: November 5th, 2025

""" Imports """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore", category=ConvergenceWarning)

""" Part 1: Data Processing - 20 marks """
# Define Image Shape (500,500,3)

# Establish Train and Validation Data Directories

# Perform Data Augmentation (Re-scaling, shear range, zoom range)
# Use Keras' Image Preprocessing Pipeline or Torchvision Transforms

# Create Train and Validation Generator Using Keras' imagedatasetfromdirectory
# or PyTorch's Dataloader

""" Part 2: Neural Network Architecture Design - 30 marks """
# We now build custom neural network

# Convolutional Layers (Conv2D) Explore: # of filters, kernal size, stride

# MaxPooling2D Layer, pool extracted features

# Flatten Layer

# Dense and Dropout Layers, to perform final predictions
# Final dense should only have 3 neurons, correlating to label classes

""" Part 3: Hyperparameter Analysis - 20 marks """
# Common Functions within Convolution layers are relu and LeakyRelu

# Common Functions within Fully connected dense layers are usually relu or elu
# For final layer for multi-class classification is softmax

# Number of neurons or filters for the dense and convolutional layers respectively
# Filters should be base 2, neurons should be varied based on performance

# Last Parameters should be loss function and optimizers usually within
# Keras' compile function
# Loss Function and Optimizer to start would be catergorical crossentropy and adma, respectively

""" Part 4: Model Evaluation - 10 marks """
# Evaluate the loss and accuracy performance of the model
# They should act inversely to eachother


