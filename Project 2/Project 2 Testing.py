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

""" Part 5: Model Testing - 20 marks """
# Data processing of the test images to the format of the input data
# Use image package from keras preprocessing to load, convert to an array
# and normalize by dividing by 255

# Since its multi-class, final model layer uses softmax, need to find the
# maximum probability from the model prediction

# Final prediction should look like one provided in assignment