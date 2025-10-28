""" AER850: Project 2 """
# Name: Alexander Wojtkowski
# Student #: 501168859

# Due Date: November 5th, 2025

""" IMPORTS """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

""" VARIABLES """

np.random.seed(42)
keras.utils.set_random_seed(42)

TRAIN_MODEL = 0 # 1 to train model, 0 to load model

""" Part 1: Data Processing - 20 marks """
# Define Image Shape (500,500,3)
IMG_HEIGHT = 500
IMG_WIDTH = 500
IMG_Shape = (IMG_HEIGHT,IMG_WIDTH,3) # Height Width Channels

# Establish Train and Validation Data Directories
training_dir = "Data/train"
validation_dir = "Data/valid"

# Perform Data Augmentation (Re-scaling, shear range, zoom range)
# Use Keras' Image Preprocessing Pipeline or Torchvision Transforms

normalization_layer = keras.layers.Rescaling(1/255)

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

# Create Train and Validation Generator Using Keras' imagedatasetfromdirectory
# or PyTorch's Dataloader
train_ds = keras.utils.image_dataset_from_directory(
    training_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    shuffle=True
)

valid_ds = keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    shuffle=False
)

train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization_layer(x)), y))
valid_ds = valid_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

# Optimize data pipeline performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)
    

if TRAIN_MODEL == 1:
    """ Part 2: Neural Network Architecture Design - 30 marks """
    # We now build custom neural network
    model = keras.Sequential([
        keras.layers.Input(shape=IMG_Shape),
        
    # Convolutional Layers (Conv2D) Explore: # of filters, kernal size, stride
    # MaxPooling2D Layer, pool extracted features
        keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        
        keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        
        keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    # Flatten Layer
        keras.layers.Flatten(),
        
    # Dense and Dropout Layers, to perform final predictions
    # Final dense should only have 3 neurons, correlating to label classes
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
    
        keras.layers.Dense(3, activation='softmax')
        ])
    
    model.summary()
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # because labels are integer-encoded by image_dataset_from_directory
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=10
    )
    
    model.save('First_Model.keras')
    
    loss, acc = model.evaluate(valid_ds)
    print(f"Validation Accuracy: {acc:.2f}")
else:
    loaded_model = load_model('First_Model.keras')
    loss, acc = loaded_model.evaluate(valid_ds)
    print(f"Validation Accuracy: {acc:.2f}")

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


