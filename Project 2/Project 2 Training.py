""" AER850: Project 2 """
# Name: Alexander Wojtkowski
# Student #: 501168859

# Due Date: November 5th, 2025

""" IMPORTS """

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

""" VARIABLES """

np.random.seed(42)
keras.utils.set_random_seed(42)

MODEL_NAME = "Model_5.keras"

IMG_HEIGHT = 500
IMG_WIDTH = 500

EPOCHS = 30

BATCH_SIZE = 32

MODEL_SELECTION = '2' # 1 for the first model, 2 for the other

""" Part 1: Data Processing - 20 marks """
# Define Image Shape (500,500,3)
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
    keras.layers.RandomZoom(0.1)
])

# Create Train and Validation Generator Using Keras' imagedatasetfromdirectory
# or PyTorch's Dataloader
train_ds = keras.utils.image_dataset_from_directory(
    training_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, # Use the new BATCH_SIZE variable
    shuffle=True
)

valid_ds = keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization_layer(x)), y))
valid_ds = valid_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

# Optimize data pipeline performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

""" Part 2: Neural Network Architecture Design - 30 marks """
"""                        &                              """
"""       Part 3: Hyperparameter Analysis - 20 marks      """

if MODEL_SELECTION == '1':
    print("Training Model 1")

    model = keras.Sequential([
        keras.layers.Input(shape=IMG_Shape),

        keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2,2)),

        keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2,2)),

        keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2,2)),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(3, activation='softmax')
    ])

elif MODEL_SELECTION == '2':
    print("Training Model 2")

    model = keras.Sequential([
        keras.layers.Input(shape=IMG_Shape),

        keras.layers.Conv2D(32, (5,5), padding='same'),
        keras.layers.LeakyReLU(negative_slope=0.1),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2,2)),

        keras.layers.Conv2D(64, (5,5), padding='same'),
        keras.layers.LeakyReLU(negative_slope=0.1),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2,2)),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.35),
        keras.layers.Dense(3, activation='softmax')
    ])

else:
    raise ValueError("Invalid MODEL_SELECTION value. Choose '1' or '2'.")

model.summary()

optimizer = keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks = [early_stop],
    verbose=1
)

model.save(MODEL_NAME)

loss, acc = model.evaluate(valid_ds)
print(f"Validation Accuracy: {acc:.2f}")

""" Part 4: Model Evaluation - 10 marks """
# Plot Training and Validation Performance
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()