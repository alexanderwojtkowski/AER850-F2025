""" AER850: Project 2 """
# Name: Alexander Wojtkowski
# Student #: 501168859

# Due Date: November 5th, 2025

""" Imports """

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

""" VARIABLES """

np.random.seed(42)
keras.utils.set_random_seed(42)

MODEL_NAME = "Proj2_Mdl1.keras"

IMG_HEIGHT = 500
IMG_WIDTH = 500

""" Part 5: Model Testing - 20 marks """
# Data processing of the test images to the format of the input data
# Use image package from keras preprocessing to load, convert to an array
# and normalize by dividing by 255

# Since its multi-class, final model layer uses softmax, need to find the
# maximum probability from the model prediction

# Final prediction should look like one provided in assignment

model = keras.models.load_model(MODEL_NAME)

# Define Image Shape (500,500,3)
IMG_Shape = (IMG_HEIGHT,IMG_WIDTH,3) # Height Width Channels

# Establish Test Data Directories
testing_dir = "Data/test"

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
test_ds = keras.utils.image_dataset_from_directory(
    testing_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    shuffle=True
)

class_names = test_ds.class_names
print("Class names:", class_names)

test_ds = test_ds.map(lambda x, y: (data_augmentation(normalization_layer(x)), y))

loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.2f}")

plt.figure(figsize=(10, 10))

# Loop through a few test samples
for images, labels in test_ds.take(5):
    predictions = model.predict(images)
    pred_probs = tf.nn.softmax(predictions[0]).numpy()
    
    predicted_idx = np.argmax(pred_probs)
    true_idx = labels.numpy()[0]
    
    true_label = class_names[true_idx]
    predicted_label = class_names[predicted_idx]
    
    # Plot image
    plt.figure(figsize=(5,5))
    plt.imshow(images[0].numpy())
    plt.axis("off")

    # Titles
    plt.title(f"True Crack Classification Label: {true_label}\n"
              f"Predicted Crack Classification Label: {predicted_label}",
              fontsize=10, pad=20)

    # Display probabilities in bottom-left corner
    text_y = 460  # adjust based on image size
    for i, cls in enumerate(class_names):
        plt.text(10, text_y - (i * 20), 
                 f"{cls.capitalize()}: {pred_probs[i]*100:.1f}%", 
                 color='lime', fontsize=10, weight='bold')

    plt.show()