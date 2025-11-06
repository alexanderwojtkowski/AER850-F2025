""" AER850: Project 2 """
# Name: Alexander Wojtkowski
# Student #: 501168859

# Due Date: November 5th, 2025

""" Imports """

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

""" VARIABLES """

np.random.seed(42)
keras.utils.set_random_seed(42)

MODEL_NAME = "Final_Model_2.keras"

IMG_HEIGHT = 500
IMG_WIDTH = 500

""" Part 5: Model Testing - 20 marks """

model = keras.models.load_model(MODEL_NAME)

# Define Image Shape (500,500,3)
IMG_Shape = (IMG_HEIGHT,IMG_WIDTH,3) # Height Width Channels

# Establish Test Data Directories
testing_dir = "Data/test"

normalization_layer = keras.layers.Rescaling(1/255)

test_ds = keras.utils.image_dataset_from_directory(
    testing_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    shuffle=True
)

class_names = test_ds.class_names
print("Class names:", class_names)

test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.2f}")

# Paths to specific test images
specific_images = {
    "crack": "Data/test/crack/test_crack.jpg",
    "missing-head": "Data/test/missing-head/test_missinghead.jpg",
    "paint-off": "Data/test/paint-off/test_paintoff.jpg"
}

plt.figure(figsize=(10, 10))

for class_name, img_path in specific_images.items():
    # Load and preprocess image
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 500, 500, 3)
    img_array = img_array / 255.0  # normalize

    # Predict
    predictions = model.predict(img_array)
    pred_probs = tf.nn.softmax(predictions[0]).numpy()
    predicted_idx = np.argmax(pred_probs)
    predicted_label = class_names[predicted_idx]

    # Plot image
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")

    plt.title(f"True Crack Classification Label: {class_name}\n"
              f"Predicted Crack Classification Label: {predicted_label}",
              fontsize=10, pad=20)

    # Display probabilities for all classes
    text_y = 460
    for i, cls in enumerate(class_names):
        plt.text(10, text_y - (i * 20),
                 f"{cls.capitalize()}: {pred_probs[i]*100:.1f}%",
                 color='lime', fontsize=10, weight='bold')

    plt.show()