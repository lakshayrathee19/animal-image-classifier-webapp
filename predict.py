import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model("animal_model.h5")

# Path to test image
img_path = "test_images/Giraffe_3_2.jpg"  # Change this to your image

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# List of class names (should match training folders)
class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
               'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

# Predict
pred = model.predict(img_array)
predicted_index = np.argmax(pred)
predicted_class = class_names[predicted_index]

print("Predicted Animal:", predicted_class)
