import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model_path = 'models/exponent_model_final.keras'  # Adjust if needed
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image
def preprocess_image(image_path, target_size=(64, 64)):
    image = load_img(image_path, color_mode='grayscale', target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize pixel values
    return image

# Function to make predictions
def predict_exponential(image_path, model):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    base_pred, exp_pred = model.predict(image)
    base = np.argmax(base_pred) + 1  # Adjust back to 1-based
    exponent = np.argmax(exp_pred) + 1  # Adjust back to 1-based
    return base, exponent

# Example prediction
random_image_name = 'data/image_1.png'  # Change this to the image you want to test
base, exponent = predict_exponential(random_image_name, model)
print(f'Base: {base}, Exponent: {exponent}')
