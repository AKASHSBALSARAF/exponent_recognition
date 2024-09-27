import os
import csv
import random
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Parameters
img_size = (64, 64)

# Function to preprocess images
def preprocess_image(image_path, target_size=img_size):
    image = load_img(image_path, color_mode='grayscale', target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize pixel values
    return image

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    images = np.array([preprocess_image(path) for path in data['image_path']])
    bases = np.array(data['base']) - 1  # Adjust to be zero-based
    exponents = np.array(data['exponent']) - 1  # Adjust to be zero-based
    return images, bases, exponents

