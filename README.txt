# Exponent Recognition and Detection Using CNN

## Project Overview
This project implements a system for recognizing and detecting exponentiation in images using Convolutional Neural Networks (CNNs). The goal is to generate a dataset of images containing mathematical expressions of exponentiation, train a CNN model on this dataset, and then use the trained model to make predictions on new images.

## Table of Contents
- [Dataset Generation](#dataset-generation)
- [Model Training](#model-training)
- [Making Predictions](#making-predictions)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

## Dataset Generation

### `generate_dataset.py`
This script generates a dataset of images that depict exponentiation in a mathematical format. The dataset includes images with varying bases and exponents, as well as optional noise and blur effects.

#### How to Use `generate_dataset.py`
1. **Modify Parameters**: Adjust the following parameters in the code:
   - **Output Directory**: Set the `output_dir` variable to specify where the generated images and CSV file will be saved.
   - **Number of Images**: Change the range in the loop (`for i in range(1, N)`) to specify how many images you want to generate.
   - **Image Properties**: Customize properties like font size, blur, and noise as needed.

2. **Run the Script**: Execute the script using Python:
   ```bash
   python generate_dataset.py
