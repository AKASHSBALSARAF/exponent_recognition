import os
import csv
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Set the output directory for saving images and CSV file
output_dir = r"E:\dataset"  # Modify this path to your desired output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create a CSV file to record the data
csv_file = os.path.join(output_dir, 'dataset.csv')  # The name of the CSV file
csv_columns = ['image_path', 'base', 'exponent', 'blur', 'noise']  # Column names for the CSV

# Function to generate an image and save it
def generate_image(base, exponent, image_path, fontsize, blur, noise):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.text(0.5, 0.5, f'{base}' + r'$^{%d}$' % exponent, fontsize=fontsize, ha='center', va='center')
    ax.axis('off')
    
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Apply Gaussian blur if specified
    if blur:
        ksize = random.choice([3, 5, 7])  # Randomly choose the kernel size for blurring
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    
    # Add noise to the image if specified
    if noise:
        noise_amount = random.uniform(0.02, 0.1)  # Randomly choose the noise level
        noise_image = np.random.normal(loc=0, scale=255*noise_amount, size=image.shape)
        image = np.clip(image + noise_image, 0, 255).astype(np.uint8)
    
    # Save the generated image to the specified path
    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Open the CSV file for writing
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    
    # Generate a specified number of images
    num_images = 20  # Modify this to change the number of images generated
    for i in range(1, num_images + 1):  # Adjust the loop for the specified number of images
        base = random.randint(1, 100)  # Random base number between 1 and 100
        exponent = random.randint(1, 100)  # Random exponent number between 1 and 100
        fontsize = random.randint(20, 50)  # Varying font size between 20 and 50
        blur = random.choice([True, False])  # Randomly decide to blur the image or not
        noise = random.choice([True, False])  # Randomly decide to add noise or not
        image_path = os.path.join(output_dir, f'image_{i}.png')  # Generate the image path
        
        # Generate the image with the specified parameters
        generate_image(base, exponent, image_path, fontsize, blur, noise)
        
        # Record the data in the CSV file
        writer.writerow({
            'image_path': image_path,
            'base': base,
            'exponent': exponent,
            'blur': blur,
            'noise': noise
        })

print(f"Dataset generation complete. Images and CSV file saved to {output_dir}.")

