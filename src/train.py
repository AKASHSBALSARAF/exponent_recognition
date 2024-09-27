
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np

from .data_loader import load_dataset
from .model import build_model

# Load and preprocess dataset
file_path = r'E:\TEMP2\exponent_dataset\dataset.csv'  # Update path as needed
images, bases, exponents = load_dataset(file_path)

# Split the dataset
X_train, X_test, y_train_base, y_test_base, y_train_exp, y_test_exp = train_test_split(
    images, bases, exponents, test_size=0.2, random_state=42)

# Build the model
model = build_model()

# Define callbacks for early stopping and saving the best model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(filepath=r'E:\TEMP2\exponent_model_best.keras', save_best_only=True, monitor='val_loss')
]

# Train the model
history = model.fit(X_train, {'base_output': y_train_base, 'exponent_output': y_train_exp},
                    validation_data=(X_test, {'base_output': y_test_base, 'exponent_output': y_test_exp}),
                    epochs=50, batch_size=32, callbacks=callbacks)

# Save the final model locally
model_save_path = r'E:\TEMP2\exponent_model_final.keras'  # Update path as needed
model.save(model_save_path)
