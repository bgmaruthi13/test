# Semantic Segmentation Model

import os
import numpy as np
import tensorflow as tf
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
IMG_SIZE = (128, 128)
DATASET_PATH = "Unet_Dataset"

# Load and preprocess data
def load_data(path):
    images = [img_to_array(load_img(f"{path}/images/{file}", target_size=IMG_SIZE)) / 255.0 for file in os.listdir(f"{path}/images")]
    masks = [img_to_array(load_img(f"{path}/masks/{file}", target_size=IMG_SIZE, color_mode="grayscale")) / 255.0 for file in os.listdir(f"{path}/masks")]
    return np.array(images), np.expand_dims(np.array(masks), -1)

images, masks = load_data(DATASET_PATH)
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# Define and compile U-Net model
model = Unet('resnet34', input_shape=(*IMG_SIZE, 3), classes=1, activation='sigmoid', encoder_weights='imagenet')
model.compile(optimizer='adam', loss=bce_jaccard_loss, metrics=[iou_score])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=8)

# Evaluate and visualize results
loss, iou = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test IoU: {iou}")

for i in range(3):  # Display 3 predictions
    plt.subplot(1, 3, 1); plt.imshow(X_test[i]); plt.title("Image")
    plt.subplot(1, 3, 2); plt.imshow(y_test[i].squeeze(), cmap='gray'); plt.title("Ground Truth")
    plt.subplot(1, 3, 3); plt.imshow(model.predict(X_test[i:i+1]).squeeze(), cmap='gray'); plt.title("Prediction")
    plt.show()
