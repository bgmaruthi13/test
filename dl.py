# Model 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define dataset path
dataset_path = "3_food_classes"

# Image parameters
img_height, img_width = 128, 128  # Larger image size for better feature extraction
batch_size = 32

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Build Improved Model
model = Sequential()

# Add layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes

# Model Summary
model.summary()

# Compile the Model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(
    'best_model_weights.h5',  # Save the weights of the best model
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,  # Stop training if no improvement for 10 epochs
    verbose=1,
    restore_best_weights=True
)

# Train the Model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping]
)



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
