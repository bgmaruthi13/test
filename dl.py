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
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

image_dir='Unet_Dataset/images/'
mask_dir='Unet_Dataset/MASKS_BW/'

# Function to load all images into a numpy array
from tensorflow.keras.preprocessing.image import load_img, img_to_array
def load_images_from_folder(folder, target_size):
    images = []
    for filename in sorted(os.listdir(folder)):  # Sorting ensures correct pairing
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)

# Parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
TOTAL_IMAGES = 150

# Step 1: Load all the images
images = load_images_from_folder(image_dir, target_size=(IMG_HEIGHT, IMG_WIDTH))
masks = load_images_from_folder(mask_dir, target_size=(IMG_HEIGHT, IMG_WIDTH))
masks = np.expand_dims(masks[..., 0], axis=-1)  # Keep masks binary (128x128x1)

len(images), len(masks)

from sklearn.model_selection import train_test_split

# 2. Scale images and masks
images = images / 255.0  # Scale images to range [0, 1]
masks = masks / 255.0    # Scale masks to range [0, 1]

# 3. Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)
X_train.shape, X_test.shape

# pip install if this doesn't work
# Required for newer versions of TensorFlow
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

# 4. Define the pre-trained segmentation model (U-Net)
BACKBONE = 'resnet34'  # Using ResNet34 backbone
preprocess_input = sm.get_preprocessing(BACKBONE)

# Preprocess data
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

# Define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), classes=1, activation='sigmoid')

# 5. Compile the model
model.compile(optimizer='adam',
              loss=sm.losses.bce_jaccard_loss,  # Binary crossentropy + Jaccard loss
              metrics=[sm.metrics.iou_score])  # Intersection over Union (IoU)
