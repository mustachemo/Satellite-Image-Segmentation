"This file will contain the object detection model."

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

def build_model(input_shape=(1280, 720, 3), num_classes=2):
    print(f'Tensorflow version:  {tf.__version__}\nGPU available: {tf.test.is_built_with_cuda}')

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Additional convolutional layers for feature refinement
    # x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    # x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    # Bounding Box Regression
    bbox_regression = tf.keras.layers.Conv2D(4, (3, 3), activation='linear', padding='same')(x)
    bbox_regression = tf.keras.layers.Reshape((-1, 4))(bbox_regression)  # Reshape to (batch_size, -1, 4)
    
    # Classification
    classification = tf.keras.layers.Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(x)
    classification = tf.keras.layers.Reshape((-1, num_classes))(classification)  # Reshape to (batch_size, -1, num_classes)
    
    # Concatenate the outputs
    outputs = tf.keras.layers.Concatenate(axis=-1)([bbox_regression, classification])
    
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def od_loss(y_true, y_pred):
    y_true_regression, y_true_classification = tf.split(y_true, [4, -1], axis=-1)
    y_pred_regression, y_pred_classification = tf.split(y_pred, [4, -1], axis=-1)
    
    # Define the localization (bounding box regression) loss
    localization_loss = MeanSquaredError()(y_true_regression, y_pred_regression)
    
    # Define the classification loss (binary cross-entropy)
    classification_loss = BinaryCrossentropy()(y_true_classification, y_pred_classification)
    
    # Total loss
    total_loss = localization_loss + classification_loss

    return total_loss

# Compile your model
model = build_model()
model.compile(optimizer='adam', loss=od_loss)