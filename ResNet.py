import tensorflow as tf 
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

def residual_block(x, filters):
    shortcut = x 
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def build_resnet():
    inputs = tf.keras.Input((32, 32, 3))
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = residual_block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    
    return model