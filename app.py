import tensorflow as tf 
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from LeNet import build_lenet
from AlexNet import build_alexnet
from VGG import build_vgg
from ResNet import build_resnet

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, X_test = X_train/255.0, X_test/255.0

if not os.path.exists("save_models"):
    os.makedirs('save_models')

def train_and_evaluate(model, model_name):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    checkpointing = tf.keras.callbacks.ModelCheckpoint(
        filepath = f"save_models/{model_name}.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
    
    model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=.2, verbose=1, callbacks=[checkpointing])
    
    model.evaluate(X_test, y_test, verbose=1)
    
train_and_evaluate(build_lenet(), 'LeNet')
train_and_evaluate(build_alexnet(), 'AlexNet')
train_and_evaluate(build_vgg(), 'VGG')
train_and_evaluate(build_resnet(), 'ResNet')