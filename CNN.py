import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from matplotlib import pyplot
from flask import Flask, request, jsonify
import random
import tkinter as tk
from PIL import Image, ImageTk
from io import BytesIO
from threading import Thread
from queue import Queue


train_datagen = ImageDataGenerator(
    rescale=1./255.,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)

training_set = train_datagen.flow_from_directory(
   '/content/machine-learning/img/digits/two-digitsData/train',
   target_size=(28,28),
   batch_size=32,
   class_mode='categorical',
   color_mode='grayscale'
)
print(training_set.class_indices)

test_set = train_datagen.flow_from_directory(
   '/content/machine-learning/img/digits/two-digitsData/test',
   target_size=(28,28),
   batch_size=32,
   class_mode='categorical',
   color_mode='grayscale' 
)

model = Sequential()
model.add(Conv2D(filters=5,
                 kernel_size=3,
                 activation='relu',
                 input_shape=[28,28,1])) 
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Flatten())

model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=2, activation='softmax')) 
adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(training_set, epochs=1)

model.evaluate(test_set)

def singlePredict(path):
  test_image = image.load_img(path,
                              target_size=[28,28], color_mode='grayscale')
  test_image = image.img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis=0)
  result = model.predict(test_image/255.)
  print(result)

singlePredict('/content/machine-learning/img/digits/two-digitsData/train/2/10016.jpg')

