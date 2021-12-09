import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense,Input,Conv2D,MaxPool2D,Activation,Dropout,Flatten, BatchNormalization
from tensorflow.keras.models import Model, load_model
from keras.callbacks import Callback
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD
import datetime
import random as rn

df = pd.read_csv("data/raw_data/driver_imgs_path.csv")

tf.keras.backend.clear_session()

INPUT_SIZE = [256, 192]

# Source: https://keras.io/api/preprocessing/image/

train_gen = ImageDataGenerator(rescale = 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split = 0.3
    )

train_generator = train_gen.flow_from_directory(
    directory="./data/raw_data/imgs/train/",
    target_size=(256, 192),
    batch_size=32,
    class_mode='categorical',
    subset="training"
    )

validation_generator = train_gen.flow_from_directory(
    directory="./data/raw_data/imgs/train/",
    target_size=(256, 192),
    batch_size=32,
    class_mode='categorical',
    subset="validation"
    )

vgg16_model = tf.keras.applications.VGG16(input_shape = (256, 192, 3), include_top=False, weights="imagenet")

for layer in vgg16_model.layers:
  layer.trainable = False

input_layer = vgg16_model.output

Conv1_1 = Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='valid',data_format='channels_last',
              activation='relu',kernel_initializer=tf.keras.initializers.he_normal(seed=0),name='Conv1')(input_layer)
Batch1_1 = BatchNormalization(axis = 3)(Conv1_1)
Conv2_1 = Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='valid',data_format='channels_last',
              activation='relu',kernel_initializer=tf.keras.initializers.he_normal(seed=0),name='Conv2')(Batch1_1)
Batch2_1 = BatchNormalization(axis = 3)(Conv2_1)
Pool1_1 = MaxPool2D(pool_size=(2,2),strides=(2,2),padding='valid',data_format='channels_last',name='Pool1')(Batch2_1)

flat1 = Flatten()(Pool1_1)
Out1 = Dense(units=10,activation='softmax',kernel_initializer=tf.keras.initializers.glorot_normal(seed=3),name='Output')(flat1)
model1 = Model(inputs = vgg16_model.input, outputs = Out1)


learning_rate = 0.1
epochs = 8
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

model1.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

model1.summary()

model1.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // 32,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // 32,
    epochs = epochs
    )

model1.save("./models/model1/model1.h5")