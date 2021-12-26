import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense,Input,Conv2D,MaxPool2D,Activation,Dropout,Flatten, BatchNormalization, Conv3D, MaxPool3D, Dropout, GlobalAveragePooling2D, MaxPooling3D
from tensorflow.keras.models import Model, load_model
from keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD
import datetime
import random as rn
from skimage.color import rgb2gray 
from tensorflow.keras.optimizers.schedules import ExponentialDecay


df = pd.read_csv("data/raw_data/driver_imgs_path.csv")

#p002, p052, p022, p012

tf.keras.backend.clear_session()

INPUT_SIZE = [256, 256]
BATCH_SIZE = 32
# Source: https://keras.io/api/preprocessing/image/

train_gen = ImageDataGenerator(rescale = 1./255,
    preprocessing_function = preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split = 0.3
    )

train_generator = train_gen.flow_from_directory(
    directory="./data/raw_data/imgs/train/",
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="training"
    )

validation_generator = train_gen.flow_from_directory(
    directory="./data/raw_data/imgs/train/",
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="validation"
    )


# input_layer = Flatten(name = "flat1")(vgg16_model.output)
# Batch1_1 = BatchNormalization(axis = 3)(Conv1_1)


inception_model = InceptionV3(input_shape = INPUT_SIZE + [3], include_top=False, weights="imagenet")
for layer in inception_model.layers:
  layer.trainable = False

# inception_model.summary()


input_layer = inception_model.layers[-1].output



print(input_layer.shape)

Conv1 = Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='valid',data_format='channels_last',
              activation='relu',kernel_initializer=tf.keras.initializers.he_normal(),name='Conv1')(input_layer)
Pool1 = MaxPool2D(pool_size=(2,2),strides=(2,2),padding='valid',name='Pool1')(Conv1)
flat = Flatten()(Pool1)
FC1 = Dense(units=16,activation='relu',kernel_initializer=tf.keras.initializers.he_normal(),name='FC1')(flat)
FC2 = Dense(units=32,activation='relu',kernel_initializer=tf.keras.initializers.he_normal(),name='FC2')(FC1)
drop = tf.keras.layers.Dropout(0.2)(FC2)
Out = Dense(units=10,activation='softmax',kernel_initializer=tf.keras.initializers.he_normal(),name='Output')(drop)
model1 = Model(inputs = inception_model.input, outputs = Out)
# model1 = Model(inputs = vgg16_model.input, outputs = D1)

checkpoint = ModelCheckpoint(filepath="./models/model1/model2.h5", verbose=2, save_best_only=True)
early_stop = EarlyStopping(monitor="accuracy", min_delta=0, patience=2)
callbacks = [checkpoint, early_stop]
epochs = 20
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy', metrics=['accuracy'])

model1.summary()

history1 = model1.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs = epochs,
    callbacks = callbacks
    )

# model1.save("./models/model1/model1.h5")

model1.summary()

fig, axes = plt.subplots(1, 2, figsize = (10, 5))
axes[0].plot(range(1, len(history1.history['accuracy']) + 1), history1.history['accuracy'], linestyle = 'solid', marker = 'o', color = 'crimson', label = 'Training Accuracy')
axes[0].plot(range(1, len(history1.history['val_accuracy']) + 1), history1.history['val_accuracy'], linestyle = 'solid', marker = 'o', color = 'dodgerblue', label = 'Testing Accuracy')
axes[0].set_xlabel('Epochs', fontsize = 14)
axes[0].set_ylabel('Accuracy',fontsize = 14)
axes[0].set_title('CNN Dropout Accuracy Trainig VS Testing', fontsize = 14)
axes[0].legend(loc = 'best')
axes[1].plot(range(1, len(history1.history['loss']) + 1), history1.history['loss'], linestyle = 'solid', marker = 'o', color = 'crimson', label = 'Training Loss')
axes[1].plot(range(1, len(history1.history['val_loss']) + 1), history1.history['val_loss'], linestyle = 'solid', marker = 'o', color = 'dodgerblue', label = 'Testing Loss')
axes[1].set_xlabel('Epochs', fontsize = 14)
axes[1].set_ylabel('Loss',fontsize = 14)
axes[1].set_title('CNN Dropout Loss Trainig VS Testing', fontsize = 14)
axes[1].legend(loc = 'best')

plt.show()
#######################################################################################################################################################################

vgg16_model = tf.keras.applications.VGG16(input_shape = INPUT_SIZE + [3], include_top=False, weights="imagenet")

for layer in vgg16_model.layers:
  layer.trainable = False
# vgg16_model.summary()
input_layer = vgg16_model.layers[-1].output

flat2 = tf.keras.layers.Flatten()(input_layer)
dense2_1 = tf.keras.layers.Dense(16,activation='relu')(flat2)
dense2_2 = tf.keras.layers.Dense(8,activation='relu')(dense2_1)
drop2_1=tf.keras.layers.Dropout(0.2)(dense2_2)
dense2_3=tf.keras.layers.Dense(16,activation='relu')(drop2_1)
drop2_2 = tf.keras.layers.Dropout(0.2)(dense2_3)
out2 = tf.keras.layers.Dense(10,activation='softmax')(drop2_2)

model2=tf.keras.Model(vgg16_model.input, out2)

model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // 32,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // 32,
    epochs = 10,
    callbacks = callbacks
    )

print(train_generator.image_shape)

fig, axes = plt.subplots(1, 2, figsize = (10, 5))
axes[0].plot(range(1, len(history2.history['accuracy']) + 1), history2.history['accuracy'], linestyle = 'solid', marker = 'o', color = 'crimson', label = 'Training Accuracy')
axes[0].plot(range(1, len(history2.history['val_accuracy']) + 1), history2.history['val_accuracy'], linestyle = 'solid', marker = 'o', color = 'dodgerblue', label = 'Testing Accuracy')
axes[0].set_xlabel('Epochs', fontsize = 14)
axes[0].set_ylabel('Accuracy',fontsize = 14)
axes[0].set_title('CNN Dropout Accuracy Trainig VS Testing', fontsize = 14)
axes[0].legend(loc = 'best')
axes[1].plot(range(1, len(history2.history['loss']) + 1), history2.history['loss'], linestyle = 'solid', marker = 'o', color = 'crimson', label = 'Training Loss')
axes[1].plot(range(1, len(history2.history['val_loss']) + 1), history2.history['val_loss'], linestyle = 'solid', marker = 'o', color = 'dodgerblue', label = 'Testing Loss')
axes[1].set_xlabel('Epochs', fontsize = 14)
axes[1].set_ylabel('Loss',fontsize = 14)
axes[1].set_title('CNN Dropout Loss Trainig VS Testing', fontsize = 14)
axes[1].legend(loc = 'best')