import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense,Input,Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tqdm import tqdm
from tensorflow.keras.optimizers.schedules import ExponentialDecay

tf.keras.backend.clear_session()

df = pd.read_csv("data/raw_data/driver_imgs_path.csv")

# Reading and Processing image... 
def img_process(df):
    data = []
    for i in tqdm(range(10)):
        imgs = os.listdir("./data/raw_data/imgs/train/c"+str(i))
        for j in range(len(imgs)):
            img_name = "./data/raw_data/imgs/train/c"+str(i)+"/"+imgs[j]
            img = cv2.imread(img_name) # Reading img...
            #img = color.rgb2gray(img) # RGB to Grayscale...
            img = img[50:,100:-50] # Zooming in on the dirver...
            img = cv2.resize(img,(224,224)) # Resizing the img to 224x224...
            driver = df[df['img'] == imgs[j]]['subject'].values[0] # driver's id example: p002...
            data.append([img,i,driver])
    return data

# data = img_process(df)

# with open('./data/raw_data/data.txt', 'wb') as f:
#         pickle.dump(data, f)

data = pd.read_pickle(r'./data/raw_data/data.txt')

test_pid = ["p045", "p075", "p022", "p012"]
X_train, X_test, y_train, y_test = [], [], [], []
for i in data:
    if i[2] in test_pid:
        X_test.append(i[0])
        y_test.append(i[1])
    else: 
        X_train.append(i[0])
        y_train.append(i[1])

X_train = np.array(X_train).reshape(-1,224,224,3)
X_test = np.array(X_test).reshape(-1,224,224,3)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


INPUT_SIZE = [224, 224]
BATCH_SIZE = 64

# Model from scratch... following vgg16 architecture...
# Using keras sequential api...
model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.5))
model1.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.5))
model1.add(Conv2D(128, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(128, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(128, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.5))
model1.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.5))
model1.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Flatten())
model1.add(Dense(64, activation = 'relu'))
model1.add(Dense(64, activation='relu'))
model1.add(Dense(10, activation='softmax'))
opt = Adam(learning_rate = 0.001)
model1.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])


# Source: https://keras.io/api/preprocessing/image/

train_gen = ImageDataGenerator(rescale = 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    )

train_generator = train_gen.flow(
    X_train,
    y_train,
    batch_size=BATCH_SIZE
)

checkpoint1 = ModelCheckpoint(filepath="./models/model1/model1.h5", verbose=2, save_best_only=True)
early_stop = EarlyStopping(monitor="accuracy", min_delta=0, patience=2)
callbacks1 = [checkpoint1, early_stop]
model1_trained = model1.fit(
    train_generator, 
    steps_per_epoch = len(X_train) / BATCH_SIZE, 
    callbacks=callbacks1,
    epochs = 25, 
    validation_data = (X_test, y_test))


fig, axes = plt.subplots(1, 2, figsize = (10, 5))
axes[0].plot(range(1, len(model1_trained.history['accuracy']) + 1), model1_trained.history['accuracy'], linestyle = 'solid', marker = 'o', color = 'crimson', label = 'Training Accuracy')
axes[0].plot(range(1, len(model1_trained.history['val_accuracy']) + 1), model1_trained.history['val_accuracy'], linestyle = 'solid', marker = 'o', color = 'dodgerblue', label = 'Testing Accuracy')
axes[0].set_xlabel('Epochs', fontsize = 14)
axes[0].set_ylabel('Accuracy',fontsize = 14)
axes[0].set_title('CNN Dropout Accuracy Trainig VS Testing', fontsize = 14)
axes[0].legend(loc = 'best')
axes[1].plot(range(1, len(model1_trained.history['loss']) + 1), model1_trained.history['loss'], linestyle = 'solid', marker = 'o', color = 'crimson', label = 'Training Loss')
axes[1].plot(range(1, len(model1_trained.history['val_loss']) + 1), model1_trained.history['val_loss'], linestyle = 'solid', marker = 'o', color = 'dodgerblue', label = 'Testing Loss')
axes[1].set_xlabel('Epochs', fontsize = 14)
axes[1].set_ylabel('Loss',fontsize = 14)
axes[1].set_title('CNN Dropout Loss Trainig VS Testing', fontsize = 14)
axes[1].legend(loc = 'best')

plt.show()

#######################################################################################################################################################################

# Transfer learning using vgg16...

tf.keras.backend.clear_session()

INPUT_SIZE = [224, 224]
BATCH_SIZE = 64

X_train = np.array(X_train).reshape(-1,224,224,3)
X_test = np.array(X_test).reshape(-1,224,224,3)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


train_gen = ImageDataGenerator(
    height_shift_range=0.3,
    width_shift_range = 0.3,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3
    )

train_generator = train_gen.flow(
    X_train,
    y_train,
    batch_size=BATCH_SIZE
)

checkpoint2 = ModelCheckpoint(filepath="./models/model2/with_conv.h5", verbose=2, save_best_only=True)
early_stop = EarlyStopping(monitor="accuracy", min_delta=0, patience=5)
callbacks2 = [early_stop, checkpoint2]
input = Input(name = 'img_input', shape=(224, 224, 3))
vgg16_model = VGG16(input_tensor = input, include_top=False, weights="imagenet")

# for layer in vgg16_model.layers:
#   layer.trainable = False
# vgg16_model.summary()

input_layer = vgg16_model(input)

conv = Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu', 
          kernel_initializer=tf.keras.initializers.he_normal())(input_layer)
pool = GlobalAveragePooling2D()(conv)
d1 = Dense(units=1024,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(pool)
drop1 = Dropout(0.5)(d1)
d2 = Dense(units=512,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(drop1)
bn = BatchNormalization()(d2)
drop2 = Dropout(0.2)(bn)
d3 = Dense(units=512,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(drop2)
Out2 = Dense(units=10,activation='softmax', kernel_initializer=tf.keras.initializers.he_normal())(d3)
model2=tf.keras.Model(input, Out2)

sgd = SGD(learning_rate=0.001)
model2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model2.fit(
    train_generator, 
    steps_per_epoch = len(X_train) / BATCH_SIZE, 
    callbacks=callbacks2,
    epochs = 20, 
    validation_data = (X_test, y_test))

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
plt.show()