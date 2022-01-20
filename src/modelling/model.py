#############################################################################################################################################################################################################
#                                                                                  F I L E   D E S C R I P T I O N                                                                                          #
#############################################################################################################################################################################################################
#                                                                                                                                                                                                           #
# This file contains the code related to Data manipulation such as dividing data into test and train, image augmentation, feature extraction and training of models like traditional machine learning       #
# algorithm (Random Forest) , Convolution Neural Network trained from scratch, Convolution Neural Network trained with transfer learning using pre-trained VGG16 model. Plotting of accuracy and loss for   #
# training and testing has also been done here and plots are saved so that it can be displayed in the nootbook.                                                                                             #
#############################################################################################################################################################################################################


# Importing relvant packages...
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import cv2
import glob
from sklearn import metrics
from skimage.filters import sobel
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense,Input,Conv2D, Dropout, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier


# Function to plot CNN Accuracy and Loss for Training vs Testing...
def plotter(history):
    fig, axes = plt.subplots(1, 2, figsize = (10, 5)) # fig of 1 row and 2 cols with 10x5 size...
    # In First column of figure, plotting accuracy and val accuracy from trained model object(history)...
    axes[0].plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], linestyle = 'solid', marker = 'o', color = 'crimson', label = 'Training Accuracy') 
    axes[0].plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], linestyle = 'solid', marker = 'o', color = 'dodgerblue', label = 'Testing Accuracy')
    axes[0].set_xlabel('Epochs', fontsize = 14)
    axes[0].set_ylabel('Accuracy',fontsize = 14)
    axes[0].set_title("CNN Dropout Accuracy Training vs Testing", fontsize = 14)
    axes[0].legend(loc = 'best') # Location of the legend, whereever is more empty space put legent there (loc= 'best')...
    # In Second column of figure, plotting accuracy and val accuracy from trained model object(history)...
    axes[1].plot(range(1, len(history.history['loss']) + 1), history.history['loss'], linestyle = 'solid', marker = 'o', color = 'crimson', label = 'Training Loss')
    axes[1].plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], linestyle = 'solid', marker = 'o', color = 'dodgerblue', label = 'Testing Loss')
    axes[1].set_xlabel('Epochs', fontsize = 14)
    axes[1].set_ylabel('Loss',fontsize = 14)
    axes[1].set_title("CNN Dropout Loss Training vs Testing", fontsize = 14)
    axes[1].legend(loc = 'best')
    plt.show()


#############################################################################################################################################################################################################
############################################################################ TRADITIONAL MACHINE LEARNING ALGORITHM (RANDOM FOREST) #########################################################################
#############################################################################################################################################################################################################

df = pd.read_csv("data/driver_imgs_path.csv") # importing csv file that contains subject id, classname and img name...

# function to Read and Process the images... 
def img_process(df):
    data = []
    for i in tqdm(range(10)): # 10 directories for 10 classes...
        imgs = os.listdir("./data/imgs/train/c"+str(i)) # all the image names from 1 directory is stored in imgs...
        for j in range(len(imgs)):
            img_name = "./data/imgs/train/c"+str(i)+"/"+imgs[j]
            img = cv2.imread(img_name) # Reading img...
            #img = color.rgb2gray(img) # RGB to Grayscale...
            img = img[50:,100:-50] # Zooming in on the dirver...
            img = cv2.resize(img,(224,224)) # Resizing the img to 224x224...
            driver = df[df['img'] == imgs[j]]['subject'].values[0] # driver's id example: p002...
            data.append([img,i,driver])
    return data

data = img_process(df) # processing all the images...

# storing the processed images as pickle file...
with open('./data/data.txt', 'wb') as f:
        pickle.dump(data, f)

# reading the processed data...
data = pd.read_pickle(r'./data/data.txt')

# Splitting the data into test and train sets based on person Id...
# Keeping "p045", "p075", "p022", "p012" in test and rest in train set...
test_pid = ["p045", "p075", "p022", "p012"]
X_train, X_test, y_train, y_test = [], [], [], []
for i in data:
    if i[2] in test_pid:
        X_test.append(i[0])
        y_test.append(i[1])
    else: 
        X_train.append(i[0])
        y_train.append(i[1])

# Reshaping the test and train to 224x224 with 3 channels (rgb)
X_train = np.array(X_train).reshape(-1,224,224,3)
X_test = np.array(X_test).reshape(-1,224,224,3)

# Normalizing the pixel values to range from 0 to 1...
X_train = (X_train*1.0)/255.0
X_test = (X_test*1.0)/255.0


# Function to extract features from the image dataset...
# dataset: images from which features need to be extracted...
def feature_extractor(dataset):
    
    # placeholder for final features... (final dataset)
    image_dataset = pd.DataFrame() 
    
    # iterating over all the images...
    for i in tqdm(range(np.array(dataset).shape[0])):  
        feat_df = pd.DataFrame()  # placeholder for features of one image...
        input_img = np.array(dataset)[i, :,:,:] # i = image index, all image rows, all image cols, all the channels...
        pix_val = input_img.reshape(-1) # making the 2 dimentional img to 1 dimentonal
        feat_df['Pixel_Value'] = pix_val  # storing them in a feat_df dataframe under column name 'Pixel_value'
        counter = 1 # gabor counter...
        kernels = [] # placeholder for calculated filter values...
        for theta in range(2): # using 3 different values of theta...  
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  # using 2 values of sigma...
                lamda = np.pi/4 
                gamma = 0.5
                gabor_label = 'Gabor' + str(counter) # column name as "Gabor1" , "Gabor2" and so on...
                
                # calculating kernel values based on sigma, theta, lambda, gama, and shape of kernel (9,9)...
                # CV_32FC2 is a 32-bit, floating-point, and 2-channels structure...
                kernel = cv2.getGaborKernel((9, 9), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)   
                
                kernels.append(kernel) # storing the kernel/filter values...
                
                # Applying the gabor filter on to the image calculated in the previous step...
                # CV_8UC3 is an 8-bit unsigned integer matrix(2D)/image with 3 channels...
                fimg = cv2.filter2D(input_img, cv2.CV_8UC3, kernel) 
                fimg = fimg.reshape(-1) # reshaping it to 1D...
                feat_df[gabor_label] = fimg # adding to data frame...
                counter += 1  
                
        # applying sobel filter for edge detectection...
        edge_sobel = sobel(input_img)
        edge_sobel1 = edge_sobel.reshape(-1)
        feat_df['Sobel'] = edge_sobel1
        image_dataset = image_dataset.append(feat_df)
        
    # returning dataset with 3 type of features, 1: pixel values, 2: gabor filter, 3: sobel edge detector...
    return image_dataset

# Randomly selecting 4000 images/data points as training set to reduce the computation overhead...
indx = np.random.choice(X_train.shape[0], size=4000, replace=False)

# converting the target varial to categorical...
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# selecting rows based on 4000 indx chosen previously...
X_train_subset = X_train[indx]
y_train_subset = y_train_cat[indx]

# Saving all the data required for trainig and testing into pickle file...
with open('./data/X_train_4000.txt', 'wb') as f:
         pickle.dump(X_train_subset, f)
with open('./data/y_train_4000.txt', 'wb') as f:
         pickle.dump(y_train_subset, f)
image_features = feature_extractor(X_train_subset)
with open('./data/feature.txt', 'wb') as f:
         pickle.dump(image_features, f)


# Reading relevant data...
image_features = pd.read_pickle(r'./data/feature.txt')
y_train = pd.read_pickle(r'./data/y_train_4000.txt')


image_features = np.expand_dims(image_features, axis=0) # converting long to wide data... (changing rows to columns as axis = 0)
X_train_RF = np.reshape(image_features, (4000, -1))  # Reshape to #images, features

# Initializing the random forest classifier with number of trees in the forest = 50 and random state can be any number... but wondering why 42? 
# Take a look at this article... something everybody should know... :p https://hitchhikers.fandom.com/wiki/42 
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42) 

# fitting the model...
RF_model.fit(X_train_RF, y_train) 

# Saving the model...
with open('./models/model/model', 'wb') as f:
         pickle.dump(RF_model, f)

# choosing 500 data points/images for testing...
test_indx = np.random.choice(np.array(X_test).shape[0], size=500, replace=False)
X_test_subset = [X_test[i] for i in test_indx]
y_test_subset = np.array(y_test)[test_indx]

# saving labels and images as pickle file...
with open('./data/X_test.txt', 'wb') as f:
         pickle.dump(X_test_subset, f)
with open('./data/y_test.txt', 'wb') as f:
         pickle.dump(y_test_subset, f)

#############################################################################################################################################################################################################
####################################################################################### CNN FROM sCRATCH (VGG16 ARCHITECTURE) ###############################################################################
#############################################################################################################################################################################################################

df = pd.read_csv("data/driver_imgs_path.csv") # importing csv file that contains subject id, classname and img name...

tf.keras.backend.clear_session() # Removes the values in the graph(network connections) but do not delete the graph itself... helps in RAM cleaning...

# Splitting the data into test and train sets based on person Id...
# Keeping "p045", "p075", "p022", "p012" in test and rest in train set...
test_pid = ["p045", "p075", "p022", "p012"]
cnn_test = df[df['subject'].isin(test_pid)]
cnn_train = df[~df['subject'].isin(test_pid)]

INPUT_SIZE = [224, 224] # Image size...
BATCH_SIZE = 32 # Number of data points in each batch...

# Model from scratch... following vgg16 architecture...
# Using keras sequential api...
model1 = Sequential() # initializing the model...

# Convolution layer with 32 filters of size 3x3, activation function used is ReLU and "same" padding to keep the input and output size same...
model1.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(MaxPooling2D(pool_size=(2, 2))) # Maxpooling with 2x2 moving window... takes the maximum of 4 values present in the window...
model1.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.2)) # Dropout layer to avoid overfitting...
model1.add(Conv2D(128, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(128, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(128, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.2))
model1.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3),padding = 'same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Flatten()) # Flatten layer... converts the multi-dimentional tensor to 1-D tensor... which is suitable for dense layer...
model1.add(Dense(256, activation = 'relu')) # dense layer with 256 nodes to learn patterns from the features extracted from previous layers...
model1.add(Dense(128, activation='relu'))
model1.add(Dense(10, activation='softmax')) # Output layer with 10 nodes... (equal to number of classes)
opt = Adam(learning_rate = 0.001) # Adam optimizer with learning rate 0.001...
model1.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy']) # model with categorical_crossentropy loss and evaluation metric as accuracy...


# Source: https://keras.io/api/preprocessing/image/


# ImageDataGenerator is a handy api from keras to read images and perform image augmentation...
datagen=ImageDataGenerator(rescale=1./255., # normalizing pixel values to range from 0 to 1...
    height_shift_range=0.3, # Shifting image vertically...
    width_shift_range = 0.3, # Shifting image horizontally...
    rotation_range=30, # Rorating image by 30 degree...
    shear_range=0.3, # stretching image from one end...
    zoom_range=0.3, # Zooming on the image...
    validation_split=0.25) # using 25% of the images as validation data...

# using ImageDataGenerator object to read data from the dataframe cnn_train...
train_generator=datagen.flow_from_dataframe(
    dataframe=cnn_train, # dataframe from where the columns having image path and labels need to be read...
    directory="./data/imgs/train", # path where all the class directories of the images are present... 
    x_col="full_path", # column name having image path...
    y_col="classname", # column with labels...
    subset="training", # training set...
    batch_size=32, # 32 datapoints in each batch...
    seed=42, # random seed with 42... remember? 42 is important... Here's the link again if you missed last time... https://hitchhikers.fandom.com/wiki/42
    shuffle=True, # Jumble the data points...
    class_mode="categorical", # solving categorical problem... 
    target_size=(224, 224) # image size... 224x224...
)

# constructing validation set generator... will be used for validation of the of the model...
valid_generator=datagen.flow_from_dataframe(
    dataframe=cnn_train,
    directory="./data/imgs/train",
    x_col="full_path",
    y_col="classname",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(224,224)
)

# ModelCheckpoint callback is used to save only the best model out of all the epochs...
checkpoint1 = ModelCheckpoint(filepath="./models/model1/model1.h5", verbose=2, save_best_only=True)

# EarlyStopping callback is used to stop the training when accuracy doesn't improve for 5 epochs...
early_stop = EarlyStopping(monitor="accuracy", min_delta=0, patience=5)

callbacks1 = [checkpoint1, early_stop]

# Fitting/training model...
model1_trained = model1.fit(
    train_generator, 
    steps_per_epoch =  train_generator.n/ BATCH_SIZE, # number of steps in each epoch...
    callbacks=callbacks1, # Callback is an object that can perform several operatrions like stopping model early from training, write to tensorboard...
    epochs = 10, # train model 10 times...
    validation_data = valid_generator) # validation set...

# plotting the train, test accuracy and loss...
plotter(model1_trained)

#############################################################################################################################################################################################################
###################################################################### CNN WITH TRANSFER LEARNING USING VGG16 PRE-TRAINED MODEL #############################################################################
#############################################################################################################################################################################################################

 # Removes the values in the graph(network connections) but do not delete the graph itself... helps in RAM cleaning...
tf.keras.backend.clear_session()

# Loading data...
data = pd.read_pickle(r'./data/data.txt')

# Splitting the data into test and train sets based on person Id...
# Keeping "p045", "p075", "p022", "p012" in test and rest in train set...
test_pid = ["p045", "p075", "p022", "p012"]
X_train, X_test, y_train, y_test = [], [], [], []
for i in data:
    if i[2] in test_pid:
        X_test.append(i[0])
        y_test.append(i[1])
    else: 
        X_train.append(i[0])
        y_train.append(i[1])


INPUT_SIZE = [224, 224] # Image size...
BATCH_SIZE = 64 # Number of data points in each batch...

# Reshaping the test and train to 224x224 with 3 channels (rgb)
X_train = np.array(X_train).reshape(-1,224,224,3)
X_test = np.array(X_test).reshape(-1,224,224,3)

# converting the target varial to categorical...
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# ImageDataGenerator is a handy api from keras to read images and perform image augmentation...
train_gen = ImageDataGenerator(
    height_shift_range=0.3, # Shifting image vertically...
    width_shift_range = 0.3, # Shifting image horizontally...
    rotation_range=30, # Rorating image by 30 degree...
    shear_range=0.3, # stretching image from one end...
    zoom_range=0.3 # Zooming on the image...
)

# Passing data from np.array to create train generator...
train_generator = train_gen.flow(
    X_train,
    y_train_cat,
    batch_size=BATCH_SIZE
)

# ModelCheckpoint callback is used to save only the best model out of all the epochs...
checkpoint2 = ModelCheckpoint(filepath="./models/model2/with_conv2.h5", verbose=2, save_best_only=True)

# EarlyStopping callback is used to stop the training when accuracy doesn't improve for 5 epochs...
early_stop = EarlyStopping(monitor="accuracy", min_delta=0, patience=5)

callbacks2 = [early_stop, checkpoint2]

# input of size 224x224x3... hight x width = 224 x 224, number of channels = 3...
input = Input(name = 'img_input', shape=(224, 224, 3))

# Loading weights of the vgg16 pre-trained model without including top layers... imagenet is a dataset on which vgg16 was trained...
vgg16_model = VGG16(input_tensor = input, include_top=False, weights="imagenet")

# Making vgg16 layers as non-trainable...
for layer in vgg16_model.layers:
  layer.trainable = False

input_layer = vgg16_model(input) # defining input layer...


# Functional API of keras is used for this network...

# Convolution layer with 128 filters of size 3x3 and stride = 1x1, activation function used is ReLU and "valid" padding to keep the input and output size same...
# he normal kernel initializer is used as it performs well with non-linear activation functions like ReLU... 
conv = Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu', 
          kernel_initializer=tf.keras.initializers.he_normal())(input_layer)
pool = GlobalAveragePooling2D()(conv) # the pool size is set to the input size and it outputs the average of the pool...
d1 = Dense(units=1024,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(pool)
drop1 = Dropout(0.5)(d1) # dropout was used to deactivate some of the nodes to avoid overfitting...
d2 = Dense(units=512,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(drop1)
bn = BatchNormalization()(d2) # batch normalization was used normalize weights batch wise and in turn reduce the chances of overfitting...
drop2 = Dropout(0.2)(bn)
d3 = Dense(units=512,activation='relu', kernel_initializer=tf.keras.initializers.he_normal())(drop2)
Out2 = Dense(units=10,activation='softmax', kernel_initializer=tf.keras.initializers.he_normal())(d3)
model2=tf.keras.Model(input, Out2) 


sgd = SGD(learning_rate=0.001) # SGD optimizer was used to have gradual movement is reaching the optimal weights...
model2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting/training model...
history2 = model2.fit(
    train_generator, 
    steps_per_epoch = len(X_train) / BATCH_SIZE, # number of steps in each epoch...
    callbacks=callbacks2,
    epochs = 40, 
    validation_data = (X_test, y_test_cat))


# plotting the train, test accuracy and loss...
plotter(history2)