import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math
import pandas as pd
import random
import json
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

def preprocess(img):
    '''
    preprocess: Performs preprocessing for
    images to be fed into neural network
    Input: Numpy array of an image
    Output: 64x64x3 numpy array
    '''

    # crop top and bottom from image
    img = img[40:120, :]
    # resize image to 64 by 64 pixels
    img = cv2.resize(img, (64, 64))
    # change to YUV colour space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    return img

if __name__ == "__main__":
    # read in driving log data to pandas dataframe
    df_filtered = pd.read_csv('data/driving_log.csv')

    # X and y lists to hold images
    X = []
    y = []
    # path to the image file directory
    path = "data/"
    # steering offset to be added to left and right images
    # for data augmentation
    STEERING_OFFSET = 0.25

    # mean and standard deviation of steering angles
    mn = np.mean(df_filtered['steering'])
    std = np.std(df_filtered['steering'])

    # import all images from folders
    for i in range(len(df_filtered)):

        # c, l, and r are the center, left, and right images
        # respectively
        c = mpimg.imread(path + df_filtered.iloc[i,:]['center'])
        l = mpimg.imread(path + df_filtered.iloc[i,:]['left'][1:])
        r = mpimg.imread(path + df_filtered.iloc[i,:]['right'][1:])
        
        # if steering angle is less than 0.1, give probability equal to its
        # normal distribution value of being incorporated
        num = np.random.normal(mn,std)
        if not ((abs(df_filtered.iloc[i,:]['steering']) < 0.1) and df_filtered.iloc[i,:]['steering'] < num): 

            # append center image and its vertically reflection to X, its label and -1 * label to Y
            X.append(c)
            X.append(cv2.flip(c,1))
            y.append(df_filtered.iloc[i,:]['steering'])
            y.append(-1.*df_filtered.iloc[i,:]['steering'])

            # append left and right images and their vertical reflections to X
            X.append(l)
            X.append(cv2.flip(l,1))
            X.append(r)
            X.append(cv2.flip(r,1))

            # append labels as the center steering angle offset by the steering offset constant
            # append the vertical reflections of these labels as -1 * label
            y.append(df_filtered.iloc[i,:]['steering'] + STEERING_OFFSET)
            y.append(-1.*(df_filtered.iloc[i,:]['steering'] + STEERING_OFFSET))
            y.append(df_filtered.iloc[i,:]['steering'] - STEERING_OFFSET)
            y.append(-1.*(df_filtered.iloc[i,:]['steering'] - STEERING_OFFSET))

    # perform preprocessing on each image
    for i, el in enumerate(X):
        X[i] = preprocess(el)
    # convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # create training and validation sets with validation set size of 20%.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2) 

    # model created based off of NVIDIA model from 
    # http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    model = Sequential()

    # normalization layer, takes whatever input shape is present in first element of X
    model.add(BatchNormalization(input_shape=X[0].shape))

    # three 5x5 convolution layers with 2x2 strides, ELU Activation
    # and 0.2 dropout after each convolution layer
    model.add(Conv2D(24, 5, 5,subsample=(2,2), dim_ordering='tf'))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(36, 5, 5,subsample=(2,2) ,dim_ordering='tf'))
    model.add(Dropout(0.2))
    model.add(Activation('elu'))
    model.add(Conv2D(48, 5, 5,subsample=(2,2), dim_ordering='tf'))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))

    # two 3x3 convolution layers with no stride, ELU activation,
    # and 0.2 dropout after each convolution layer
    model.add(Conv2D(64, 3, 3, dim_ordering='tf'))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, 3, 3, dim_ordering='tf'))
    model.add(Activation('elu'))

    # flatten layer followed by 0.2 dropout
    model.add(Flatten())
    model.add(Dropout(0.2))

    # three fully connected layers with 100, 50, and 10 nodes respectively,
    # ELU activation, and 0.2 dropout after each layer other than the last two.
    model.add(Dense(100))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    # tanh activation followed by a single node to be used as the steering input
    model.add(Activation('tanh'))
    model.add(Dense(1))

    # adam optimizer to minimize the mean squared error with accuracy metric
    model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['accuracy'])
    model.summary()

    # use fit generator with preprocessed training and validation data to train model
    # and save the model and training parameters
    imageGen = ImageDataGenerator()
    # stop training if no longer improving, save best model only
    callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, verbose=0, save_weights_only=True),
    ]
    history = model.fit_generator(imageGen.flow(X_train, y_train, batch_size=256), samples_per_epoch=len(X_train),
                                nb_epoch=1000, validation_data=imageGen.flow(X_val, y_val, batch_size=256),
                                nb_val_samples=len(X_val),callbacks=callbacks)

    #model.save_weights("model.h5", True)
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
