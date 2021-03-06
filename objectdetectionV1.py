"""
This file is the initial version of the CNN/LSTM model for object detection.
It trains on the TinyTLPV2 dataset. However, it only trains on 2 videos and tests on 1 video.
Because of this, it has poor performance (~40% validation accuracy).
"""

"""Imports"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import math as m

print(tf.__version__)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed as td
from tensorflow.keras.layers import LSTM, Conv2D, MaxPool2D, Flatten

from tensorflow.keras.callbacks import CSVLogger

from time import gmtime, strftime


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


"""Config Variables"""
dirBase = '/data/angelinic0/clinicData'

np.random.seed(42)
time_step = 5                     # The LSTM uses 5 sequential images to generate an output
epochs = 25
batch_size = 10

# Global dimensional values for rescaled image dimensions
INPUT_WIDTH = 0
INPUT_HEIGHT = 0

"""Image Preprocessing Methods"""

def pathToGreyScale(path):
  """
  Rescales from 1280x720 to 50x & converts ONE image from rgb to grayscale.
  :String path: The path to the image
  :return: The normalized greyscale image
  """
  W = 120 # Percentage to resize the image by

  origImg = cv2.imread(path)

  height, width, depth = origImg.shape # 720, 1280, 3
  imgScale = W/width
  newX, newY = origImg.shape[1]*imgScale, origImg.shape[0]*imgScale # Set the new X and Y according to the scale
  newimg = cv2.resize(origImg,(int(newX), int(newY))) # Resize the image with the new X and Y values

  # Greyscale the image by "averaging" the 3 matrix values (flattening the depth to 1 matrix instead of 3)
  greyscale = 0.2989*newimg[:,:,0] + 0.5870*newimg[:,:,1] + 0.1140 *newimg[:,:,2]

  greyscale = greyscale/255 # Scale down the numpy array to reduce computational intensity
  greyscale = greyscale[..., np.newaxis] # Add a new dimension

  # Expect greyscale.shape to have 3 dimensions (its height, its width, 1)
  assert(greyscale.shape == (greyscale.shape[0], greyscale.shape[1], 1))
  return greyscale

def loadData(pathToImages, timestep):
  """
  Loads data into dataArray from the given pathToImages
  :String pathToImages: The path to the images you want to load
  :int timestep: Given timestep, different than config variable time_step
  """
  fileNames = os.listdir(pathToImages) # List containing the names of every file in the directory
  fileNames.sort()
  # Run one image through pathToGreyScale to get its dimensions
  dimImg = pathToGreyScale(dirBase + '/TrainingData/CarChase1/00001.jpg')
  dataArray = np.ndarray(shape=(len(fileNames)-(timestep), timestep, dimImg.shape[0], dimImg.shape[1], 1))
  for i in range(len(fileNames)-timestep): # 0, 1, 2 ... , (fileNames-(timestep-1))+1
    for time_index in range(timestep): # 0, 1, 2 ... , timestep-1
      singleImgPath = os.path.join(pathToImages, fileNames[i+time_index]) # fileNames[i+time_index] for example: [0...4], [1...5] ... [i, i+(time_step-1)]
      if os.path.isfile(singleImgPath): # If singleImgPath exists, load it into dataArray
        dataArray[i, time_index, :, :] = pathToGreyScale(singleImgPath)
  return dataArray

"""Load Train/Test Datasets"""

# Load Training Data Set

# Load CarChase1
training_set_path = dirBase + '/TrainingData/CarChase1/'
train_data1 = loadData(training_set_path, time_step)

# Load CarChase2
training_set_path = dirBase + '/TrainingData/CarChase2/' 
train_data2 = loadData(training_set_path, time_step)

# Combine train_data1 and train_data2
train_data = np.vstack((train_data1, train_data2))
print(train_data.shape)

# Set INPUT_HEIGHT and INPUT_WIDTH variables according to training set
INPUT_HEIGHT = train_data.shape[2]
INPUT_WIDTH = train_data.shape[3]

# Load Test Data
testing_set_path = dirBase + '/TestingData/CarChase3/'
test_data = loadData(testing_set_path, time_step)
print(test_data.shape)

"""Load Labels"""

# Load Training Labels
train_labels = pd.read_csv(dirBase + '/TrainingLabels/Combined_Labels.csv')
train_labels = train_labels.to_numpy()
train_labels = np.vstack((train_labels[4:599,:], train_labels[604:,:]))
print('Train Labels: ' + str(train_labels.shape))

# Load Testing Labels
test_labels = pd.read_csv(dirBase + '/TestingLabels/CarChase3.csv')
test_labels = test_labels.to_numpy()
test_labels = test_labels[4:,:]
print(test_labels.shape)

"""Model"""

model = Sequential()
model.add(Conv2D(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())

input = layers.Input(batch_shape=(batch_size, time_step, INPUT_HEIGHT, INPUT_WIDTH, 1))
tdOut = td(model)(input)
lstmOut = layers.LSTM(50, activation='tanh')(tdOut)
preds = layers.Dense(5, activation='softmax')(lstmOut)

tdmodel = tf.keras.models.Model(inputs=input, outputs=preds)

"""Compile Model"""

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
tdmodel.compile(optimizer=opt, loss='MSE', metrics=['accuracy'])

"""Model.Fit"""

history = tdmodel.fit(x = train_data,
                      y = train_labels,
                      batch_size = batch_size,
                      epochs = epochs, 
                      steps_per_epoch = int(len(train_data)/batch_size)-1,
                      shuffle = True,
                      validation_steps = int(len(test_data)/batch_size)-1,
                      validation_data = (test_data, test_labels))

