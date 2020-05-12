"""Imports"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import math as m

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed as td
from tensorflow.keras.layers import LSTM, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.callbacks import CSVLogger

from time import gmtime, strftime

from sequencer import DataSequencer, imgToGreyScale

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

"""Config Variables"""

dirBase = "/data/sohailf7/"

np.random.seed(42)
time_step = 5                     # The LSTM uses 5 sequential images to generate an output
epochs = 50
batch_size = 32

# Global dimensional values for rescaled image dimensions
INPUT_WIDTH = 0
INPUT_HEIGHT = 0

"""Image Preprocessing Methods"""

def loadData(pathToFolder, timestep):
  """
  Traverse a single folder and return a np array containing directories to each image and a np array containing its labels
  :string pathToFolder: The path for 
  """
  fileNames = os.listdir(os.path.join(pathToFolder, "img")) # List containing the names of every file in the directory
  fileNames.sort()
  
  dataArray = np.ndarray(shape=(len(fileNames)-(timestep-1), timestep), dtype="object")

  for i in range(len(fileNames)-(timestep-1)): # 0 to 596
    for time_index in range(timestep): # 0 to 4
      singleImgPath = os.path.join(pathToFolder, "img", fileNames[i+time_index]) # fileNames[i+time_index] for example: [0...4], [1...5] ... [i, i+(time_step-1)]
      if os.path.isfile(singleImgPath): # If singleImgPath exists, load it into dataArray
        dataArray[i, time_index] = singleImgPath
        
  folderLabels = np.loadtxt(os.path.join(pathToFolder, "groundtruth_rect.txt"), delimiter=",")
  fixedFolderLabels = folderLabels[4:, 1:] # Delete the first column which indicates frame number

  return dataArray, fixedFolderLabels

def iterateDataSet(path):
  """
  Iterate through all of the folders
  :return nparray: containing directory names of each picture
  :return nparray: containing labels for each video
  """
  allFolders = os.listdir(path) # All the folder names

  for i, singleFolder in enumerate(allFolders):
    singlePath = os.path.join(path, singleFolder) # Path to a single folder

    folderImgPaths, folderLabels = loadData(singlePath, time_step)

    if i == 0:
      allLabels = folderLabels
      allPaths = folderImgPaths
    else:
      allLabels = np.vstack((allLabels, folderLabels))
      allPaths = np.vstack((allPaths, folderImgPaths))
  
  return allPaths, allLabels


"""Load Train/Test Datasets
    Following a 70/30 train/test split"""

# Load Training Data Set - Train on Alladin - KinBall1
trainDir = os.path.join(dirBase, "Train")
trainPathsAndLabels = iterateDataSet(trainDir)
trainPaths = trainPathsAndLabels[0]
trainLabels = trainPathsAndLabels[1]



# Load Test Data - Test on KinBall2 - ZebraFish
testDir = os.path.join(dirBase, "Test")
testPathsAndLabels = iterateDataSet(testDir)
testPaths = testPathsAndLabels[0]
testLabels = testPathsAndLabels[1]
np.save('testPaths', testPaths)


# Set INPUT_HEIGHT and INPUT_WIDTH variables by running a single image through imgToGreyScale
dimImg = cv2.imread(os.path.join(dirBase, "Test/KinBall2/img/00600.jpg"))
resizedImg = imgToGreyScale(dimImg)
INPUT_HEIGHT = resizedImg.shape[0]
INPUT_WIDTH = resizedImg.shape[1]

print("INPUT HEIGHT: " + str(INPUT_HEIGHT))
print("INPUT WIDTH: " + str(INPUT_WIDTH))


print("TRAIN PATHS SHAPE" + str(trainPaths.shape))
print("TRAIN LABELS SHAPE" + str(trainLabels.shape))
print()
print("TEST PATHS SHAPE" + str(testPaths.shape))
print("TEST LABELS SHAPE" + str(testLabels.shape))
print()


"""Create Model"""
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

opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
tdmodel.compile(optimizer=opt, loss=tf.keras.losses.CategoricalHinge(), metrics=['accuracy'])

"""Model.Fit"""
training_generator = DataSequencer(trainPaths, trainLabels, batch_size, time_step, (INPUT_HEIGHT, INPUT_WIDTH, 1))
testing_generator = DataSequencer(testPaths, testLabels, batch_size, time_step, (INPUT_HEIGHT, INPUT_WIDTH, 1))

history = tdmodel.fit(x = training_generator,
                      epochs = epochs,
                      shuffle = False,
                      validation_data = testing_generator)

