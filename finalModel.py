"""Imports"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import multiprocessing

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed as td
from tensorflow.keras.layers import LSTM, Conv2D, MaxPool2D, Flatten

from sequencer import DataSequencer, imgToGreyScale

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dirBase = "/data/sohailf7/Test"

np.random.seed(42)
time_step = 5                     # The LSTM uses 5 sequential images to generate an output
epochs = 50
batch_size = 5

# Global dimensional values for rescaled image dimensions
INPUT_WIDTH = 0
INPUT_HEIGHT = 0


def loadData(pathToImages, timestep):
    """
    Returns array containing all paths for a single folder
    """
    fileNames = os.listdir(os.path.join(pathToImages, "img")) # List containing the names of every file in the directory
    fileNames.sort()

    dataArray = np.ndarray(shape=(len(fileNames)-(timestep-1), timestep), dtype="object")

    for i in range(len(fileNames)-(timestep-1)): # 0 to 596
        for time_index in range(timestep): # 0 to 4
            singleImgPath = os.path.join(pathToImages, "img", fileNames[i+time_index]) # fileNames[i+time_index] for example: [0...4], [1...5] ... [i, i+(time_step-1)]
            if os.path.isfile(singleImgPath): # If singleImgPath exists, load it into dataArray
                dataArray[i, time_index] = singleImgPath

    folderLabels = np.loadtxt(os.path.join(pathToImages, "groundtruth_rect.txt"), delimiter=",")
    fixedFolderLabels = folderLabels[4:, 1:] # Delete the first column which indicates frame number

    return dataArray, fixedFolderLabels

def iterateDataSet(path, singleFolder):

  singlePath = os.path.join(path, singleFolder) # Path to a single folder

  folderImgPaths, folderLabels = loadData(singlePath, time_step)

  return folderImgPaths, folderLabels


allFolders = fileNames = os.listdir(dirBase)
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(
    delayed(iterateDataSet)(dirBase, singleFolder) for singleFolder in allFolders)

for i in range(len(results)):
    if i == 0:
        trainData = results[i][0]
        trainlabels = results[i][1]
    else:
        trainData = np.vstack((trainData, results[i][0]))
        trainlabels= np.vstack((trainlabels, results[i][1]))


trainPaths, testPaths, trainLabels, testLabels = train_test_split(trainData, trainlabels, random_state=42, shuffle=False)

# Set INPUT_HEIGHT and INPUT_WIDTH variables by running a single image through imgToGreyScale
dimImg = cv2.imread(trainData[0, 0])
resizedImg = imgToGreyScale(dimImg)
INPUT_HEIGHT = resizedImg.shape[0]
INPUT_WIDTH = resizedImg.shape[1]

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
preds = layers.Dense(5, activation='relu')(lstmOut)

tdmodel = tf.keras.models.Model(inputs=input, outputs=preds)


opt = tf.keras.optimizers.Adam(learning_rate=0.001)
tdmodel.compile(optimizer=opt, loss='MSE', metrics=['accuracy'])

training_generator = DataSequencer(trainPaths, trainLabels, batch_size, time_step, (INPUT_HEIGHT, INPUT_WIDTH, 1))
testing_generator = DataSequencer(testPaths, testLabels, batch_size, time_step, (INPUT_HEIGHT, INPUT_WIDTH, 1))

history = tdmodel.fit(x=training_generator,
                      epochs=epochs,
                      steps_per_epoch=len(training_generator),
                      shuffle=True,
                      validation_steps=len(testing_generator),
                      validation_data=testing_generator)

