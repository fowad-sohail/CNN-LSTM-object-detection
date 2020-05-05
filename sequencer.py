import tensorflow as tf
import cv2
import numpy as np


class DataSequencer(tf.keras.utils.Sequence):
    # Initialization function
    def __init__(self, X_data, y_data, batch_size, time_steps, input_shape, shuffle=True):
        self.X_data = X_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.input_shape = input_shape

        self.shuffle = shuffle
        # Array of indexes with shuffle
        if self.shuffle:
            self.indexes = np.arange(len(self.X_data))
            np.random.shuffle(self.indexes)
        else:
            self.indexes = np.arange(len(self.X_data))
        self.on_epoch_end()

    # Get length of batch training
    def __len__(self):
        return int(np.floor(len(self.X_data) / self.batch_size))

    # Returning a batch item
    def __getitem__(self, index):
        # Get range of indexes based on the training index and the batch size
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size-1]


        # Create empty arrays used a ndarray for some reason with the shape
        X = np.ndarray((self.batch_size, self.time_steps, *self.input_shape), dtype=float)
        y = np.empty((self.batch_size, 5), dtype=float)

        # Loop for finding where the video frames switch
        for i, j in enumerate(indexes):
            # For time frames
            for time_idx in range(self.time_steps):
                # Load image
                frame = cv2.imread(self.X_data[j, time_idx])
                normed_img = imgToGreyScale(frame)

                X[i, time_idx, :] = normed_img
            y[i] = self.y_data[j]
        return X, y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

def imgToGreyScale(origImg):
    """
    Rescales from 1280x720 to 50x & converts ONE image from rgb to grayscale.
    :String path: The path to the image
    :return: The normalized greyscale image
    """
    W = 120 # Percentage to resize the image by

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