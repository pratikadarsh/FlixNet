''' Generator class for feeding the data to the keras model.'''


import os
import numpy as np
import keras
import cv2 as cv

class DataGenerator(keras.utils.Sequence):
    """ Generates data for the model training and validation."""

    def __init__(self, list_IDs, img_folder, labels, batch_size=32, dim=(224,224,3), n_classes=(7, 4, 10), shuffle=True):
        'Initialization'
        self.img_folder = os.path.join(os.path.dirname(__file__), "..", img_folder)
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.indexes = np.arange(len(self.list_IDs))
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end() 

    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data. """

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, neck, sleeve, pattern = self.__data_generation(list_IDs_temp)
        return X, [neck, sleeve, pattern]

    def on_epoch_end(self):
        """ Shuffles the indexes after each epoch. """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """ Generates data containing batch_size samples. """
        
        X = []
        neck = []
        sleeve_length = []
        pattern = []
        
        # Generate data
        for ID in list_IDs_temp:
            # Store sample
            img = cv.imread(os.path.join(self.img_folder, ID))
            img = cv.resize(img, (self.dim[0], self.dim[1]))
            X.append(img)

            # Store class
            neck.append(keras.utils.to_categorical(self.labels[ID][0], num_classes=self.n_classes[0]))
            sleeve_length.append(keras.utils.to_categorical(self.labels[ID][1], num_classes=self.n_classes[1]))
            pattern.append(keras.utils.to_categorical(self.labels[ID][2], num_classes=self.n_classes[2]))

        return np.asarray(X), np.asarray(neck), np.asarray(sleeve_length), np.asarray(pattern)
