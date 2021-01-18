import os.path
import random
import sys

# import cv2
import h5py
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.mnist_font_dataset import MNISTFontDataset


class MNISTFontSplitDataset(BaseDataset):

    def __init__(self, opt):

        self.split_db = []
        for i in range(10):
            self.split_db.append(MNISTFontDataset(opt, i))
        self.compute_weights()

    def __getitem__(self, index):

        result = {'weights': self.weights}
        for k, v in enumerate(self.split_db):
            database = v
            if index >= len(database):
                index = index % len(database)

            index_value = database[index]
            result['A_' + str(k)] = index_value['A']
            result['B_' + str(k)] = index_value['B']
            result['A_paths_' + str(k)] = index_value['A_paths']
            result['B_paths_' + str(k)] = index_value['B_paths']

        return result

    def compute_weights(self):
        self.weights = []
        num_of_labels = np.zeros((10, 10))
        for i in range(10):
            database = self.split_db[i]
            for k in range(10):
                num_of_labels[i][k] = np.sum(np.array(database.label) == k)
        self.weights = num_of_labels / np.sum(num_of_labels, axis=0)

    def __len__(self):
        """Return the total number of images in the dataset."""
        length = 0
        for i in self.split_db:
            if len(i) > length:
                length = len(i)

        return length

