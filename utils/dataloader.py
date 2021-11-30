import math
from random import shuffle

import keras
import numpy as np
from PIL import Image

from .utils import cvtColor


class DCganDataset(keras.utils.Sequence):
    def __init__(self, train_lines, input_shape, batch_size):
        super(DCganDataset, self).__init__()

        self.train_lines    = train_lines
        self.train_batches  = len(train_lines)
        self.input_shape    = input_shape
        self.batch_size     = batch_size

    def __len__(self):
        return math.ceil(self.train_batches / float(self.batch_size))

    def preprocess_input(self, image, mean, std):
        image = (image/255 - mean)/std
        return image

    def __getitem__(self, index):
        if index == 0:
            self.on_epoch_begin()
        images = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i       = i % self.train_batches
            image   = Image.open(self.train_lines[i].split()[0])
            image   = cvtColor(image).resize([self.input_shape[1], self.input_shape[0]], Image.BICUBIC)

            image   = np.array(image, dtype=np.float32)
            images.append(self.preprocess_input(image, 0.5, 0.5))
        return np.array(images)

    def on_epoch_begin(self):
        shuffle(self.train_lines)
