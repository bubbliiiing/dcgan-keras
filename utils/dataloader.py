import math
from random import shuffle

import cv2
import keras
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image


class DCganDataset(keras.utils.Sequence):
    def __init__(self, train_lines, image_size, batch_size):
        super(DCganDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.batch_size = batch_size
        self.global_index = 0

    def __len__(self):
        return math.ceil(self.train_batches / float(self.batch_size))

    def pre_process(self, image, mean, std):
        image = (image/255 - mean)/std
        return image

    def __getitem__(self, index):
        if self.global_index == 0:
            shuffle(self.train_lines)

        images = []
        lines = self.train_lines
        n = self.train_batches
        for _ in range(self.batch_size):
            #----------------------------------------------#
            #   读取图像并进行归一化，归一化到-1-1之间
            #----------------------------------------------#
            img = Image.open(lines[self.global_index].split()[0]).resize(self.image_size[0:2], Image.BICUBIC)
            img = np.array(img, dtype=np.float32)

            img = self.pre_process(img, [0.5,0.5,0.5], [0.5,0.5,0.5])
            images.append(img)

            self.global_index = (self.global_index + 1) % n
        return np.array(images)
