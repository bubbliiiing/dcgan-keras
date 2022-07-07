import math
from random import shuffle

import keras
import numpy as np
from PIL import Image

from  .utils import cvtColor, preprocess_input


class DCganDataset(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, batch_size):
        super(DCganDataset, self).__init__()

        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        
        self.input_shape        = input_shape
        self.batch_size         = batch_size

    def __len__(self):
        return math.ceil(self.length / float(self.batch_size))
    
    def on_epoch_end(self):
        shuffle(self.annotation_lines)

    def __getitem__(self, index):
        images = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i       = i % self.length
            image   = Image.open(self.annotation_lines[i].split()[0])
            image   = cvtColor(image).resize([self.input_shape[1], self.input_shape[0]], Image.BICUBIC)

            image   = np.array(image, dtype=np.float32)
            images.append(preprocess_input(image))
        return np.array(images, dtype=np.float32)