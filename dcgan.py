import itertools

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nets.dcgan import generator


#---------------------------------------------#
#   注意修改model_path、channel和input_shape
#   需要和训练时的设置一样。
#---------------------------------------------#
class DCGAN(object):
    _defaults = {
        #-----------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #-----------------------------------------------#
        "model_path"        : 'model_data/Generator_Flower.h5',
        #-----------------------------------------------#
        #   卷积通道数的设置
        #-----------------------------------------------#
        "channel"           : 64,
        #-----------------------------------------------#
        #   输入图像大小的设置
        #-----------------------------------------------#
        "input_shape"       : [64, 64],
    }

    #---------------------------------------------------#
    #   初始化DCGAN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    #---------------------------------------------------#
    #   创建生成模型
    #---------------------------------------------------#
    def generate(self):
        self.net = generator(self.channel, self.input_shape)
        self.net.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

    #---------------------------------------------------#
    #   生成5x5的图片
    #---------------------------------------------------#
    def generate_5x5_image(self, save_path):
        randn_in = np.random.randn(5*5, 100)
        test_images = self.net.predict(randn_in)

        #-------------------------------#
        #   利用plt进行绘制
        #-------------------------------#
        size_figure_grid = 5
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(5*5):
            i = k // 5
            j = k % 5
            ax[i, j].cla()
            ax[i, j].imshow((test_images[k] * 0.5 + 0.5))

        label = 'predict_5x5_results'
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig(save_path)

    #---------------------------------------------------#
    #   生成1x1的图片
    #---------------------------------------------------#
    def generate_1x1_image(self, save_path):
        randn_in = np.random.randn(1, 100)
        test_images = self.net.predict(randn_in)
        
        test_images = (test_images[0] * 0.5 + 0.5) * 255
        Image.fromarray(np.uint8(test_images)).save(save_path)



