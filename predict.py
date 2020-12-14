#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from dcgan import DCGAN
from PIL import Image

dcgan = DCGAN()
while True:
    img = input('Just Click Enter~')
    dcgan.generate_5x5_image()
    dcgan.generate_1x1_image()
