import math

from keras import initializers, layers
from keras.models import Model


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def generator(d = 128, input_shape = [64, 64]):
    conv_options = {
        'kernel_initializer': initializers.normal(mean=0.0, stddev=0.02),
    }
    batchnor_options = {
        'gamma_initializer' : initializers.normal(mean=0.1, stddev=0.02),
        'beta_initializer'  : initializers.constant(0),
        'momentum'          : 0.9
    }

    inputs = layers.Input([100,])

    #----------------------------------------------#
    #   当生成的图片是64, 64, 3的时候
    #----------------------------------------------#
    s_h, s_w = input_shape[0], input_shape[1]
    # 32, 32
    s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
    # 16, 16
    s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
    # 8, 8
    s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
    # 4, 4
    s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

    #----------------------------------------------#
    #   100, -> 8192, 
    #----------------------------------------------#
    x = layers.Dense(s_h16*s_w16*d*8, **conv_options)(inputs)

    #----------------------------------------------#
    #   8192, -> 4, 4, 512 
    #----------------------------------------------#
    x = layers.Reshape([s_h16,s_w16,d*8])(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.Activation("relu")(x)
 
    #----------------------------------------------#
    #   4, 4, 512 -> 8, 8, 256 
    #----------------------------------------------#
    x = layers.Conv2DTranspose(filters=d*4, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.Activation("relu")(x)

    #----------------------------------------------#
    #   8, 8, 256 -> 16, 16, 128 
    #----------------------------------------------#
    x = layers.Conv2DTranspose(filters=d*2, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.Activation("relu")(x)

    #----------------------------------------------#
    #   16, 16, 128 -> 32, 32, 64 
    #----------------------------------------------#
    x = layers.Conv2DTranspose(filters=d, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.Activation("relu")(x)

    #----------------------------------------------#
    #   32, 32, 64 -> 64, 64, 3 
    #----------------------------------------------#
    x = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.Activation("tanh")(x)
    
    model = Model(inputs, x)
    return model

def discriminator(d = 128, input_shape = [64, 64]):
    conv_options = {
        'kernel_initializer': initializers.normal(mean=0., stddev=0.02),
    }
    batchnor_options = {
        'gamma_initializer' : initializers.normal(mean=0.1, stddev=0.02),
        'beta_initializer'  : initializers.constant(0),
        'momentum'          : 0.9
    }

    #----------------------------------------------#
    #   64, 64, 3 -> 32, 32, 64
    #----------------------------------------------#
    inputs = layers.Input([input_shape[0], input_shape[1], 3])
    x = layers.Conv2D(filters=d, kernel_size=4, strides=2, padding="same", **conv_options)(inputs)
    x = layers.LeakyReLU(0.2)(x)

    #----------------------------------------------#
    #   32, 32, 64 -> 16, 16, 128
    #----------------------------------------------#
    x = layers.Conv2D(filters=2*d, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.LeakyReLU(0.2)(x)

    #----------------------------------------------#
    #   16, 16, 128 -> 8, 8, 256
    #----------------------------------------------#
    x = layers.Conv2D(filters=4*d, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.LeakyReLU(0.2)(x)

    #----------------------------------------------#
    #   8, 8, 256 -> 4, 4, 512
    #----------------------------------------------#
    x = layers.Conv2D(filters=8*d, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)

    #----------------------------------------------#
    #   4*4*512, -> 1, 
    #----------------------------------------------#
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, **conv_options)(x)
    x = layers.Activation("sigmoid")(x)
    
    model = Model(inputs, x)
    return model

if __name__ == "__main__":
    model = discriminator(128)
    model.summary()
