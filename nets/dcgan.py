import math

import keras
from keras import initializers, layers
from keras.models import Model, Sequential


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))
    
def generator(d=128, image_shape=[64,64,3]):
    conv_options = {
        'kernel_initializer': initializers.normal(mean=0.0, stddev=0.02),
    }
    batchnor_options = {
        'gamma_initializer' : initializers.normal(mean=0.1, stddev=0.02),
        'beta_initializer'  : initializers.constant(0),
        'momentum'          : 0.9
    }

    inputs = layers.Input([100,])

    s_h, s_w = image_shape[0], image_shape[1]
    s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
    s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
    s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
    s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

    x = layers.Dense(s_h16*s_w16*d*8, **conv_options)(inputs)
    x = layers.Reshape([s_h16,s_w16,d*8])(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.Activation("relu")(x)
 
    x = layers.Conv2DTranspose(filters=d*4, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2DTranspose(filters=d*2, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2DTranspose(filters=d, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.Activation("tanh")(x)
    
    model = Model(inputs, x)
    return model

def discriminator(d=128, image_shape=[64,64,3]):
    conv_options = {
        'kernel_initializer': initializers.normal(mean=0., stddev=0.02),
    }
    batchnor_options = {
        'gamma_initializer' : initializers.normal(mean=0.1, stddev=0.02),
        'beta_initializer'  : initializers.constant(0),
        'momentum'          : 0.9
    }

    inputs = layers.Input(image_shape)
    x = layers.Conv2D(filters=d, kernel_size=4, strides=2, padding="same", **conv_options)(inputs)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(filters=2*d, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(filters=4*d, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(filters=8*d, kernel_size=4, strides=2, padding="same", **conv_options)(x)
    x = layers.BatchNormalization(**batchnor_options)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, **conv_options)(x)
    x = layers.Activation("sigmoid")(x)
    
    model = Model(inputs, x)
    return model

def combine_model(generator, discriminator, optimizer, latent_dim=100):
    # conbine是生成模型和判别模型的结合
    # 判别模型的trainable为False
    # 用于训练生成模型
    z = layers.Input(shape=(1, 1, latent_dim))
    img = generator(z)

    discriminator.trainable = False

    valid = discriminator(img)

    combine_model = Model(z, valid)
    return combine_model

if __name__ == "__main__":
    # model = generator(128)
    # model.summary()
    model = discriminator(128)
    model.summary()
