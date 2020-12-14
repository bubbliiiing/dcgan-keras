import numpy as np
import keras.backend as K
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm
from random import shuffle
import tensorflow as tf

from nets.dcgan import combine_model, discriminator, generator
from utils.dataloader import DCganDataset
from utils.utils import show_result

def fit_one_epoch(G_model, D_model, Combine_model, epoch, epoch_size, gen, Epoch, batch_size, save_interval):
    G_total_loss = 0
    D_total_loss = 0

    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, images in enumerate(gen):
            if iteration >= epoch_size:
                break
            y_real                  = np.ones([batch_size,1])
            y_fake                  = np.zeros([batch_size,1])

            d_loss_real             = D_model.train_on_batch(images, y_real)

            noise                   = np.random.randn(batch_size, 100)
            G_result                = G_model.predict(noise)
            d_loss_fake             = D_model.train_on_batch(G_result, y_fake)
            d_loss                  = 0.5 * np.add(d_loss_real, d_loss_fake)

            y_real                  = np.ones([batch_size,1])
            noise                   = np.random.randn(batch_size, 100)
            g_loss                  = Combine_model.train_on_batch(noise, y_real)
            G_total_loss            += g_loss
            D_total_loss            += d_loss

            pbar.set_postfix(**{'G_loss'    : G_total_loss / (iteration + 1), 
                                'D_loss'    : D_total_loss / (iteration + 1),
                                'lr'        : K.get_value(D_model.optimizer.lr)},)
            pbar.update(1)

            if iteration % save_interval == 0:
                show_result(epoch+1,G_model)

    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('G Loss: %.4f || D Loss: %.4f ' % (G_total_loss/(epoch_size+1),D_total_loss/(epoch_size+1)))

    if (epoch+1) % 10==0:
        G_model.save_weights('logs/G_Epoch%d-GLoss%.4f-DLoss%.4f.h5'%((epoch+1),G_total_loss/(epoch_size+1), D_total_loss/(epoch_size+1)))
        D_model.save_weights('logs/D_Epoch%d-GLoss%.4f-DLoss%.4f.h5'%((epoch+1),G_total_loss/(epoch_size+1), D_total_loss/(epoch_size+1)))
        print('Saving state, iter:', str(epoch+1))


if __name__ == "__main__":
    image_shape = [64, 64, 3]

    # 数据集存放路径
    annotation_path = "train_lines.txt"

    # 生成网络和评价网络
    G_model = generator(64, image_shape)
    D_model = discriminator(64, image_shape)

    # G_model_path = "model_data/Generator_Flower.h5"
    # D_model_path = "model_data/Discriminator_Flower.h5"
    # G_model.load_weights(G_model_path, by_name=True, skip_mismatch=True)
    # D_model.load_weights(D_model_path, by_name=True, skip_mismatch=True)
    
    with open(annotation_path) as f:
        lines = f.readlines()
    num_train = len(lines)

    #------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Epoch总训练世代
    #------------------------------------------------------#
    if True:
        # 训练参数设置
        lr = 0.002
        batch_size = 64
        Init_epoch = 0
        Epoch = 500
        save_interval = 50

        # Adam optimizer
        D_model.compile(loss="binary_crossentropy", optimizer=Adam(lr, 0.5, 0.999))

        noise = layers.Input(shape=(100,))
        img = G_model(noise)
        D_model.trainable = False
        valid = D_model(img)
        Combine_model = Model(noise, valid)

        Combine_model.compile(loss="binary_crossentropy", optimizer=Adam(lr, 0.5, 0.999))

        gen = DCganDataset(lines, image_shape, batch_size)

        epoch_size = max(1, num_train//batch_size)

        for epoch in range(Init_epoch, Epoch):
            fit_one_epoch(G_model, D_model, Combine_model, epoch, epoch_size, gen, Epoch, batch_size, save_interval)

            lr = K.get_value(Combine_model.optimizer.lr) * 0.99
            K.set_value(Combine_model.optimizer.lr, lr)

            lr = K.get_value(D_model.optimizer.lr) * 0.99
            K.set_value(D_model.optimizer.lr, lr)
