import keras.backend as K
import numpy as np
from tqdm import tqdm

from utils.utils import show_result


def fit_one_epoch(G_model, D_model, Combine_model, epoch, epoch_step, gen, Epoch, batch_size, save_interval):
    G_total_loss = 0
    D_total_loss = 0

    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, images in enumerate(gen):
            if iteration >= epoch_step:
                break
            y_real                  = np.ones([batch_size,1])
            y_fake                  = np.zeros([batch_size,1])

            #----------------------------------------------------#
            #   先训练评价器
            #   利用真假图片训练评价器
            #   目的是让评价器更准确
            #----------------------------------------------------#
            d_loss_real             = D_model.train_on_batch(images, y_real)

            noise                   = np.random.randn(batch_size, 100)
            G_result                = G_model.predict(noise)
            d_loss_fake             = D_model.train_on_batch(G_result, y_fake)
            d_loss                  = 0.5 * np.add(d_loss_real, d_loss_fake)

            #----------------------------------------------------#
            #   再训练生成器
            #   目的是让生成器生成的图像，被评价器认为是正确的
            #----------------------------------------------------#
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

    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('G Loss: %.4f || D Loss: %.4f ' % (G_total_loss / (epoch_step + 1), D_total_loss / (epoch_step + 1)))
    print('Saving state, iter:', str(epoch + 1))

    #----------------------------#
    #   每10个时代保存一次
    #----------------------------#
    if (epoch + 1) % 10==0:
        G_model.save_weights('logs/G_Epoch%d-GLoss%.4f-DLoss%.4f.h5'%(epoch + 1, G_total_loss / (epoch_step + 1), D_total_loss / (epoch_step + 1)))
        D_model.save_weights('logs/D_Epoch%d-GLoss%.4f-DLoss%.4f.h5'%(epoch + 1, G_total_loss / (epoch_step + 1), D_total_loss / (epoch_step + 1)))
        print('Saving state, iter:', str(epoch + 1))