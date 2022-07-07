import datetime
import os

import keras
import keras.backend as K
import tensorflow as tf
from keras import layers
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils.multi_gpu_utils import multi_gpu_model

from nets.dcgan import discriminator, generator
from utils.callbacks import LossHistory
from utils.dataloader import DCganDataset
from utils.utils import get_lr_scheduler, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #---------------------------------------------------------------------#
    #   train_gpu   训练用到的GPU
    #               默认为第一张卡、双卡为[0, 1]、三卡为[0, 1, 2]
    #               在使用多GPU时，每个卡上的batch为总batch除以卡的数量。
    #---------------------------------------------------------------------#
    train_gpu       = [0,]
    #---------------------------------------------------------------------#
    #   如果想要断点续练就将model_path设置成logs文件夹下已经训练的权值文件。 
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''。
    #---------------------------------------------------------------------#
    G_model_path    = ""
    D_model_path    = ""
    #---------------------------------------------------------------------#
    #   卷积通道数的设置
    #---------------------------------------------------------------------#
    channel         = 64
    #---------------------------------------------------------------------#
    #   图像大小的设置，如[128, 128]
    #   设置后在训练时生成的图像看不出来，需要在预测时看单张图像。
    #---------------------------------------------------------------------#
    input_shape     = [64, 64]
    
    #------------------------------#
    #   训练参数设置
    #------------------------------#
    Init_Epoch      = 0
    Epoch           = 500
    batch_size      = 64
    
    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 2e-3
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=2e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.5
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0  
    #------------------------------------------------------------------#
    num_workers         = 4
    #------------------------------#
    #   每隔50个step保存一次图片
    #------------------------------#
    photo_save_step     = 50
    
    #------------------------------------------#
    #   获得图片路径
    #------------------------------------------#
    annotation_path = "train_lines.txt"

    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))

    #----------------------------#
    #   生成网络和评价网络
    #----------------------------#
    G_model_body = generator(channel, input_shape)
    D_model_body = discriminator(channel, input_shape)
    
    #------------------------------------------#
    #   将训练好的模型重新载入
    #------------------------------------------#
    if G_model_path != '':
        G_model_body.load_weights(G_model_path, by_name=True, skip_mismatch=True)
    if D_model_path != '':
        D_model_body.load_weights(D_model_path, by_name=True, skip_mismatch=True)

    if ngpus_per_node > 1:
        G_model = multi_gpu_model(G_model_body, gpus=ngpus_per_node)
        D_model = multi_gpu_model(D_model_body, gpus=ngpus_per_node)
    else:
        G_model = G_model_body
        D_model = D_model_body

    #--------------------------------------------#
    #   回调函数
    #--------------------------------------------#
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    callback        = TensorBoard(log_dir=log_dir)
    callback.set_model(G_model)
    loss_history    = LossHistory(log_dir)
    
    with open(annotation_path) as f:
        lines = f.readlines()
    num_train = len(lines)

    show_config(
        input_shape = input_shape, Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train
        )
    #------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : Adam(lr = Init_lr, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        D_model.compile(loss="binary_crossentropy", optimizer=optimizer)

        D_model.trainable = False
        noise           = layers.Input(shape=(100,))
        img             = G_model(noise)
        valid           = D_model(img)
        Combine_model   = Model(noise, valid)

        Combine_model.compile(loss="binary_crossentropy", optimizer=optimizer)

        gen = DCganDataset(lines, input_shape, batch_size)

        for epoch in range(Init_Epoch, Epoch):

            K.set_value(Combine_model.optimizer.lr, lr_scheduler_func(epoch))
            K.set_value(D_model.optimizer.lr, lr_scheduler_func(epoch))

            fit_one_epoch(G_model, D_model, Combine_model, loss_history, epoch, epoch_step, gen, Epoch, save_period, save_dir, photo_save_step)