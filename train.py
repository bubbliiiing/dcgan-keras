import keras.backend as K
from keras import layers
from keras.models import Model
from keras.optimizers import Adam

from nets.dcgan import discriminator, generator
from utils.dataloader import DCganDataset
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #-------------------------------#
    #   卷积通道数的设置
    #-------------------------------#
    channel         = 64
    #--------------------------------------------------------------------------#
    #   如果想要断点续练就将model_path设置成logs文件夹下已经训练的权值文件。 
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''。
    #--------------------------------------------------------------------------#
    G_model_path    = ""
    D_model_path    = ""
    #-------------------------------#
    #   输入图像大小的设置
    #-------------------------------#
    input_shape     = [64, 64]

    #------------------------------#
    #   训练参数设置
    #------------------------------#
    Init_epoch      = 0
    Epoch           = 500
    batch_size      = 64
    lr              = 0.002
    #------------------------------#
    #   每隔50个step保存一次图片
    #------------------------------#
    save_interval   = 50
    #------------------------------------------#
    #   获得图片路径
    #------------------------------------------#
    annotation_path = "train_lines.txt"

    #----------------------------#
    #   生成网络和评价网络
    #----------------------------#
    G_model = generator(channel, input_shape)
    D_model = discriminator(channel, input_shape)
    
    #------------------------------------------#
    #   将训练好的模型重新载入
    #------------------------------------------#
    if G_model_path != '':
        G_model.load_weights(G_model_path, by_name=True, skip_mismatch=True)
    if D_model_path != '':
        D_model.load_weights(D_model_path, by_name=True, skip_mismatch=True)
    
    with open(annotation_path) as f:
        lines = f.readlines()
    num_train = len(lines)

    #------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        D_model.compile(loss="binary_crossentropy", optimizer=Adam(lr, 0.5, 0.999))

        D_model.trainable = False
        noise           = layers.Input(shape=(100,))
        img             = G_model(noise)
        valid           = D_model(img)
        Combine_model   = Model(noise, valid)

        Combine_model.compile(loss="binary_crossentropy", optimizer=Adam(lr, 0.5, 0.999))

        gen = DCganDataset(lines, input_shape, batch_size)

        epoch_step      = num_train // batch_size
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for epoch in range(Init_epoch, Epoch):
            fit_one_epoch(G_model, D_model, Combine_model, epoch, epoch_step, gen, Epoch, batch_size, save_interval)

            lr = K.get_value(Combine_model.optimizer.lr) * 0.99
            K.set_value(Combine_model.optimizer.lr, lr)

            lr = K.get_value(D_model.optimizer.lr) * 0.99
            K.set_value(D_model.optimizer.lr, lr)
