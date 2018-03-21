from __future__ import print_function

import math
from random import randint
import numpy as np
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras import backend as K

from data_processing.models.fcdensenet.fcDensenet import DenseNetFCN

working_path = "../../data/out/"

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.

downsample = True
img_rows = 128
img_cols = 128


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# use generator to reduce the pressure of GPU
def data_generator(data, targets, batch_size):
    ylen = len(targets)
    loopcount = ylen // batch_size
    while (True):
        i = randint(0, loopcount)
        yield data[i * batch_size:(i + 1) * batch_size], targets[i * batch_size:(i + 1) * batch_size]


# How to use net?
def train_and_predict(use_existing=False):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train = np.load(working_path+"trainImages.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path+"trainMasks.npy").astype(np.float32)

    imgs_test = np.load(working_path+"testImages.npy").astype(np.float32)
    imgs_mask_test_true = np.load(working_path+"testMasks.npy").astype(np.float32)

    if downsample:
        imgs_train = np.resize(imgs_train, [imgs_train.shape[0], imgs_train.shape[1], img_rows, img_cols])
        imgs_mask_train = np.resize(imgs_mask_train, [imgs_mask_train.shape[0], imgs_mask_train.shape[1], img_rows, img_cols])
        imgs_test = np.resize(imgs_test, [imgs_test.shape[0], imgs_test.shape[1], img_rows, img_cols])
        imgs_mask_test_true = np.resize(imgs_mask_test_true, [imgs_mask_test_true.shape[0], imgs_mask_test_true.shape[1], img_rows, img_cols])

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean  # images should already be standardized, but just in case
    imgs_train /= std

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    # model = unet()
    model = DenseNetFCN(input_shape=(1, img_rows, img_cols),nb_dense_block=int(math.log(img_rows, 2)),)
    # Saving weights to unet.hdf5 at checkpoints
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
    #
    # Should we load existing weights? 
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        model.load_weights('./unet.hdf5')
        
    # 
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970 
    # I was able to run 20 epochs with a training set size of 320 and 
    # batch size of 2 in about an hour. I started getting reseasonable masks 
    # after about 3 hours of training. 
    #
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    
    # model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=20, verbose=1, shuffle=True,
    # callbacks=[model_checkpoint])

    # 使用generator降低内存，TODO：载入val而不是test
    train_generator = data_generator(imgs_train, imgs_mask_train, batch_size=2)
    model.fit_generator(train_generator,
                        steps_per_epoch=100,
                        epochs=20,
                        max_queue_size=100,
                        validation_data=(imgs_test, imgs_mask_test_true),
                        validation_steps=5,
                        callbacks=[model_checkpoint])

    # loading best weights from training session
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('./unet.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    num_test = len(imgs_test)
    imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
    if downsample:
        imgs_mask_test = np.resize(imgs_mask_test, [imgs_mask_test.shape[0], imgs_mask_test.shape[1], img_rows, img_cols])
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
    np.save('masksTestPredicted.npy', imgs_mask_test)
    mean = 0.0
    for i in range(num_test):
        mean += dice_coef_np(imgs_mask_test_true[i, 0], imgs_mask_test[i, 0])
    mean /= num_test
    print("Mean Dice Coeff : ",mean)


if __name__ == '__main__':
    train_and_predict(False)
