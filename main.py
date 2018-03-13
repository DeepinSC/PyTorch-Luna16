import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.densenet import DenseNet121

from keras import backend as K
smooth = 1.
# dice loss
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

# read files

train_data_path = './data/out/trainImages.npy'
train_mask_path = './data/out/trainMasks.npy'
test_data_path = './data/out/testImages.npy'

train_data = np.load(train_data_path).astype(np.float32)
train_mask = np.load(train_mask_path).astype(np.float32)

train_x = Variable(torch.from_numpy(train_data))
train_y = Variable(torch.from_numpy(train_mask))

densenet = DenseNet121()
densenet.zero_grad()

output = densenet(train_x)
loss = dice_coef_loss(output, train_y)
print(loss)

optimizer = optim.SGD(densenet.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    output = densenet(train_x)
    loss = dice_coef_loss(output, train_y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print("%s MSE Loss: %s", epoch, loss)

