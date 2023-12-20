import tensorflow as tf
from tensorflow.keras import backend as K


ALPHA = 0.25
GAMMA = 1


def dice_coef(y_true, y_pred, smooth=1e-9):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f**2) + K.sum(y_pred_f**2) + smooth)

def dice_loss(y_true, y_pred):
    return tf.reduce_mean(1-dice_coef(y_true, y_pred))

def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss

def DiceBCELoss(y_true, y_pred, smooth=1e-9):
    # flatten label and prediction tensors
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    dice_loss = 1-((2. * intersection + smooth) / (K.sum(y_true_f**2) + K.sum(y_pred_f**2) + smooth))

    BCE =  K.binary_crossentropy(y_true_f, y_pred_f)                                                                                                                  
    Dice_BCE = BCE + dice_loss

    return Dice_BCE


def L2Loss(y_true, y_pred):
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)

    L2_Loss = (inputs-targets)**2

    return L2_Loss