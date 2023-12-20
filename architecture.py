import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization


initializer = tf.keras.initializers.GlorotNormal(seed=1234)


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


def downsampling_block(a, filters, kernel_size=(3,3), pad="same", stride=1):   #padding for retaining the resolution
    a = BatchNormalization()(a)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=pad, strides=stride, activation="relu",
                               kernel_initializer=initializer)(a)    #input a
    conv = BatchNormalization(axis=1, scale=False)(conv)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=pad, strides=stride, activation="relu",
                               kernel_initializer=initializer)(conv)    #input conv
    pool = keras.layers.MaxPool2D((2,2), (2,2))(conv)
    return conv, pool


def upsampling_block(a, skip1, filters, kernel_size=(3,3), pad="same", stride=1): #skip as list with all mods.
    us = keras.layers.UpSampling2D((2,2))(a)
    conc = keras.layers.Concatenate()([us, skip1])
    conc = BatchNormalization()(conc)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=pad, strides=stride, activation="relu",
                               kernel_initializer=initializer)(conc)
    norm2 = BatchNormalization(axis=1, scale=False)(conv)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=pad, strides=stride, activation="relu",
                               kernel_initializer=initializer)(norm2)
    return conv


def MMFF(flair, t1, t2, filters, kernel_size1=(1,1), kernel_size2=(3,3), pad="same", stride=1):
    flair = BatchNormalization()(flair)
    flairconv1 = keras.layers.Conv2D(filters, kernel_size1, padding=pad, strides=stride, activation="relu",
                                     kernel_initializer=initializer)(flair)
    flairconv1 = BatchNormalization()(flairconv1)
    flairconv2 = keras.layers.Conv2D(filters, kernel_size2, padding=pad, strides=stride, activation="relu",
                                     kernel_initializer=initializer)(flairconv1)
    t1 = BatchNormalization()(t1)
    T1conv1 = keras.layers.Conv2D(filters, kernel_size1, padding=pad, strides=stride, activation="relu",
                                  kernel_initializer=initializer)(t1)
    T1conv1 = BatchNormalization()(T1conv1)
    t1conv2 = keras.layers.Conv2D(filters, kernel_size2, padding=pad, strides=stride, activation="relu",
                                  kernel_initializer=initializer)(T1conv1)
    t2 = BatchNormalization()(t2)
    T2conv1 = keras.layers.Conv2D(filters, kernel_size1, padding=pad, strides=stride, activation="relu",
                                  kernel_initializer=initializer)(t2)
    T2conv1 = BatchNormalization()(T2conv1)
    t2conv2 = keras.layers.Conv2D(filters, kernel_size2, padding=pad, strides=stride, activation="relu",
                                  kernel_initializer=initializer)(T2conv1)
    conc = keras.layers.Concatenate()([flairconv2, t1conv2, t2conv2])
    return conc


def MSFU(multimodal, lowres, filters, kernel_size1=(1,1), kernel_size2=(2,2), kernel_size3=(3,3), pad="same", stride=1):
    lowres = BatchNormalization()(lowres)
    convLow = keras.layers.Conv2D(filters, kernel_size1, padding=pad, strides=stride, activation="relu",
                                  kernel_initializer=initializer)(lowres)
    usLow = BatchNormalization()(convLow)
    usLow = keras.layers.UpSampling2D((2,2))(convLow)
    conc = keras.layers.Concatenate()([multimodal, usLow])
    conc = BatchNormalization()(conc)
    concConv = keras.layers.Conv2D(filters, kernel_size1, padding=pad, strides=stride, activation="relu",
                                   kernel_initializer=initializer)(conc)
    concConv = BatchNormalization(axis=1, scale=False)(concConv)
    concConv2 = keras.layers.Conv2D(filters, kernel_size3, padding=pad, strides=stride, activation="relu",
                                    kernel_initializer=initializer)(concConv)
    return concConv2


def bottom(a, filters, kernel_size=(3,3), pad="same", stride=1):
    a = BatchNormalization()(a)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=pad, strides=stride, activation="relu",
                               kernel_initializer=initializer)(a)
    conv = BatchNormalization()(conv)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=pad, strides=stride, activation="relu",
                               kernel_initializer=initializer)(conv)
    return conv


def UNet(IMAGE_HEIGHT, IMAGE_WIDTH, filter):
    inputs_T1 = keras.layers.Input((IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    inputs_T2 = keras.layers.Input((IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    inputs_FLAIR = keras.layers.Input((IMAGE_HEIGHT, IMAGE_WIDTH,1))
    filters = [filter, filter*2, filter*4, filter*8, filter*16]

    p0_T1 = inputs_T1
    p0_T2 = inputs_T2
    p0_FLAIR = inputs_FLAIR

    c1_T1, p1_T1 = downsampling_block(p0_T1, filters[0])
    c1_T2, p1_T2 = downsampling_block(p0_T2, filters[0])
    c1_FLAIR, p1_FLAIR = downsampling_block(p0_FLAIR, filters[0])
    multimodal1 = MMFF(c1_FLAIR, c1_T1, c1_T2, filters[0] )

    c2_T1, p2_T1 = downsampling_block(p1_T1, filters[1])
    c2_T2, p2_T2 = downsampling_block(p1_T2, filters[1])
    c2_FLAIR, p2_FLAIR = downsampling_block(p1_FLAIR, filters[1])
    multimodal2 = MMFF(c2_FLAIR, c2_T1, c2_T2, filters[1] )

    c3_T1, p3_T1 = downsampling_block(p2_T1, filters[2])
    c3_T2, p3_T2 = downsampling_block(p2_T2, filters[2])
    c3_FLAIR, p3_FLAIR = downsampling_block(p2_FLAIR, filters[2])
    multimodal3 = MMFF(c3_FLAIR, c3_T1, c3_T2, filters[2] )

    c4_T1, p4_T1 = downsampling_block(p3_T1, filters[3])
    c4_T2, p4_T2 = downsampling_block(p3_T2, filters[3])
    c4_FLAIR, p4_FLAIR = downsampling_block(p3_FLAIR, filters[3])
    multimodal4 = MMFF(c4_FLAIR, c4_T1, c4_T2, filters[3] )

    btm_T1 = bottom(p4_T1, filters[4])
    btm_T2 = bottom(p4_T2, filters[4])
    btm_FLAIR = bottom(p4_FLAIR, filters[4])
    multimodal5 = MMFF(btm_FLAIR, btm_T1, btm_T2, filters[4] )

    up1 = MSFU(multimodal4, multimodal5, filters[3])
    up2 = MSFU(multimodal3, up1, filters[2])
    up3 = MSFU(multimodal2, up2, filters[1])
    up4 = MSFU(multimodal1, up3, filters[0])

    outputs = keras.layers.Conv2D(1, (1, 1), dtype='float32', padding="same", activation="sigmoid")(up4)
   
    model = keras.models.Model([inputs_FLAIR, inputs_T1, inputs_T2], outputs)
    return model


def callback(checkpoint_filepath, checkpoint_filepath2):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=200),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath2,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=False,
            save_freq='epoch',
            period=1
        )
    ]

    return callbacks
