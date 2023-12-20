# Author: Florian Raab
# Institution: University of Regensburg

import os
import argparse
import numpy as np
import random as rn
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.metrics import Recall, Precision
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import slicer
import losses
import choose_net
import architecture
import data_generator
import folder_structure
#exit()
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mask", type=str, default="mask1", required=True,
                    help="Number of the Mask! Default is Mask 1.")
parser.add_argument("-ep", "--epochs", type=int, default="1000", required=True,
                    help="Number of the Epochs! Default is 1000.")
parser.add_argument("-f", "--filters", type=int, default="32", required=True,
                    help="Number of the filters in first downsampling stage! Default is 32!")
parser.add_argument("-exp", "--experiment", type=str, default="ISBI", required=True,
                    help="Name of the experiment. Default is ISBI!")
parser.add_argument("-bs", "--batchsize", type=int, default="26", required=True,
                    help="Batch size. Default is 15!")
parser.add_argument("-n", "--num", type=int, default="5", required=False,
                    help="Declares, which combinations should be trained with this network! Default is 5")
parser.add_argument("-s", "--slice", type=bool, default=False, required=False,
                    help="Declares, if data should be sliced or not! Default is True")

args = parser.parse_args()
msk = args.mask
exp_name = args.experiment
num_of_combinations = args.num
to_slice = False

NUM_OF_EPOCHS = args.epochs
BS = args.batchsize
FILTERS = args.filters

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
rn.seed(1254)
tf.random.set_seed(90)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# CONSTANTS
SEED = 12345678
#BATCH_SIZE_TRAIN = 26
#BATCH_SIZE_TEST = BATCH_SIZE_TRAIN

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

access_rights = 0o777

subj_identifier = ["training_01", "training_02",
                   "training_03", "training_04", "training_05"]

timepoint_identifier = ["_01_", "_02_", "_03_", "_04_", "_05_"]

msk_identifier = ["mask1", "mask2"]

modalities = ["flair", "t1", "t2", "pd"]

base_data = "" # path where your base volumes are stored

dest_data = "" # path, where your sliced data should be stored

dest_trained_nets = f"" # path, where your trained networks should be stored


one = subj_identifier[0]
two = subj_identifier[1]
three = subj_identifier[2]
four = subj_identifier[3]
five = subj_identifier[4]



train1 = [one, two, three, four, "_123_4"]
train2 = [one, two, three, five, "_123_5"]
train3 = [one, two, four, three, "_124_3"]
train4 = [one, two, four, five, "_124_5"]
train5 = [one, three, four, two, "_134_2"]
train6 = [one, three, four, five, "_134_5"]
train7 = [two, three, four, one, "_234_1"]
train8 = [two, three, four, five, "_234_5"]
train9 = [one, two, five, three, "_125_3"]
train10 = [one, two, five, four, "_125_4"]
train11 = [one, three, five, two, "_135_2"]
train12 = [one, three, five, four, "_135_4"]
train13 = [two, three, five, one, "_235_1"]
train14 = [two, three, five, four, "_235_4"]
train15 = [one, four, five, two, "_145_2"]
train16 = [one, four, five, three, "_145_3"]
train17 = [two, four, five, one, "_245_1"]
train18 = [two, four, five, three, "_245_3"]
train19 = [three, four, five, one, "_345_1"]
train20 = [three, four, five, two, "_345_2"]

if num_of_combinations == 1:
    train_subs = [train1, train2, train3, train4, train5]
elif num_of_combinations == 2:
    train_subs = [train6, train7, train8, train9, train10]
elif num_of_combinations == 3:
    train_subs = [train11, train12, train13, train14, train15]
elif num_of_combinations == 4:
    train_subs = [train16, train17, train18, train19, train20]
else:
    train_subs = [train1, train2, train3, train4, train5, train6, train7, train8, train9, train10, train11, train12,
                  train13, train14, train15, train16, train17, train18, train19, train20]


"""
  val_loss_history = history_callback.history['val_loss']
  loss_history = history_callback.history['loss']

  numpy_val_loss_history = np.array(val_loss_history)
  numpy_loss_history = np.array(loss_history)

  np.savetxt(checkpoint_folder + "valid_loss_history_32_512.txt", numpy_val_loss_history, delimiter=",")
  np.savetxt(checkpoint_folder + "loss_history_32_512.txt", numpy_loss_history, delimiter=",")

  savepathModel = checkpoint_folder + '/LastEpoch.h5'
  model.save(savepathModel)
"""


if __name__ == "__main__":
    print("Value for slice is: ", to_slice)
    print("Train_subs List is: ", train_subs)
    if to_slice:
        for subj in subj_identifier:
            dest_folders_valid = folder_structure.create_paths(base_data, str(msk), str(subj), access_rights, False)
            dest_folders_train = folder_structure.create_paths(base_data, str(msk),  str(subj), access_rights, True)

            for i in range(0, 5):
                if i == 4 and subj != "training_03":
                    continue
                else:
                    subj_acq = subj + timepoint_identifier[i]
                    flair_img, t1_img, t2_img, pd_img, msk1_img, msk2_img = \
                        slicer.find_and_load_data(modalities, msk_identifier, base_data,
                                                  str(subj_acq))

                    if msk == msk_identifier[0]:
                        print("mask 1 was chosen!")
                        print("\n\n\n\n\n\n\n\n\n\n")
                        cnt_valid = slicer.slice_and_save_vol_img(IMG_SIZE, msk1_img, flair_img, t1_img, t2_img, pd_img,
                                                                  subj_acq, dest_folders_valid[0],
                                                                  dest_folders_valid[1], dest_folders_valid[2],
                                                                  dest_folders_valid[3], dest_folders_valid[4], False)

                        cnt_train = slicer.slice_and_save_vol_img(IMG_SIZE, msk1_img, flair_img, t1_img, t2_img, pd_img,
                                                                  subj_acq, dest_folders_train[0],
                                                                  dest_folders_train[1], dest_folders_train[2],
                                                                  dest_folders_train[3], dest_folders_train[4], True)
                    elif msk == msk_identifier[1]:
                        print("mask 2 was chosen!")
                        print("\n\n\n\n\n\n\n\n\n\n")
                        cnt_valid = slicer.slice_and_save_vol_img(IMG_SIZE, msk2_img, flair_img, t1_img, t2_img, pd_img,
                                                                  subj_acq, dest_folders_valid[0],
                                                                  dest_folders_valid[1], dest_folders_valid[2],
                                                                  dest_folders_valid[3], dest_folders_valid[4], False)

                        cnt_train = slicer.slice_and_save_vol_img(IMG_SIZE, msk2_img, flair_img, t1_img, t2_img, pd_img,
                                                                  subj_acq, dest_folders_train[0],
                                                                  dest_folders_train[1], dest_folders_train[2],
                                                                  dest_folders_train[3], dest_folders_train[4], True)
                    else:
                        print("ERROR! The rater you have chosen is not available in this training procedure!")
                        print("\n\n\n\n\n\n\n\n\n\n")
                        exit()

    for sub in train_subs:
        paths_first = folder_structure.get_paths(base_data, msk, str(sub[0]), True)
        paths_second = folder_structure.get_paths(base_data, msk, str(sub[1]), True)
        paths_third = folder_structure.get_paths(base_data, msk, str(sub[2]), True)
        paths_valid = folder_structure.get_paths(base_data, msk, str(sub[3]), False)

        paths_train = [paths_first, paths_second, paths_third]

        dest_folders_tr, dest_folders_val, checkp_folder, trained_folder = folder_structure.create_folder_for_fold(
                                                                                                   dest_trained_nets,
                                                                                                   dest_data, exp_name,
                                                                                                   sub[4], msk,
                                                                                                   access_rights)

        n = 0

        for i in range(0, 4):
            if i == 3:
                folder_structure.copy_data_for_fold(paths_valid, dest_folders_val)
            else:
                folder_structure.copy_data_for_fold(paths_train[i], dest_folders_tr)

        print("Successfully created fold's dataset!")

        NUM_TR = len(glob(dest_folders_tr[0] + '/*.png'))
        EP_STEP_TR = NUM_TR // BS

        NUM_VAL = len(glob(dest_folders_val[0] + '/*.png'))
        EP_STEP_VAL = NUM_VAL // BS

        print("Epoch steps train is: ", EP_STEP_TR)
        print("Epoch steps val is: ", EP_STEP_VAL)

        datagen_train = data_generator.datagen_train()
        datagen_val = data_generator.datagen_val()

        fl_gen_tr = data_generator.datagenerator(datagen_train, dest_folders_tr[0][:-4], IMG_SIZE, BS, SEED)
        t1_gen_tr = data_generator.datagenerator(datagen_train, dest_folders_tr[1][:-4], IMG_SIZE, BS, SEED)
        t2_gen_tr = data_generator.datagenerator(datagen_train, dest_folders_tr[2][:-4], IMG_SIZE, BS, SEED)
        msk_gen_tr = data_generator.datagenerator(datagen_train, dest_folders_tr[4][:-4], IMG_SIZE, BS, SEED)

        fl_gen_val = data_generator.datagenerator(datagen_val, dest_folders_val[0][:-4], IMG_SIZE, BS, SEED)
        t1_gen_val = data_generator.datagenerator(datagen_val, dest_folders_val[1][:-4], IMG_SIZE, BS, SEED)
        t2_gen_val = data_generator.datagenerator(datagen_val, dest_folders_val[2][:-4], IMG_SIZE, BS, SEED)
        msk_gen_val = data_generator.datagenerator(datagen_val, dest_folders_val[4][:-4], IMG_SIZE, BS, SEED)

        tr_gen = data_generator.Generator(fl_gen_tr, t1_gen_tr, t2_gen_tr, msk_gen_tr)
        val_gen = data_generator.Generator(fl_gen_val, t1_gen_val, t2_gen_val, msk_gen_val)

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0003,
            decay_steps=300,
            decay_rate=0.9,
            staircase=True)

        model = architecture.UNet(IMAGE_HEIGHT, IMAGE_WIDTH, FILTERS)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        metrics = [Recall(), Precision(), "acc"]
        model.compile(loss=losses.dice_loss, optimizer=opt, metrics=metrics)
        model.summary()

        checkp_file1 = checkp_folder + f'/MultiModalUNet_BS{BS}_{16*BS}_DSC_batchnorm_{FILTERS}_{exp_name}' + sub[4] + \
                       '.epoch_{epoch:03d}-loss_{val_loss:.6f}.h5'
        checkp_file2 = checkp_folder + f'/MultiModalUNet_BS{BS}_{16*BS}_DSC_batchnorm_{FILTERS}_{exp_name}_periodicalSaving' +\
                       sub[4] + '.epoch_{epoch:03d}-loss_{val_loss:.6f}.h5'

        history_callback = model.fit_generator(tr_gen, steps_per_epoch=EP_STEP_TR, validation_data=val_gen,
                                               validation_steps=EP_STEP_VAL, epochs=NUM_OF_EPOCHS,
                                               use_multiprocessing=True, workers=32,
                                               callbacks=architecture.callback(checkp_file1, checkp_file2))

        val_loss_history = history_callback.history['val_loss']
        loss_history = history_callback.history['loss']

        numpy_val_loss_history = np.array(val_loss_history)
        numpy_loss_history = np.array(loss_history)

        np.savetxt(checkp_folder + "/valid_loss_history.txt", numpy_val_loss_history, delimiter=",")
        np.savetxt(checkp_folder + "/loss_history.txt", numpy_loss_history, delimiter=",")
        savepath_model = checkp_folder + '/LastEpoch.h5'
        model.save(savepath_model)

        choose_net.pick_best(checkp_folder, trained_folder)

