import os
import sys
import shutil
import numpy as np
from glob import glob
from distutils.dir_util import copy_tree

val_hist_txt = '/valid_loss_history.txt'
tr_hist_txt = '/loss_history.txt'


def pick_best(base_folder, dest):
    val_hist = base_folder + val_hist_txt
    tr_hist = base_folder + tr_hist_txt
    shutil.copy(val_hist, dest)
    shutil.copy(tr_hist, dest)
    test_array_temp = np.loadtxt(val_hist)
    test_array = np.copy(test_array_temp)
    step_size = 50
    denominator = step_size - 1
    var = 1000
    best_epoch_start = 0
    start = len(test_array) - 150
    end = len(test_array) - step_size
    best_epoch_end = 0
    var_list = []
    first_epoch_list = []
    last_epoch_list = []
    best_loss_list = []
    loss_list_for_choosing_net = []
    epoch_list_loss = []
    best_loss_epoch_list = []
    mean_loss_list = []
    for j in range(start, end):
        best_loss = 1000
        mean_temp = 0
        var_temp = 0
        nominator = 0
        best_loss_epoch = 0
        for i in range(j, j+step_size):
            mean_temp += test_array[i]
            if(test_array[i] < best_loss):
                best_loss = test_array[i]
                best_loss_epoch = i+1
        mean = mean_temp / step_size
        mean_loss_list.append(mean)

        for t in range (j, j+step_size):
            nominator += (test_array[i] - mean)**2
        var_temp = nominator/denominator

        var_list.append(var_temp)
        first_epoch_list.append(j+1)
        last_epoch_list.append(j+step_size+1)
        best_loss_list.append(best_loss)
        best_loss_epoch_list.append(best_loss_epoch)

        if (var_temp <= var):
            var = var_temp
            best_epoch_start = j+1
            best_epoch_end = j + step_size + 1

    print(f"Lowest variance is: {var} in the sequence between epoch {best_epoch_start} and {best_epoch_end}")

    var_array = np.asarray(var_list)
    first_epoch_array = np.asarray(first_epoch_list)
    last_epoch_array = np.asarray(last_epoch_list)
    best_loss_array = np.asarray(best_loss_list)
    best_loss_epoch_array = np.asarray(best_loss_epoch_list)
    mean_loss_array = np.asarray (mean_loss_list)

    choose_net_array = np.column_stack((mean_loss_array, var_array, best_loss_array, best_loss_epoch_array,
                                        first_epoch_array, last_epoch_array))

    resulting_array = dest + 'valid_loss_plus_epoch_number.txt'

    choose_net_array_sorted = choose_net_array[np.lexsort((choose_net_array[:, 0], choose_net_array[:, 1]))]

    for j in range (0, 1):
        best_start = int(choose_net_array_sorted[j][4])
        best_end = int(choose_net_array_sorted[j][5])
        for i in range (best_start-1, best_end):
            loss_list_for_choosing_net.append(test_array[i])
            epoch_list_loss.append(i+1)

    loss_array_for_chosing_net = np.asarray(loss_list_for_choosing_net)
    epoch_array_loss = np.asarray(epoch_list_loss)

    end_array = np.column_stack((loss_array_for_chosing_net, epoch_array_loss))
    end_array_sorted = end_array[np.lexsort((end_array[:, 1], end_array[:, 0]))]

    print(end_array_sorted)

    for i in range(0, len(end_array_sorted)):
        if (end_array_sorted[i][1]%3 == 0):
            substring = ''
            print(f'Loss: {end_array_sorted[i][0]} at Epoch {str(int(end_array_sorted[i][1]))} was the best option, '
                  f'that has been saved!')

            if (end_array_sorted[i][1] < 100):
                substring = 'epoch_0' + str(int(end_array_sorted[i][1]))
            else:
                substring = 'epoch_' + str(int(end_array_sorted[i][1]))

            print(f"substring is: {substring}")

            neuronal_nets = glob(base_folder + '/*.h5')
            for net in neuronal_nets:
                if substring in net:
                    shutil.copy(net, dest)
                    print(f"The best net: {net} \n has been copied to the corresponding destination!")
                    os.system(f'rm -r {base_folder}')
                    break;
            break;
    print("Best net was chosen!")

    return
