from os import listdir
import numpy as np
import os
import matplotlib.pyplot as plt


def zero_mean_normalization(list_tmp):
    """
      範圍：使資料 平均為0， 標準差為1
      Normalization = x - mean / std
    """
    n_list = (list_tmp - np.mean(list_tmp)) / np.std(list_tmp)
    return n_list


def min_max_normalization(list_tmp):
    """
      範圍：[0, 1]
      Normalization = (x - min) / (Max - min)
    """
    n_list = (list_tmp - np.min(list_tmp)) / (np.max(list_tmp) - np.min(list_tmp))
    return n_list


def min_max_normalization_expand(list_tmp):
    """
      範圍：[-1, 1]
      Normalization = (x - min / Max - min) * 2 - 1
    """
    n_list = ((list_tmp - np.min(list_tmp)) / (np.max(list_tmp) - np.min(list_tmp))) * 2 - 1
    return n_list


def pre_normalization(x):

    # FIXME 目前是預設針對6軸 "各別" 正規化
    if np.array(x).shape[-1] == 6:
        ax = np.array(x)[:, 0]
        n_ax = min_max_normalization(ax)
        ay = np.array(x)[:, 1]
        n_ay = min_max_normalization(ay)
        az = np.array(x)[:, 2]
        n_az = min_max_normalization(az)
        gx = np.array(x)[:, 3]
        n_gx = min_max_normalization(gx)
        gy = np.array(x)[:, 4]
        n_gy = min_max_normalization(gy)
        gz = np.array(x)[:, 5]
        n_gz = min_max_normalization(gz)

        list_tmp = [n_ax, n_ay, n_az, n_gx, n_gy, n_gz]
        # numpy.swapaxes(a, axis1, axis2) -> Interchange two axes of an array
        # (6, 79) -> (79, 6)
        normalization_arr = np.swapaxes(list_tmp, 0, 1)  # Type: Numpy array
        normalization_list = normalization_arr.tolist()
        return normalization_list
    else:
        print('Normalization Cannot handle shape - {}'.format(np.shape(x)))
        os.exit(0)







