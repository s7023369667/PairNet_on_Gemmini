import numpy as np
import random
import copy
from keras.utils.np_utils import to_categorical
from preprocessing_utils import pre_normalization


def preprocessing(train, train_label):
    x, y = [], []
    length = 50
    dim = 6
    out_dim = 12

    for i in range(len(train)):  # train裡面放的是所有資料(扣掉最後50筆)
        if len(train[i]) != len(train_label[i]) or len(train[i]) == 0:
            continue

        # over-lapping 如同測試集的做法 -> 實驗結過不理想
        l = len(train[i])

        for j in range(0, (l - length + 1), length):
            x.append(train[i][j:j + length])
            y.append(to_categorical(train_label[i][j:j + length], out_dim))

    c = list(zip(x, y))
    random.shuffle(c)
    x, y = zip(*c)

    x = np.array(x).reshape(len(x), length, dim).astype(float)
    y = np.array(y).reshape(len(y), length, out_dim).astype(int)

    """
      切割成 訓練集 / 驗證集
    """
    val_split_len = int(len(x) * 0.75)

    x_train = x[:val_split_len]  # (val_split_len) %
    y_train = y[:val_split_len]

    x_test = x[val_split_len:]  # (1 - val_split_len) %
    y_test = y[val_split_len:]

    return x_train, y_train, x_test, y_test
