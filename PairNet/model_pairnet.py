import keras.backend as K
from keras.models import Model
from keras.layers import *
from termcolor import colored, cprint
import pprint
import os
import tensorflow as tf
from keras.optimizers import Adadelta

'''
 Conv1D(filters, kernel size, strides, padding, activation)
     filters - 輸出的維度
     kernel size - 卷積核大小
     strides - 步長
     padding - 填補以符合
     activation - 可選擇活化函數
 '''

def build_model(input_shape, dim, odim, channel):
    cprint(colored('Now Import Model - Pairnet'), 'magenta', 'on_grey')
    input1 = Input(shape=(input_shape, dim))

    out = Conv1D(channel, 3, strides=1, activation=None, use_bias=False, name='conv1d_1')(input1)  # (?, 48, 128)
    out = BatchNormalization(name='batch_normalization_1')(out)
    out = Activation('relu', name='relu_1')(out)

    out = Conv1D(channel, 2, strides=2, activation=None, use_bias=False, name='conv1d_2')(out)  # (?, 24, 128)
    out = BatchNormalization(name='batch_normalization_2')(out)
    out = Activation('relu', name='relu_2')(out)

    out = Conv1D(channel, 2, strides=2, activation=None, use_bias=False, name='conv1d_3')(out)  # (?, 12, 128)
    out = BatchNormalization(name='batch_normalization_3')(out)
    out = Activation('relu', name='relu_3')(out)

    out = Conv1D(channel * 2, 2, strides=2, activation=None, use_bias=False, name='conv1d_4')(out)  # (?, 6, 256)
    out = BatchNormalization(name='batch_normalization_4')(out)
    out = Activation('relu', name='relu_4')(out)

    out = Conv1D(channel * 2, 2, strides=2, activation=None, use_bias=False, name='conv1d_5')(out)  # (?, 3, 256)
    out = BatchNormalization(name='batch_normalization_5')(out)
    out = Activation('relu', name='relu_5')(out)

    out = GlobalAveragePooling1D(name='global_average_pooling1d')(out)  # (?, 256)'

    out = Dense(odim, activation='softmax', name='dense_1')(out)  # (?, 12)

    model = Model(inputs=input1, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    gesN = 12
    channel = 64
    # model_tmp = build_model(50, 6, gesN, channel)
    model_tmp = build_model(50, 6, gesN, channel)
    # print('total_float_ops - ', get_flops(model_tmp))
    model_tmp.summary()

    """
      可以手動使層變成不可訓練 -> transfer learning 可能會用到
    """
    # model_tmp.layers[2].trainable = False
    # for layer in model_tmp.layers:
    #     print(layer.name, layer.trainable)

    # with open('model_pairnet.txt', 'w') as fh:
    #     model_tmp.summary(print_fn=lambda x: fh.write(x + '\n'))
