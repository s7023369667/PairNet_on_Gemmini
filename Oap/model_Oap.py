from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
import tensorflow_addons as tfa
import keras.backend as K
import numpy as np


def OaP_GD_loss(y_true, y_pred):
    qq = 0.00001
    LF = K.switch(K.equal(y_true, 1),
                  lambda: -1 * K.pow((1 - y_pred), 2) * K.log(y_pred + qq),
                  lambda: -1 * K.pow((1 - y_true), 4) * K.pow(y_pred, 2) * K.log(1 - y_pred + qq))
    return LF
    # y_true = 1  K.pow( (1-y_pred) ,2) * K.log(y_pred)
    # else        K.pow( (1-y_true) ,4) * K.pow(y_pred,2) * K.log(1-y_pred)


def build_model(input_shape, dim, odim):
    """part1 backbone"""
    channel = 64
    In = Input(shape=(input_shape, dim), name='In')
    NML = tfa.layers.GroupNormalization(groups=6, name='Gp')(In)

    conv1_1 = Conv1D(channel, 3, strides=1, padding='same', activation=None, use_bias=False, name='conv1_1')(NML)
    relu1_1 = Activation('relu', name='relu1_1')(conv1_1)
    NML1_1 = tfa.layers.GroupNormalization(groups=16, name='Gp1_1')(relu1_1)

    conv1_2 = Conv1D(channel, 2, strides=1, padding='same', activation=None, use_bias=False, name='conv1_2')(NML1_1)
    relu1_2 = Activation('relu', name='relu1_2')(conv1_2)
    NML1_2 = tfa.layers.GroupNormalization(groups=16, name='Gp1_2')(relu1_2)

    conv1_3 = Conv1D(channel, 2, strides=1, padding='same', activation=None, use_bias=False, name='conv1_3')(NML1_2)
    relu1_3 = Activation('relu', name='relu1_3')(conv1_3)
    NML1_3 = tfa.layers.GroupNormalization(groups=16, name='Gp1_3')(relu1_3)

    conv1_4 = Conv1D(channel, 2, strides=1, padding='same', activation=None, use_bias=False, name='conv1_4')(NML1_3)
    relu1_4 = Activation('relu', name='relu1_4')(conv1_4)
    NML1_4 = tfa.layers.GroupNormalization(groups=16, name='Gp1_4')(relu1_4)

    """Short-cut add NML1_2"""
    add1_1 = add([NML1_2, NML1_4], name='add1_1')
    conv1_5 = Conv1D(channel, 2, strides=1, padding='same', activation=None, use_bias=False, name='conv1_5')(add1_1)
    relu1_5 = Activation('relu', name='relu1_5')(conv1_5)
    NML1_5 = tfa.layers.GroupNormalization(groups=16, name='Gp1_5')(relu1_5)
    """Short-cut add NML1_1"""
    add1_2 = add([NML1_1, NML1_5], name='add1_2')
    conv1_6 = Conv1D(channel, 2, strides=1, padding='same', activation=None, use_bias=False, name='conv1_6')(add1_2)
    relu1_6 = Activation('relu', name='relu1_6')(conv1_6)
    NML1_6 = tfa.layers.GroupNormalization(groups=16, name='Gp1_6')(relu1_6)

    pl1_6 = AveragePooling1D(pool_size=4, strides=2, padding='valid', name='pl1_6')(NML1_6)
    flt1_6 = Flatten(name='flt1_4')(pl1_6)
    output_1 = Dense(odim, activation='softmax', name='output_1')(flt1_6)
    '''part1 end'''

    ''' part2 最左邊分支(PKI)'''
    conv2_1 = Conv1D(channel, 2, strides=1, padding='valid', activation=None, use_bias=False, name='conv2_1')(NML1_6)
    relu2_1 = Activation('relu', name='relu2_1')(conv2_1)
    NML2_1 = tfa.layers.GroupNormalization(groups=16, name='Gp2_1')(relu2_1)

    # conv2_2 = Conv1D(channel, 2, strides=1, padding='valid', activation=None, use_bias=False, name='conv2_2')(NML2_1)
    # relu2_2 = Activation('relu', name='relu2_2')(conv2_2)
    # NML2_2 = tfa.layers.GroupNormalization(groups=16, name='Gp2_2')(relu2_2)
    pl2_2 = AveragePooling1D(pool_size=4, strides=2, padding='valid', name='pl2_2')(NML2_1)
    flt2_2 = Flatten(name='flt2_2')(pl2_2)

    output_2 = Dense(odim, activation='softmax', name='output_2')(flt2_2)
    '''part2 end'''
    ''' part3 中間分支(SKI)'''
    conv3_1 = Conv1D(channel, 2, strides=1, padding='valid', activation=None, use_bias=False, name='conv3_1')(NML1_6)
    relu3_1 = Activation('relu', name='relu3_1')(conv3_1)
    NML3_1 = tfa.layers.GroupNormalization(groups=16, name='Gp3_1')(relu3_1)

    # conv3_2 = Conv1D(channel, 2, strides=1, padding='valid', activation=None, use_bias=False, name='conv3_2')(NML3_1)
    # relu3_2 = Activation('relu', name='relu3_2')(conv3_2)
    # NML3_2 = tfa.layers.GroupNormalization(groups=16, name='Gp3_2')(relu3_2)
    pl3_2 = AveragePooling1D(pool_size=4, strides=2, padding='valid', name='pl3_2')(NML3_1)
    flt3_2 = Flatten(name='flt3_2')(pl3_2)

    output_3 = Dense(odim, activation='softmax', name='output_3')(flt3_2)
    '''part3 end'''

    model = Model(inputs=In, outputs=[output_1, output_2, output_3])
    Ada = optimizers.Adadelta(learning_rate=1)
    model.compile(loss=[OaP_GD_loss, OaP_GD_loss, OaP_GD_loss], optimizer=Ada, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    gesN = 2
    model_main = build_model(50, 6, gesN + 1)
    model_main.summary()
    for layer in model_main.layers:
        print(layer.name)
        print(np.array(layer.get_weights()).shape)
        print(layer.get_weights())
