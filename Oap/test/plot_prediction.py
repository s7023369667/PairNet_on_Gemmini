import matplotlib.pyplot as plt
import os
import math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import load_model
from Oap.train.Oap_GD_loss import OaP_GD_loss
from plot_sensor import plot_sample


def test(queue, window_size, count, gesN, Threshold):
    global PKI_t, SKI_t, PKI_c, SKI_c, anchor1_pki, anchor2_pki, anchor1_ski, anchor2_ski
    dim = 6
    Tmin = 0
    Tmax = 100
    stride_size = 1
    window = queue[:]  # (50,6)
    window = np.array([window])  # (1,50,6)
    window.reshape(1, window_size, dim)  # 將50個Sample點製作成window shape
    window = window.astype("float32")
    pre = func(window)
    for i in range(stride_size):  # stride size setting
        queue.pop(0)
    """match algorithm"""
    anchor1_pki.append(np.max(pre[1][0][1].numpy()))
    anchor2_pki.append(np.max(pre[1][0][2].numpy()))
    anchor1_ski.append(np.max(pre[2][0][1].numpy()))
    anchor2_ski.append(np.max(pre[2][0][2].numpy()))
    PKI_arg = np.argmax(pre[1])
    SKI_arg = np.argmax(pre[2])
    match_time = count - window_size
    if PKI_arg != 0:
        if pre[1][0][PKI_arg] >= Threshold:
            PKI_t = match_time
            PKI_c = PKI_arg
    if SKI_arg != 0:
        if pre[2][0][SKI_arg] >= Threshold:
            SKI_t = match_time
            SKI_c = SKI_arg
            if (PKI_c == SKI_c) & (SKI_t - PKI_t >= Tmin) & (SKI_t - PKI_t <= Tmax):
                result = SKI_c + (11 - gesN)
                region = gesture_region(PKI_t, SKI_t)
                PKI_t, PKI_c, SKI_t, SKI_c = 0, 0, 0, 0
                yield result, region


def gesture_region(PKI_t, SKI_t):
    radius = int(((SKI_t - PKI_t) / 2) * 1.5)
    gesture_region = [PKI_t - radius, SKI_t + radius]
    return gesture_region


def read_sample(txt):
    queue = []
    window_size = 50
    gesN = 2
    Threshold = 0.7
    count = 0
    with open(txt) as test_case:
        pre = []
        for line in test_case:
            s = line.split()
            count += 1
            queue.append(list(map(eval, s)))
            if len(queue) == window_size:
                for res in test(queue, window_size, count, gesN, Threshold):
                    pre.append(res)
    print(pre)
    pre_ges_region = []
    for i in range(len(pre)):
        pre_ges_region.append(pre[i][1])
    plot_sample(txt, pre_ges_region)
    plt_scores(anchor1_pki, anchor1_ski, anchor2_pki, anchor2_ski, count)


def plt_scores(anchor1_pki, anchor1_ski, anchor2_pki, anchor2_ski, count):
    zeros = [0] * (((count - len(anchor2_ski)) // 2) - 12)
    remain = [0] * (((count - len(anchor2_ski)) // 2 + (count - len(anchor2_ski)) % 2) + 12)
    anchor1_pki = zeros + anchor1_pki + remain
    anchor2_pki = zeros + anchor2_pki + remain
    anchor1_ski = zeros + anchor1_ski + remain
    anchor2_ski = zeros + anchor2_ski + remain
    plt.figure(figsize=(20, 5))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(anchor1_pki)), np.array(anchor1_pki))
    plt.plot(np.arange(len(anchor2_pki)), np.array(anchor2_pki))
    plt.axvline(x=15, color="red", linestyle='--')
    plt.axvline(x=126, color="black", linestyle='--')
    plt.axvline(x=216, color="steelblue", linestyle='--')
    plt.legend(['Class 1', 'Class 2', 'Matched for Class 2', 'Missmatched', 'Matched for Class 1'],
               loc='upper right')
    plt.xticks([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350],
               [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350])
    plt.ylabel('PKI Scores')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(anchor1_ski)), np.array(anchor1_ski))
    plt.plot(np.arange(len(anchor2_ski)), np.array(anchor2_ski))
    plt.axvline(x=38, color="red", linestyle='--')
    plt.axvline(x=153, color="black", linestyle='--')
    plt.axvline(x=261, color="steelblue", linestyle='--')
    plt.legend(['Class 1', 'Class 2', 'Matched for Class 2', 'Missmatched', 'Matched for Class 1'],
               loc='upper right')
    plt.xticks([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350],
               [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350])
    plt.ylabel('SKI Scores')
    plt.xlabel('Samples')

    plt.show()


if __name__ == "__main__":
    output_dir = "../out_figures"
    txt_path = '../test/1100920_test_(J&W&D&j&in0)/11-8-10/TD20181107-195544_(Wen)_H50_N3_K11-8-10.txt'
    model_OaP = load_model("../model_h5/best_OaP_SAM_adjust_branch_s1_1205.h5", custom_objects={
        'OaP_GD_loss': OaP_GD_loss,
        'GroupNormalization': tfa.layers.GroupNormalization})
    batch_size = 1
    input_shape = model_OaP.inputs[0].shape.as_list()
    input_shape[0] = batch_size
    func = tf.function(model_OaP).get_concrete_function(tf.TensorSpec(input_shape, model_OaP.inputs[0].dtype))
    anchor1_pki, anchor1_ski, anchor2_pki, anchor2_ski = [], [], [], []
    read_sample(txt_path)
