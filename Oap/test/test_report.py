import numpy as np
import os, glob
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.metrics import auc
import tensorflow_addons as tfa
from keras.models import load_model
from Oap.train.Oap_GD_loss import OaP_GD_loss
from pycm import *


def calculate_ED(ground_truth, predict):
    ground_truth = np.array(ground_truth)
    predict = np.array(predict)
    ground_truth = np.where(ground_truth <= (11 - gesN), 0, ground_truth)
    m = len(ground_truth)
    n = len(predict)
    DP = [[0 for i in range(m + 1)] for j in range(n + 1)]
    if m == 0:
        return n
    if n == 0:
        return m
    for i in range(1, m + 1):
        DP[0][i] = i
    for i in range(1, n + 1):
        DP[i][0] = i
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ground_truth[j - 1] == predict[i - 1]:
                tmp = 0
            else:
                tmp = 1
            top_left = DP[i - 1][j - 1] + tmp
            top = DP[i - 1][j] + 1
            left = DP[i][j - 1] + 1
            DP[i][j] = min(top, left, top_left)
    '''
        1  2  9  10  
    0   0  0  1   2  0
    0   0  0  1   2  0
    10  1  0  1   1  0
    11  2  0  1   2  0
        0  0  0   0  0
    '''
    return DP[n][m]


def test(queue, window_size, count, gesN, Threshold):
    global PKI_t, SKI_t, PKI_c, SKI_c, pki_scores, ski_scores
    dim = 6
    Tmin = 0
    Tmax = 100
    stride_size = 3
    window = queue[:]  # (50,6)
    window = np.array([window])  # (1,50,6)
    window.reshape(1, window_size, dim)  # 將50個Sample點製作成window shape
    window = window.astype("float32")
    pre = func(window)
    for i in range(stride_size):  # stride size setting
        queue.pop(0)
    """match algorithm"""
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
                gesture_region(PKI_t, SKI_t)
                PKI_t, PKI_c, SKI_t, SKI_c = 0, 0, 0, 0
                yield result


def read_samples(test_path, gesN, Threshold):
    global count, queue, reports, gt, pp
    queue = []
    window_size = 50
    count, ED = 0, 0
    TP, FP, TN, FN = 0, 0, 0, 0
    for file_label in os.listdir(test_path):
        txt_label = get_label(file_label)
        print("Test on ", test_path + file_label)
        for txt in glob.glob(test_path + file_label + "/*.txt"):
            with open(txt) as test_case:
                pre = []
                for line in test_case:
                    try:
                        s = line.split()
                        count += 1
                        queue.append(list(map(eval, s)))
                    except ValueError:
                        print(txt)
                        print("ValueError")
                        continue
                    if len(queue) == window_size:
                        for res in test(queue, window_size, count, gesN, Threshold):
                            pre.append(res)

            while len(pre) < len(txt_label):
                pre.append(0)
            txt_label = sorted(txt_label)
            pre = sorted(pre)
            gt, pp = calculate_CM(txt_label, pre)
            for i in range(len(txt_label)):
                if pre[i] > (11 - gesN) and txt_label[i] > (11 - gesN):  # 前景被正確辨識成前景(TP)
                    TP += 1
                elif pre[i] <= (11 - gesN) < txt_label[i]:  # 前景被錯誤辨識成背景(FN)
                    FN += 1
                elif pre[i] > (11 - gesN) >= txt_label[i]:  # 背景被錯誤辨識成前景(FP)
                    FP += 1
                elif pre[i] <= (11 - gesN) and txt_label[i] <= (11 - gesN):  # 背景被正確辨識成背景(TN)
                    TN += 1
            count = 0
            # print(TP, FP, TN, FN)
            # print(txt_label, pre)
            ED += calculate_ED(txt_label, pre)
            # print(ED)

    drawCM(gt, pp)
    reports = reports.append(_reports(TP, FP, TN, FN), ignore_index=True)
    # print(1 - ED / (TP + FN))
    EA.append(1 - (ED / ((TP + FN) + (FP + TN))))
    print("EA", EA)
    tpr = TP / (TP + FN)  # TP + FN = forceground
    fpr = FP / (FP + TN)  # FP + TN = background
    print(f"TPR:{tpr} , FPR:{fpr}")
    return tpr, fpr


def gesture_region(PKI_t, SKI_t):
    radius = int(((SKI_t - PKI_t) / 2) * 1.5)
    gesture_region = [PKI_t - radius, SKI_t + radius]
    return gesture_region


def get_label(file_label):
    if '-' in file_label:
        txt_label = list(map(int, file_label.split('-')))  # 資料夾名稱1-2-5 >> [1, 2, 5]
    else:
        txt_label = [int(file_label)]
    return txt_label


def calculate_CM(txt_label, pre):
    for i in range(len(txt_label)):
        if txt_label[i] > (11 - gesN) and pre[i] > (11 - gesN):
            gt.append(txt_label[i])
            pp.append(pre[i])
    return gt, pp


def drawCM(gt, pp):
    cm = ConfusionMatrix(actual_vector=gt, predict_vector=pp)
    # cm.print_matrix()
    cm.print_normalized_matrix()
    # cm.plot(cmap=plt.cm.Greens,number_label=True,plot_lib="matplotlib")


def _reports(TP, FP, TN, FN):
    d = dict()
    d['Accuracy'] = (TP + TN) / (TP + FP + TN + FN)
    d['Recall'] = TP / (TP + FN)  # 所有實際是前景中有多少能被預測為前景
    d['Precision'] = TP / (TP + FP)  # 所有預測為前景中有多少真的是前景
    d['F1_score'] = (2 * d['Precision'] * d['Recall']) / (d['Precision'] + d['Recall'])  # Precision和Recall的調和平均數
    return d


def plotAP(Precisions, Recalls):
    P, R = [1, 0], [1, 0]
    for p in Precisions:
        P.append(p)
    for r in Recalls:
        R.append(r)
    P, R = sorted(P)[::-1], sorted(R)
    ap = auc(R, P)
    plt.figure()
    plt.plot(R, P, '-*', label='PR-curve (area = %0.2f)' % ap)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR-curve')
    plt.legend(loc="lower right")
    plt.show()


def ROC(TPR, FPR):
    TPR, FPR = sorted(TPR), sorted(FPR)  # 手勢重視FPR跟Precision
    auroc = auc(FPR, TPR)
    plt.figure()
    plt.plot(FPR, TPR, '-*', label='ROC curve (area = %0.2f)' % auroc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


def pltEA(EA):
    threshold = [0.3, 0.5, 0.7, 0.9]
    plt.figure()
    plt.plot(threshold, EA, '-*')
    plt.xlabel('Thresholds')
    plt.ylabel('Edit Accuracy')
    plt.title('EA curve')
    plt.legend(loc="lower right")
    plt.show()


def plt_scores(len_sample, pki_scores, ski_scores):
    plt.subplot(2, 1, 1)
    plt.ylabel('PKI Scores')
    plt.plot(len_sample, )
    plt.subplot(2, 1, 2)
    plt.ylabel('SKI Scores')


if __name__ == '__main__':
    model_OaP = load_model("../model_h5/best_OaP_SAM_adjust_branch_s1_1205.h5", custom_objects={
        'OaP_GD_loss': OaP_GD_loss,
        'GroupNormalization': tfa.layers.GroupNormalization})
    for layer in model_OaP.layers:
        print(layer.name)
        print(np.array(layer.get_weights()).shape)
        print(layer.get_weights())
    # test_dir = "../test_data/1100920_test_(J&W&D&j&in0)/"
    test_dir = "1100920_test_(J&W&D&j&in0)/"

    # AvgLen_txt_dir = "../../data/training_data/AvgLen.txt"
    batch_size = 1
    input_shape = model_OaP.inputs[0].shape.as_list()
    input_shape[0] = batch_size
    func = tf.function(model_OaP).get_concrete_function(tf.TensorSpec(input_shape, model_OaP.inputs[0].dtype))
    PKI_t, PKI_c, SKI_t, SKI_c = 0, 0, 0, 0
    TPR, FPR, EA = [1, 0], [1, 0], []
    gt, pp = [], []
    gesN = 5
    reports = pd.DataFrame(columns=['Accuracy', 'Recall', 'Precision', 'F1_score'])
    for i in range(2, 6):
        # 目前的配對方式在threshold小的時候會有些誤差，原因是PKI一直被歸零，所以調整threshold大小來畫roc
        Threshold = (1.0 * i) / 5 - 0.1  # 0.3 0.5 0.7 0.9
        print(f"RUN({i - 1}/4) , Threshold={Threshold} : ")
        tpr, fpr = read_samples(test_dir, gesN, Threshold)
        TPR.append(tpr)
        FPR.append(fpr)
    pltEA(EA)
    # plotAP(reports['Precision'], reports['Recall'])
    ROC(TPR, FPR)