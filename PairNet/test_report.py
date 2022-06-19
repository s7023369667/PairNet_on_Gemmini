import numpy as np
import os, glob
import matplotlib.pyplot as plt
import tensorflow as tf
from pycm import *
from keras.models import load_model
from sklearn.metrics import auc
from library import *
from collections import defaultdict
from Qconv1d import *
import h5py


def test(sliding_windows: np.ndarray, gesN: int, K: int, func):
    batch_size = sliding_windows.shape[0]
    max_res1D = []
    for i in range(batch_size):
        res = func(sliding_windows[i])
        max_res1D.append(np.argmax(res))
    count = 0
    previous_predict = -1
    try:
        previous_predict = int(max_res1D[0])
    except IndexError:
        return -1
    # print(max_res1D)
    pre_label, pre_count = [0] * batch_size, [0] * batch_size
    idx = 0
    for i in range(batch_size):
        if previous_predict == max_res1D[i]:
            count += 1
        else:
            pre_label[idx] = previous_predict
            pre_count[idx] = count
            idx += 1
            count = 1
        previous_predict = max_res1D[i]
    pre_label[idx] = previous_predict
    pre_count[idx] = count
    idx += 1
    hash_count, hash_time = [0] * gesN, [0] * gesN
    time = 1
    for i in range(idx):
        if hash_count[pre_label[i]] < pre_count[i]:
            hash_count[pre_label[i]] = pre_count[i]
            hash_time[pre_label[i]] = time
            time += 1
    # print(hash_count)
    # print(hash_time)

    step1_ans, step2_ans = [0] * K, [0] * K
    pre_max, pre_min = float('inf'), float('-inf')
    for i in range(K):
        max, label = -1, -1
        for j in range(gesN):
            if max < hash_count[j] <= pre_max:
                max = hash_count[j]
                label = j
        pre_max = max
        hash_count[label] += max
        step1_ans[i] = label
    # print("step1", step1_ans)
    for i in range(K):
        min_time, label = float('inf'), -1
        for j in range(K):
            if min_time > hash_time[step1_ans[j]] > pre_min:
                min_time = hash_time[step1_ans[j]]
                label = step1_ans[j]
        pre_min = min_time
        step2_ans[i] = label
    for i in range(len(step2_ans)):
        if step2_ans[i] == 0:
            if i != len(step2_ans) - 1:
                step2_ans[i] = step2_ans[i + 1]
            else:
                step2_ans[i] = step2_ans[i - 1]
    # print("step2", step2_ans)
    return step2_ans


def read_sample(test_dir, gesN, model, save_path=None):
    batch_size = 1
    input_shape = model.inputs[0].shape.as_list()
    input_shape[0] = batch_size
    func = tf.function(model).get_concrete_function(tf.TensorSpec(input_shape, model.inputs[0].dtype))
    pre_result, gt_result = [], []
    acc = 0
    # d = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0}
    for label in os.listdir(test_dir):
        path = os.path.join(test_dir, label)
        for txt in os.listdir(path):
            # d[str(label)] += 1
            print(os.path.join(path, txt))
            sliding_windows = make_window_siginals(os.path.join(path, txt))
            pre = test(sliding_windows=sliding_windows, gesN=gesN, K=1, func=func)
            # print(label)
            # print(pre[0])
            if not pre == -1:
                gt_result.append(eval(label))
                pre_result.append(pre[0])
            else:
                print(f"remove : {txt}")
                os.remove(txt)
    false_cnt = 0
    for i in range(len(pre_result)):
        if pre_result[i] != gt_result[i]:
            false_cnt += 1
    acc = (len(pre_result) - false_cnt) / len(pre_result)
    print(f"accuracy : {acc}")
    cm = ConfusionMatrix(actual_vector=gt_result, predict_vector=pre_result)
    cm.print_matrix()
    # print(d)
    # cm.print_normalized_matrix()
    cm.plot(cmap=plt.cm.Greens, number_label=True, plot_lib="matplotlib")
    plt.show()
    # plt.savefig()
    return acc


def main():
    res = []
    gesN = 12
    path_dir = '../OapNet/train/train_raw/1071101_Johny[5]&Wen[5]_train_New12(J&W)/'
    models = ['./model/pairnet_model16_12_20220503.h5', './model/pairnet_model32_12_20220605.h5',
              './model/pairnet_model64_12_20220503.h5']
    for m in models:
        model_pairNet = load_model(f'{m}')
        res.append(read_sample(path_dir, gesN=gesN, model=model_pairNet))

    print(res)
    # # TF : [0.9888991674375578, 0.9935245143385754, 0.9935245143385754]
    # # {'1': 81, '2': 100, '3': 100, '4': 100, '5': 100, '6': 100, '7': 100, '8': 100, '9': 100, '10': 100, '11': 100}
    # # 0612 : [0.9870490286771508, 0.9824236817761333, 0.9953746530989824]


if __name__ == '__main__':
    main()

    # path_dir = '1071109_test_1-2-3-4_New12_test/'
    # model_pairNet = load_model(f'./pairnet_model16_10_20220522_seq.h5')
    # batch_size = 1
    # input_shape = model_pairNet.inputs[0].shape.as_list()
    # input_shape[0] = batch_size
    # func = tf.function(model_pairNet).get_concrete_function(tf.TensorSpec(input_shape, model_pairNet.inputs[0].dtype))
    # read_sample(test_dir=path_dir, gesN=gesN, model=func)
