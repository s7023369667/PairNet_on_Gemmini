import numpy as np
import os, glob
import matplotlib.pyplot as plt
import tensorflow as tf
from pycm import *
from keras.models import load_model
from sklearn.metrics import auc
from library import *
from collections import defaultdict


def test(sliding_windows: np.ndarray, label_len: int):
    total_res = []
    for i in range(sliding_windows.shape[0]):
        res = func(sliding_windows[i])
        total_res.append(np.argmax(res))
    count, count_result = 0, []
    previous_pre = -1
    try:
        previous_pre = total_res[0]
    except IndexError:
        print("error")
        return -1
    for i in range(len(total_res)):
        if previous_pre == total_res[i]:
            count += 1
        else:
            count_result.append([previous_pre, count])
            count = 1
        previous_pre = total_res[i]
    count_result.append([previous_pre, count])
    result = defaultdict(int)
    for i in range(len(count_result)):
        if result[count_result[i][0]] < count_result[i][1]:
            result[count_result[i][0]] = count_result[i][1]
    keys, vals = [], []
    ans = []

    for k, v in result.items():
        keys.append(k)
        vals.append(v)
    if len(vals) < label_len:
        return -1
    arg_vals = np.argpartition(np.array(vals), -label_len)[-label_len:]
    arg_vals = sorted(arg_vals)
    for i in range(label_len):
        ans.append(keys[arg_vals[i]])

    return ans


def test2(sliding_windows: np.ndarray, gesN: int, K: int):
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
    print(max_res1D)
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
    print(hash_count)
    print(hash_time)
    step1_ans, step2_ans = [0] * (K), [0] * (K)
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
    print("step1", step1_ans)
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
            if i != len(step2_ans)-1:
                step2_ans[i] = step2_ans[i + 1]
            else:
                step2_ans[i] = step2_ans[i - 1]
    print("step2:", step2_ans)
    return step2_ans


def read_sample(test_dir, gesN, save_path):
    pre_result = []
    gt_result = []
    for file_label in os.listdir(test_dir):
        txt_label = get_label(file_label)
        DO = True
        for label in txt_label:
            if label > gesN - 1:
                DO = False
                break
        if DO:
            for txt in glob.glob(test_dir + file_label + "/*.txt"):
                print(txt)
                sliding_windows = make_window_siginals(txt)
                pre = test2(sliding_windows, gesN, len(txt_label))
                # print(txt_label)
                # print(pre)
                if not pre == -1:
                    for i in range(len(pre)):
                        gt_result.append(txt_label[i])
                        pre_result.append(pre[i])
                else:
                    print(f"remove : {txt}")
                    os.remove(txt)

    if not len(pre_result) == len(gt_result):
        print("Error length ...")
    else:
        false_cnt = 0
        for i in range(len(pre_result)):
            if pre_result[i] != gt_result[i]:
                false_cnt += 1
        print(f"accuracy : {(len(pre_result) - false_cnt) / len(pre_result)}")
    cm = ConfusionMatrix(actual_vector=gt_result, predict_vector=pre_result)
    cm.print_matrix()
    # cm.print_normalized_matrix()
    cm.plot(cmap=plt.cm.Greens, number_label=True, plot_lib="matplotlib")
    plt.savefig(save_path)


def get_label(file_label):
    if '-' in file_label:
        txt_label = list(map(int, file_label.split('-')))  # 資料夾名稱1-2-5 >> [1, 2, 5]
    else:
        txt_label = [int(file_label)]
    return txt_label


if __name__ == '__main__':
    gesN = 12
    # path = '1071109_test_1-2-3-4_New12(J&W&D&j)/4-5/TD20181017-204854_(judy)_H50_N2_K4-5.txt'
    model_pairNet = load_model(f'./model/pairnet_adjRelu_model16_12_20220309.h5')
    path_dir = '1071109_test_1-2-3-4_New12_test/'
    batch_size = 1
    input_shape = model_pairNet.inputs[0].shape.as_list()
    input_shape[0] = batch_size
    func = tf.function(model_pairNet).get_concrete_function(tf.TensorSpec(input_shape, model_pairNet.inputs[0].dtype))
    read_sample(path_dir, gesN=gesN, save_path=f'./out_fig/result_confusionMatrix16_{gesN}.png')
