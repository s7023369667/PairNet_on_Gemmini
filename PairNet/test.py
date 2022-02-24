from keras.models import load_model
from preprocessing_utils import *
from os import listdir, path
from collections import Counter
import numpy as np
from texttable import Texttable
from filtered_result_sequential import sequential_filtered_result
import os
from termcolor import colored, cprint
from keras import backend as K
from time import *
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import colorama

colorama.init()


# TODO
# "IndexError: tuple index out of range"
# Python3.6中的Lambda函式，在Python3.5會出現的問題
# https://stackoverflow.com/questions/50551096/convert-keras-model-from-python3-6-to-3-5
# https://github.com/keras-team/keras/issues/7297


def texttable_print_category_hit_rate(Correct_num_arr, label_num_arr, Result_arr):
    #
    # 為保留手勢0能被當作一種手勢的可能性，所以矩陣大小設為 12
    #
    table = Texttable()
    table.set_cols_width([10, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
    count_str_list = [str(x) for x in range(12)]
    table.add_rows([["gesture", *count_str_list],
                    ["Correct", *Correct_num_arr],
                    ['Label_num', *label_num_arr],
                    ['Result', *Result_arr]])
    return table


def output_confusion_matrix(test_base, test_index, model_name,
                            Confusion_Matrix, label_num_arr, Result_arr,
                            Model_Accuracy):
    CM_out_name_minus_path = 'Confusion_Matrix/' + str(test_base[test_index]) + '/'  # 無法一次建立兩層不存在的資料夾
    if not os.path.isdir(CM_out_name_minus_path):
        os.makedirs(CM_out_name_minus_path)

    CM_out_name_minus = CM_out_name_minus_path + model_name.split('.')[0] + '_' + test_base[test_index] + '.csv'

    with open(CM_out_name_minus, 'w') as CMM_out:
        CMM_out.write(' ,')
        for t in range(0, 12):
            # CMM_out.write(str(t + 1) + ',')
            CMM_out.write(str(t) + ',')
        CMM_out.write('Denominator,Hit rate' + '\n')

        for i in range(0, 12):
            # CMM_out.write(str(i + 1) + ',')
            CMM_out.write(str(i) + ',')
            for j in range(0, 12):
                if label_num_arr[i] == 0:
                    Confusion_Matrix[i][j] = 0
                else:
                    Confusion_Matrix[i][j] = round(Confusion_Matrix[i][j] / label_num_arr[i], 4)
                CMM_out.write(str(Confusion_Matrix[i][j]) + ',')
            CMM_out.write(str(label_num_arr[i]) + ',')  # Denominator
            CMM_out.write(str(Result_arr[i]) + '\n')  # Hit rate
        CMM_out.write('\n')
        CMM_out.write('Accuracy,' + str(Model_Accuracy) + '\n')


# 用來計算辨識結果，來取最後的正確率
all_count = 0
all_correct = 0
# FIXME 在訓練時最後出來結果是　1 x 12, 所以實際上還是有可能會被辨識出 手勢"0"
# FIXME 但目前做法會是 (Label - 1)，所以假設有0的會對應到最後一個(也就是手勢11)
label_num_arr = np.array(np.zeros(12))
Correct_num_arr = np.array(np.zeros(12))
Result_arr = np.array(np.zeros(12))
test_number_for_predict = 0  # 計算每一個測試集有多少資料中(以一行為單位)

Confusion_Matrix = np.array(np.zeros((12, 12)))

current_label_count = 0
current_label_correct = 0
total_file_count = 0
total_label_count = 0


def test(f, model, label, test_base, test_index, model_name, is_Original_RNN):
    global all_count, all_correct, test_number_for_predict
    global label_num_arr, Correct_num_arr, Result_arr
    global current_label_count, current_label_correct, total_file_count, total_label_count
    global User_false_count

    sample = []

    """
      dim -> 決定輸入資料的維度 (六軸 / 陀螺儀 / 加速度計)
    """
    dim = 6
    split_length = 50

    three_axis_part = ''  # 'ACCELEROMETER' / 'GYROSCOPE'

    # Read the sequence
    with open(f) as test_case:
        for data_line in test_case:
            t = data_line.strip().split(' ')
            if dim == 6:
                sample.append(list(map(float, t)))
            elif dim == 3:
                if three_axis_part == 'ACCELEROMETER':
                    sample.append(list(map(float, t[0:3])))  # 加速度計
                elif three_axis_part == 'GYROSCOPE':
                    sample.append(list(map(float, t[3:6])))  # 陀螺儀
                else:
                    print('Did not Assign Three-Axis Data')
                    os._exit(0)

    sample = sample[:-50]  # sample最後50 + 1筆也要扣掉

    if not is_Original_RNN:
        """
          兩個維度
        """
        sample_sliding = []  # 有 sliding window 版

        # 25 + 162 + 24 = 211
        # 211 - 50 + 1 = 162
        sample_padding = [[0] * dim] * 25 + sample + [[0] * dim] * 24  # (50, 6)

        last_count = 0
        for i in range(len(sample_padding) - split_length + 1):
            test_number_for_predict += 1

            sample_tmp = sample_padding[i:i + split_length]  # sliding with window size
            sample_sliding.append(sample_tmp)

        #  raw_result -> 12 classes 的 softmax值
        #  (74, 50, 6) -> (74, 50, 12)
        raw_result_sliding = model.predict(np.array(sample_sliding).reshape(len(sample_sliding), split_length, dim))

        label_str1 = '-'.join(str(x) for x in label)
        file_name = (f.split('\\')[-1])[:-4]
        model_short = model_name_short.split('_')[-1]
        model_output = '({}) {} {}'.format(label_str1, model_short, file_name)

        filtered_result_sliding_version = []
        for i in raw_result_sliding:
            label_tmp = i.argmax()  # (12,) -> inttf.compat.v1.
            filtered_result_sliding_version.append(label_tmp)
        guess = sequential_filtered_result(filtered_result_sliding_version, len(label),
                                           isDuplicated=False)  # 防止出現長度無法對齊

    else:
        """
          三個維度
        """
        sample_non_sliding = sample  # 傳統LSTM一次只會進1*6

        raw_result_non_sliding = model.predict(np.array(sample_non_sliding).reshape(1, len(sample_non_sliding), dim))
        raw_result_non_sliding = np.squeeze(raw_result_non_sliding)  # (1, 74, 12) -> (74, 12)

        # TODO 由於Origin LSTM 沒有做 sliding window，只能針對整體做正規化(?
        filtered_result_non_sliding_version = []
        for j in raw_result_non_sliding:  # raw_result -> (162, 12)  : txt中所有資料
            filtered_result_non_sliding_version.append(j.argmax())  # filtered_result  ->  (162, )  : txt中每一筆預測的手勢

        guess = sequential_filtered_result(filtered_result_non_sliding_version, len(label),
                                           isDuplicated=False)  # TODO 給Original RNN用

    # guess = list(map(lambda x: x[0], Counter(filtered_result).most_common(len(label))))  # 最一開始的方法

    """
      描述少了哪些手勢
    """
    if len(guess) != len(label):
        print('{}File Name : {}'.format(' ' * 4, f))
        # print('{}{}'.format(' ' * 4, filtered_result_sliding_version))
        print('{}True Label: {} ; Predict: {} -> '.format(' ' * 6, label, guess), end='')
        guess = guess + [guess[-1]] * (len(label) - len(guess))  # 會出差超過兩個的情況
        cprint(colored(guess, 'white', 'on_grey'), end='\n\n')

    # 輸出最多的 len(label) 個元素  -> len(label) 代表有幾個手勢
    # Counter(filtered_result) -> Counter({2: 72, 1: 65, 7: 17, 3: 6, 4: 1, 5: 1})
    # guess -> [2, 1]
    # lambda -> 來定義函式
    #        -> x : [(2, 72), (1, 65)]
    #        -> x[0] : [2, 1]

    # print(type(guess))
    # print(guess)

    all_count += len(label)
    current_label_count += len(label)
    for i in label:
        label_num_arr[i] += 1  # 計算每種手勢出現次數(標籤)

    #
    # 以 set 的觀點， {1, 2, 1} == {1, 1, 2}
    # 但若有順序情況下會不同
    #

    """
      計算Hit rate時仍就沒有順序之分
    """
    comapre_Result = set(guess).intersection(label)

    '''
      這裡並未把辨識結果的「順序」拿來討論，只要有被辨識出來就算
      因此，先把 實際 與 辨識結果 都有的部分(交集)拿出來
      剩下來的就是沒能辨識正確的 -> 額外放進 Confusion Matrix
    '''
    tmp_label = label.copy()
    tmp_guess = guess.copy()
    for i in list(comapre_Result):
        Confusion_Matrix[i][i] += 1  # 辨識結果與實際上相同
        tmp_label.remove(i)
        tmp_guess.remove(i)

    """
      計算 Confusion Matrix中兩手勢不同的狀況
    """
    tmp_index = 0
    for _ in tmp_label:
        Confusion_Matrix[tmp_label[tmp_index]][tmp_guess[tmp_index]] += 1
        tmp_index += 1

    """
      計算辨識正確的手勢個數
    """
    all_correct += len(comapre_Result)
    current_label_correct += len(comapre_Result)
    for correct in comapre_Result:
        Correct_num_arr[correct] += 1  # 計算有被辨識出來的手勢


if __name__ == '__main__':

    Total_start = time()

    Model_name_Reader = open('Model_name.txt', 'r')

    model_base = []
    for line in open('Model_name.txt'):
        model_tmp = Model_name_Reader.readline()
        model_base.append(model_tmp[:-1])  # 處理字串後面的 '\n'

    test_base = ['1071109_test_1-2-3-4_New12(J&W&D&j)']
    test_model_nums = len(model_base)  # 讀進已經訓練好的模型的名稱

    """
        先看過一輪，設定csv的第一行 ( 代表的手勢組合 )，紀錄每個組合各自的正確率00
    """
    # Test Every Testing Set
    test_path = '1071109_test_1-2-3-4_New12/'
    Now_Time = strftime("%Y%m%d_%H%M_%S", localtime())
    Now_Time = '(' + Now_Time + ')'
    for test_index in range(0, len(test_base)):
        test_base_dir = test_base[test_index]
        path_list = sorted(os.listdir(test_base_dir), key=lambda i: len(i), reverse=False)

        with open(str(Now_Time) + test_base[test_index] + '.csv', 'a') as current_label_writer:
            current_label_writer.write(' ,')
            for label in path_list:
                label = label.replace('-', '_')
                current_label_writer.write(label + ',')
            current_label_writer.write('Accuracy \n')

    # 開始預測手勢
    for model_index in range(0, test_model_nums):

        model_name = model_base[model_index]

        # 限制GPU的用量
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6, allow_growth=True)

        is_Only_using_CPU = False  # TODO 是否禁用GPU (為加速LSTM)
        if is_Only_using_CPU:
            sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, device_count={'GPU': 0}))
            tf.compat.v1.keras.backend.get_session(sess)
        else:
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
            tf.compat.v1.keras.backend.get_session(sess)

        """
          要先set_session() 才能 load_model() -> 不然會有變數無法初始化的問題
        """
        model = load_model(model_name)  # keras.models.load_model()

        cprint(colored('[ {i} / {n} ] : {N}  (model)'.format(i=model_index, n=test_model_nums - 1, N=model_name)),
               'cyan', 'on_grey')

        is_Original_RNN = False

        for test_index in range(0, len(test_base)):

            start = time()

            print(' ' * 4, end='')
            cprint(colored('[ {i} / {n} ] : {N}  (testing set)'.format(i=test_index, n=len(test_base) - 1,
                                                                       N=test_base[test_index])), 'red', 'on_grey',
                   end='\n\n')

            # print('    ', test_base[test_index], end='')
            test_base_dir = test_base[test_index]

            model_name_split_tmp = model_name.split('_')
            model_name_short = '_'.join(model_name_split_tmp[0:5])

            with open(str(Now_Time) + test_base[test_index] + '.csv', 'a') as current_label_writer:
                current_label_writer.write(model_name_short + ',')

            test_set_file_number = 0  # 計算每一個測試集有多少資料(.txt)
            path_list = sorted(os.listdir(test_base_dir), key=lambda i: len(i), reverse=False)  # 按照順序
            for label in path_list:  # label : '1-2', '1-3', ...
                subdir = path.join(test_base_dir, label)  # 'test\test3\1-2', 'test\test3\1-3', ...

                for file in listdir(subdir):  # 'SensorData_2017_11_21_161824.txt', ....
                    if '.txt' not in file:  # 副檔名不是txt -> 不進行處理(非預期裝手勢的檔案)
                        continue
                    file_path = path.join(test_base_dir, label)  # 'test\test3\1-2'
                    file_path = path.join(file_path, file)  # 'test\test3\1-2\SensorData_2017_11_21_161824.txt'
                    # print('Testing on ' + str(file_path))
                    true_label = list(map(int, label.split('-')))  # '[1 , 2]'
                    test_set_file_number += 1
                    test(file_path, model, true_label, test_base, test_index, model_name, is_Original_RNN)

                #
                # for 結束代表同一個手勢組合結束，判斷單一組合辨識率
                #

                current_gesture_set_predict = round(current_label_correct / current_label_count, 2)
                with open(str(Now_Time) + test_base[test_index] + '.csv', 'a') as current_label_writer:
                    current_label_writer.write(str(current_gesture_set_predict) + ',')

                # 換一個Label 就初始化
                current_label_count = 0
                current_label_correct = 0

            Model_Accuracy = (1. * all_correct) / all_count
            Model_Accuracy = round(Model_Accuracy, 4)

            end = time()

            cprint(colored('(Accuracy) : {M}  ==>  Number_of_test : {f}[{d}]'.format(M=Model_Accuracy,
                                                                                     f=test_set_file_number,
                                                                                     d=all_count)), 'yellow')

            with open(str(Now_Time) + test_base[test_index] + '.csv', 'a') as current_label_writer:
                current_label_writer.write(str(Model_Accuracy) + '\n')

            for arr_index in range(0, 12):  # 求各種手勢正確率
                if label_num_arr[arr_index] == 0:
                    Result_arr[arr_index] = 0
                else:
                    Result_arr[arr_index] = round(Correct_num_arr[arr_index] / label_num_arr[arr_index], 3)

            # 用 textable 的方式來印出各種類手勢的辨識狀態 ( *** 正確率以set來計算，沒有考慮到順序 *** )
            table = texttable_print_category_hit_rate(Correct_num_arr, label_num_arr, Result_arr)
            print(table.draw())
            print()
            print('Testing time - {}s'.format(round(float(end - start), 4)), end='\n\n')

            # FIXME 將 Confusion Matrix 輸出到 .csv 中，檔名是 ( .h5名稱 + 測試集名稱 )
            # output_confusion_matrix(test_base, test_index, model_name,
            #                         Confusion_Matrix, label_num_arr, Result_arr,
            #                         Model_Accuracy)

            # 初始化全域變數
            label_num_arr = np.zeros(12)
            Correct_num_arr = np.zeros(12)
            Result_arr = np.zeros(12)
            Confusion_Matrix = np.zeros((12, 12))

            all_correct = 0
            all_count = 0
            test_number_for_predict = 0
            total_file_count = 0
            total_label_count = 0

        print()
        K.clear_session()

    Total_end = time()
    print('Total testing time - ', end='')

    Total_time = Total_end - Total_start
    Total_minute, Total_second = divmod(Total_time, 60)
    Total_hour, Total_minute = divmod(Total_minute, 60)
    cprint(colored('{}h {}m {}s'.format(round(Total_hour), round(Total_minute), round(Total_second), width=2),
                   'red', 'on_grey'))
