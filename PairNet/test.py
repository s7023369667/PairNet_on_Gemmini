import os
from texttable import Texttable
from termcolor import colored, cprint
import numpy as np
from time import *
import datetime
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import subprocess as sp
from tqdm import trange
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


def sequential_filtered_result(filtered_result, label_len, isDuplicated):
    def append_index_and_value(pre_number, Now_count):
        tmp = [pre_number, Now_count]
        return tmp

    count_result_list = []
    pre_number = -1
    try:
        pre_number = filtered_result[0]
    except IndexError:
        print('[filtered_result_sequential] Out of range - ', len(filtered_result))
    now_count = 0

    for index in np.arange(len(filtered_result)):
        if pre_number == filtered_result[index]:
            now_count += 1  # 與前面的數字相同，count + 1
        else:
            # 把 [ pre_number(type : str), Now_count(type : int) ] append count_result_list
            count_result_list.append(append_index_and_value(pre_number, now_count))
            now_count = 1  # 重新起算，現在就有一個

        pre_number = filtered_result[index]
    # 最後一個種類要額外寫進去
    count_result_list.append(append_index_and_value(pre_number, now_count))

    # count_result_list -> [['1', 52], ['5', 2], ['6', 1], ['1', 9], ['9', 2], ['1', 9], ['9', 12]]
    sort_tmp_list = []
    for index, value in count_result_list:
        sort_tmp_list.append(value)  # 把數量的部分單獨拿出來，作為sort指標
    '''
        count list - [67  2  1  1  6  3 20]
        sort index - [0 6 4 5 1 2 3]
        result_list_index - [0 6 4] ( if label_len = 3 )
        result_list_index.sort() - [0 4 6]
    '''
    sort_tmp_array = np.array(sort_tmp_list)  # list -> np.array()
    sort_tmp_index = np.argsort(-sort_tmp_array)  # 讓 index 呈降序排列
    if not isDuplicated:
        #  原本的做法是直接擷取 label 數量的前幾個index，但會造成出現兩個以上重複手勢
        #  目前假定手勢 "不會重覆"，所以原本的算法會低估手勢預測的結果
        count_result_sorted_list = []
        count_result_sorted_index_list = []  # 紀錄label, 其中不重複
        for idx in sort_tmp_index:  # 以降冪排列的順序，剃除重複的手勢種類
            tmp_label_list = [count_result_sorted_list[i][0] for i in range(len(count_result_sorted_list))]
            if count_result_list[idx][0] not in tmp_label_list:  # [0] 代表手勢種類
                count_result_sorted_list.append(count_result_list[idx])
                count_result_sorted_index_list.append(idx)

        count_result_sorted_index_list = count_result_sorted_index_list[0:label_len]
        count_result_sorted_index_list.sort()

        return_tmp_list_tmp = []
        for i in count_result_sorted_index_list:
            return_tmp_list_tmp.append(count_result_list[i][0])
        return return_tmp_list_tmp

    else:
        # 根據排序好的 index，回傳對應的手勢
        result_list_index = sort_tmp_index[0:label_len]
        result_list_index.sort()  # index

        return_predict_gesture_list = []
        for i in result_list_index:
            return_predict_gesture_list.append(count_result_list[i][0])
        # print(return_predict_gesture_list)0.

        return return_predict_gesture_list


def texttable_print_category_hit_rate(Correct_num_arr, label_num_arr, Result_arr):
    # 為保留手勢0能被當作一種手勢的可能性，所以矩陣大小設為 12
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


def cal_precision_recall(cm:np.ndarray):
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            if i > 4 and j > 4 and i == j:
                tp += cm[i][j]
            elif i <= 4 and j > 4 and i != j:
                if cm[i][j] != 0:
                    fp += cm[i][j]
            elif i > 4 and j <= 4 and i != j:
                if cm[i][j] != 0:
                    fn += cm[i][j]
            elif i <= 4 and j <= 4 and i == j:
                tn += cm[i][j]
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return accuracy, precision, recall


def tf_test(f):
    global test_number_for_predict
    sample = []
    dim = 6
    split_length = 50
    with open(f) as test_case:
        for data_line in test_case:
            t = data_line.strip().split(' ')
            sample.append(list(map(float, t)))
    sample = sample[:-50]  # sample最後50 + 1筆也要扣掉
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
    return raw_result_sliding


def test(f, label, true_label, dis_gesN):
    global all_count, all_correct, test_number_for_predict
    global label_num_arr, Correct_num_arr, Result_arr
    global current_label_count, current_label_correct, total_file_count, total_label_count
    global User_false_count

    """tensorflow test"""
    # raw_result_sliding = tf_test(f)
    """using gemmini to test"""
    raw_result_sliding = gemmini_test(label, file, ai_application='pairnet128')

    filtered_result_sliding_version = []
    for i in raw_result_sliding:
        label_tmp = i.argmax()  # (12,) -> inttf.compat.v1.
        filtered_result_sliding_version.append(label_tmp)
    guess = sequential_filtered_result(filtered_result_sliding_version, len(true_label),
                                       isDuplicated=False)  # 防止出現長度無法對齊

    """
      描述少了哪些手勢
    """
    if len(guess) != len(true_label):
        cprint('{}File Name : {}'.format(' ' * 4, f), color='red')
        # print('{}{}'.format(' ' * 4, filtered_result_sliding_version))
        cprint('{}True Label: {} ; Predict: {} -> '.format(' ' * 6, true_label, guess), end='', color='red')
        guess = guess + [guess[-1]] * (len(true_label) - len(guess))  # 會出差超過兩個的情況
        cprint(colored(guess, 'white', 'on_grey'), end='\n\n')

    cprint(text=f'True Label: {true_label}', color='blue')
    cprint(text=f'Predict   : {guess}', color='blue')

    """   
        輸出最多的 len(label) 個元素  -> len(label) 代表有幾個手勢
        Counter(filtered_result) -> Counter({2: 72, 1: 65, 7: 17, 3: 6, 4: 1, 5: 1})
        guess  -> [2, 1]
        lambda -> 來定義函式
               -> x : [(2, 72), (1, 65)]
               -> x[0] : [2, 1]
    """
    all_count += len(true_label)
    current_label_count += len(true_label)
    for i in true_label:
        label_num_arr[i] += 1  # 計算每種手勢出現次數(標籤)
    """
        以 set 的觀點， {1, 2, 1} == {1, 1, 2}
        但若有順序情況下會不同
        計算Hit rate時仍就沒有順序之分
    """
    comapre_Result = set(guess).intersection(true_label)

    '''
        這裡並未把辨識結果的「順序」拿來討論，只要有被辨識出來就算
        因此，先把 實際 與 辨識結果 都有的部分(交集)拿出來
        剩下來的就是沒能辨識正確的 -> 額外放進 Confusion Matrix
    '''
    tmp_label = true_label.copy()
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
        if correct > dis_gesN:
            Correct_num_arr[correct] += 1  # 計算有被辨識出來的手勢


def gemmini_test(sub_dir, txtfile, ai_application, main_operation=None):
    train_dir = "Qgesture_signals_training"
    test_dir = "Qgesture_signals_testing"
    hfile = txtfile[:-4] + '.h'
    make_top_hfile(test_dir, sub_dir, hfile, ai_application)
    op = 'pairNet_ALLQ_main'
    if main_operation == 'conv1d':
        op = 'mc2_conv1d_main'
    program_path = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/bareMetalC/'
    build_path = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/'
    run_path = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/'
    riscv_file = op + '-pk'
    save_txt = f'{op}.txt'
    command = f'spike --extension=gemmini pk {riscv_file} > {save_txt}'
    os.chdir(program_path)
    if not os.path.exists(program_path + op + '.c'):
        print("Cannot find the main file...")
    os.chdir(build_path)
    if os.path.exists(os.path.join(build_path, 'build')):
        build = sp.Popen("source /home/sam/chipyard/env.sh  && sudo rm -r build && ./build.sh",
                         shell=True, executable="/bin/bash", stdout=sp.PIPE).stdout.read()
    else:
        build = sp.Popen("source /home/sam/chipyard/env.sh  &&  ./build.sh",
                         shell=True, executable="/bin/bash", stdout=sp.PIPE).stdout.read()
    # for line in build.decode().split('\n'):
    #     print(line)
    os.chdir(run_path)
    sp.Popen(f"source /home/sam/chipyard/env.sh && {command}", shell=True, executable="/bin/bash",
             stdout=sp.PIPE).stdout.read()
    path = f'/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/{save_txt}'
    out_result = []
    with open(path, 'r') as f:
        lines = f.readlines()
    s, e = 0, len(lines)
    for i in range(len(lines)):
        if "Dense out" in lines[i]:
            s = i + 1
        elif "SUCCESS" in lines[i]:
            e = i - 1
    lines = lines[s:e]
    for line in lines:
        out_result.append(list(map(int, line.split())))
    return np.array(out_result)


def make_top_hfile(test_dir, sub_dir, hfile, ai_application):
    if ai_application == 'pairnet16':
        app = 'Qpairnet_params12_16_optimal.h'
    elif ai_application == 'pairnet32':
        app = 'Qpairnet_params12_32_optimal.h'
    elif ai_application == 'pairnet64':
        app = 'Qpairnet_params12_64_optimal.h'
    elif ai_application == 'pairnet128':
        app = 'Qpairnet_params12_128_optimal.h'
    else:
        print('Flag : ai_application should be "pairnet16" or "pairnet32" or "pairnet64"')
        raise KeyError
    gemmini_dir = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/include/'
    hfile_path = f'"include/{test_dir}/{sub_dir}/{hfile}"'
    top_hfile_path = os.path.join(gemmini_dir, 'top_hfile.h')
    with open(os.path.join(top_hfile_path), 'w') as file:
        file.write(f"//{datetime.datetime.now()}\n")
        file.write(f"#ifndef GEMMINI_PROJECTS_TOP_HFILE_H\n")
        file.write(f"#define GEMMINI_PROJECTS_TOP_HFILE_H\n")
        file.write(f'#include "include/{app}"\n')
        file.write(f"#include {hfile_path}\n")
        # file.write(f"#define GES_NUM {GES_NUM}\n")
        file.write(f"#endif //GEMMINI_PROJECTS_TOP_HFILE_H\n")


if __name__ == '__main__':
    # filter background gestures
    dis_geN = 4
    # 用來計算辨識結果，來取最後的正確率
    all_count = 0
    all_correct = 0
    # 在訓練時最後出來結果是　1 x 12, 所以實際上還是有可能會被辨識出 手勢"0"
    # 但目前做法會是 (Label - 1)，所以假設有0的會對應到最後一個(也就是手勢11)
    label_num_arr = np.array(np.zeros(12))
    Correct_num_arr = np.array(np.zeros(12))
    Result_arr = np.array(np.zeros(12))
    test_number_for_predict = 0  # 計算每一個測試集有多少資料中(以一行為單位)

    confusion_true_label = []
    confusion_predict_label = []

    Confusion_Matrix = np.array(np.zeros((12, 12)))

    current_label_count = 0
    current_label_correct = 0
    total_file_count = 0
    total_label_count = 0

    Total_start = time()

    # Test Every Testing Set
    test_path = '/home/sam/CLionProjects/gemmini_projects/PairNet/1071109_test_1-2-3-4_New12_test'
    # train_path = '/home/sam/CLionProjects/gemmini_projects/OapNet/train/train_raw/1071101_Johny[5]&Wen[5]_train_New12(J&W)'
    test_base = [test_path]
    Now_Time = '(' + strftime("%Y%m%d_%H%M_%S", localtime()) + ')'
    for test_index in range(0, len(test_base)):
        test_base_dir = test_base[test_index]
        path_list = sorted(os.listdir(test_base_dir), key=lambda i: len(i), reverse=False)

        with open(test_base[test_index] + f'{str(Now_Time)}.csv', 'a') as current_label_writer:
            current_label_writer.write(' ,')
            for label in path_list:
                label = label.replace('-', '_')
                current_label_writer.write(label + ',')
            current_label_writer.write('Accuracy \n')

    model_name = './model/pairnet_model128_12_20220823.h5'
    model = load_model(model_name)  # keras.models.load_model()
    for test_index in range(0, len(test_base)):
        start = time()
        test_base_dir = test_base[test_index]

        test_set_file_number = 0  # 計算每一個測試集有多少資料(.txt)
        path_list = sorted(os.listdir(test_base_dir), key=lambda i: len(i), reverse=False)  # 按照順序
        for i in trange(len(path_list)):  # label : '1-2', '1-3', ...
            sub_dir = os.path.join(test_base_dir, path_list[i])  # 'test\test3\1-2', 'test\test3\1-3', ...
            print()
            print(sub_dir)
            for file in os.listdir(sub_dir):  # 'SensorData_2017_11_21_161824.txt', ....
                if '.txt' not in file:  # 副檔名不是txt -> 不進行處理(非預期裝手勢的檔案)
                    continue
                file_path = os.path.join(test_base_dir, path_list[i])  # 'test\test3\1-2'
                file_path = os.path.join(file_path, file)  # 'test\test3\1-2\SensorData_2017_11_21_161824.txt'
                # print('Testing on ' + str(file_path))
                true_label = list(map(int, path_list[i].split('-')))  # '[1 , 2]'
                test_set_file_number += 1
                test(file_path, path_list[i], true_label, dis_geN)
            # for 結束代表同一個手勢組合結束，判斷單一組合辨識率

            current_gesture_set_predict = round(current_label_correct / current_label_count, 2)
            with open(test_base[test_index] + f'{str(Now_Time)}.csv', 'a') as current_label_writer:
                current_label_writer.write(str(current_gesture_set_predict) + ',')

            # 換一個Label 就初始化
            current_label_count = 0
            current_label_correct = 0

        all_count = 0
        all_correct = 0
        for arr_index in range(0, 12):  # 求各種手勢正確率
            if label_num_arr[arr_index] == 0:
                Result_arr[arr_index] = 0
            else:
                Result_arr[arr_index] = round(Correct_num_arr[arr_index] / label_num_arr[arr_index], 3)
                if arr_index > dis_geN:
                    all_correct += Correct_num_arr[arr_index]
                    all_count += label_num_arr[arr_index]

        Model_Accuracy = (1. * all_correct) / all_count
        Model_Accuracy = round(Model_Accuracy, 4)

        end = time()
        cprint(colored('(Accuracy) : {M}  ==>  Number_of_test : {f}[{d}]'.format(M=Model_Accuracy,
                                                                                 f=test_set_file_number,
                                                                                 d=all_count)), 'yellow')

        with open(test_base[test_index] + f'{str(Now_Time)}.csv', 'a') as current_label_writer:
            current_label_writer.write(str(Model_Accuracy) + '\n')

        # 用 textable 的方式來印出各種類手勢的辨識狀態 ( *** 正確率以set來計算，沒有考慮到順序 *** )
        table = texttable_print_category_hit_rate(Correct_num_arr, label_num_arr, Result_arr)
        print(table.draw())
        print()
        print('Testing time - {}s'.format(round(float(end - start), 4)), end='\n\n')

        # 將 Confusion Matrix 輸出到 .csv 中，檔名是 ( .h5名稱 + 測試集名稱 )
        # output_confusion_matrix(test_base, test_index, model_name,
        #                         Confusion_Matrix, label_num_arr, Result_arr,
        #                         Model_Accuracy)

        # 初始化全域變數
        label_num_arr = np.zeros(12)
        Correct_num_arr = np.zeros(12)
        Result_arr = np.zeros(12)
        # Confusion_Matrix = np.zeros((12, 12))

        all_correct = 0
        all_count = 0
        test_number_for_predict = 0
        total_file_count = 0
        total_label_count = 0

    print()
    K.clear_session()

    Total_end = time()
    print('Total testing time - ', end='')

    # y_actu = pd.Series(confusion_true_label, name='Actual')
    # y_pred = pd.Series(confusion_predict_label, name='Predicted')
    # df_confusion = pd.crosstab(y_actu, y_pred, dropna=False)
    # print(df_confusion)

    Total_time = Total_end - Total_start
    Total_minute, Total_second = divmod(Total_time, 60)
    Total_hour, Total_minute = divmod(Total_minute, 60)
    cprint(colored('{}h {}m {}s'.format(round(Total_hour), round(Total_minute), round(Total_second), width=2),
                   'red', 'on_grey'))

    # delete_index = []
    # for i in range(len(confusion_true_label)):
    #     if confusion_true_label[i] <= dis_geN or confusion_predict_label[i] <= dis_geN:
    #         delete_index.append(i)
    #
    # idx = 0
    # for index in delete_index:
    #     confusion_true_label.pop(index-idx)
    #     confusion_predict_label.pop(index-idx)
    #     idx += 1

    print(Confusion_Matrix)
    accuracy, precision, recall = cal_precision_recall(Confusion_Matrix)
    print("acc:", accuracy)
    print("precision", precision)
    print("recall", recall)