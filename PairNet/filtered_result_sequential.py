import numpy as np
from collections import Counter


def append_index_and_value(pre_number, Now_count):
    tmp = []
    tmp.append(pre_number)
    tmp.append(Now_count)
    # print(tmp)
    return tmp


def sequential_filtered_result(filtered_result, label_len, isDuplicated):
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

    if __name__ == '__main__':
        print(count_result_list)

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

    if isDuplicated == False:
        #
        #  原本的做法是直接擷取 label 數量的前幾個index，但會造成出現兩個以上重複手勢
        #  目前假定手勢 "不會重覆"，所以原本的算法會低估手勢預測的結果
        #
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


if __name__ == '__main__':
    file_name = ['SensorData_2017_11_24_110501filtered_result.txt',
                 'SensorData_2017_11_24_110505filtered_result.txt',
                 'SensorData_2017_11_24_110509filtered_result.txt',
                 'SensorData_2017_11_24_110513filtered_result.txt',
                 'SensorData_2017_11_24_110516filtered_result.txt',
                 'SensorData_2017_11_24_110520filtered_result.txt',
                 'Xt_Data_2018_07_30-16_48_54filtered_result.txt',
                 'Xt_Data_2018_07_30-16_49_02filtered_result.txt',
                 'Xt_Data_2018_07_30-16_49_13filtered_result.txt',
                 'Xt_Data_2018_07_30-16_49_23filtered_result.txt',
                 'Xt_Data_2018_07_30-16_49_30filtered_result.txt']

    for file in file_name:
        with open(file, "r") as infp:
            print(file, end='')
            data_list = infp.readline()  # <class 'str'>
            data_list = data_list[1:-1]  # 去掉 '[' ']'

        filtered_result = []
        for num in data_list.split(','):
            num = num.strip()  # 去掉多餘的空白
            filtered_result.append(num)

        print(' (len - {})'.format(len(filtered_result)))

        label_len = 2
        return_list = sequential_filtered_result(filtered_result, label_len)
        print('sequential - ', return_list)
        guess = list(map(lambda x: x[0], Counter(filtered_result).most_common(label_len)))
        print('most_common - ', guess)
        print()
