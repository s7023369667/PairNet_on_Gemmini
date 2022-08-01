import glob
import os
import subprocess as sp
from pycm import *
from library import *
from pairnet_params import *
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_gemmini(main_file='pairNet_ALLQ_main'):
    program_path = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/bareMetalC/'
    build_path = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/'
    run_path = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/'
    riscv_file = main_file + '-pk'
    save_txt = f'{main_file}.txt'
    command = f'spike --extension=gemmini pk {riscv_file} > {save_txt}'
    os.chdir(program_path)
    if not os.path.exists(program_path + main_file + '.c'):
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
    with open(path, 'r') as f:
        line = f.readlines()
    result = []
    for i in range(len(line)):
        if "Predict Result" in line[i]:
            for r in line[i].split('\t')[1:-1]:
                result.append(eval(r))
    return result


def test_gemmini(main_operation, ai_application, dir_path):
    resList = []
    pre_result, gt_result = [], []
    gemmini_dir = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/include/'
    op, app = None, None
    if main_operation == 'conv1d':
        op = 'mc2_conv1d_main'
    elif main_operation == 'conv2matmul':
        op = 'pairNet_ALLQ_main'
    else:
        print('Flag : main_operation should be "conv1d" or "conv2matmul"')
        raise KeyError
    if ai_application == 'pairnet16':
        app = 'Qpairnet_params12_16_optimal.h'
    elif ai_application == 'pairnet32':
        app = 'Qpairnet_params12_32_optimal.h'
    elif ai_application == 'pairnet64':
        app = 'Qpairnet_params12_64_optimal.h'
    else:
        print('Flag : ai_application should be "pairnet16" or "pairnet32" or "pairnet64"')
        raise KeyError
    test_dir = dir_path
    hfiles = os.listdir(os.path.join(gemmini_dir, test_dir))
    for hfile in tqdm(hfiles, desc=test_dir):
        true_label = hfile.split('_')[-1].split('.')[0]
        true_label = get_label(true_label)
        GES_NUM = len(true_label)
        gt_result.append(true_label)
        hfile_path = f'"include/{test_dir}/{hfile}"'
        top_hfile_path = os.path.join(gemmini_dir, 'top_hfile.h')
        with open(os.path.join(top_hfile_path), 'w') as file:
            file.write(f"//{datetime.datetime.now()}\n")
            file.write(f"#ifndef GEMMINI_PROJECTS_TOP_HFILE_H\n")
            file.write(f"#define GEMMINI_PROJECTS_TOP_HFILE_H\n")
            file.write(f'#include "include/{app}"\n')
            file.write(f"#include {hfile_path}\n")
            file.write(f"#define GES_NUM {GES_NUM}\n")
            file.write(f"#endif //GEMMINI_PROJECTS_TOP_HFILE_H\n")
        pre = run_gemmini(main_file=op)
        for i in range(len(true_label)):
            gt_result.append(true_label[i])
            pre_result.append(pre[i])
        print('True Label:', true_label)
        print('Predict Label:', pre)
        print()

    false_cnt = 0

    for j in range(len(gt_result)):
        if pre_result[j] != gt_result[j]:
            false_cnt += 1
    acc = (len(gt_result) - false_cnt) / len(gt_result)
    print(f"accuracy : {acc}")
    resList.append(acc)
    cm = ConfusionMatrix(actual_vector=gt_result, predict_vector=pre_result)
    cm.print_matrix()
    # cm.plot(cmap=plt.cm.Greens, number_label=True, plot_lib="matplotlib")
    # plt.show()
    # cm.print_normalized_matrix()
    plt.figure()
    cm.plot(cmap=plt.cm.Greens, number_label=True, plot_lib="matplotlib", normalized=True)
    plt.savefig(f'./{ai_application}_{acc}.png')
    # plt.show()
    return acc


def get_clean_txt(test_dir):
    clean_txt, clean_true_lable = [], []
    for file_label in os.listdir(test_dir):
        txt_label = get_label(file_label)
        for txt in glob.glob(test_dir + file_label + "/*.txt"):
            clean_txt.append(txt)
            clean_true_lable.append(txt_label)

    return clean_txt, clean_true_lable


def get_label(file_label):
    if '-' in file_label:
        txt_label = list(map(int, file_label.split('-')))  # 資料夾名稱1-2-5 >> [1, 2, 5]
    else:
        txt_label = [int(file_label)]
    return txt_label


def main():
    # path = 'OapNet/test/1100920_test_(J&W&D&j&in0)/9-8-4/TD20180927-110149_(Wen)_H50_N3_K9-8-4.txt' path =
    # path = './OapNet/train/train_raw/1071101_Johny[5]&Wen[5]_train_New12(J&W)/6/D20180919-173126_(Wen)_H50_N1_K6.txt'
    # model_path = f"PairNet/model/pairnet_model16_12_20220503.h5"
    # true_lable = [6]
    # # true_lable = list(map(int, (path.split('/')[-2]).split('-')))
    # windows = make_window_siginals(path)
    # """feed into mc2_conv1d_main.c & pairNet_ALLQ_main.c"""
    # make_Qsiginals(windows, header_name=f'./include/Qgesture_signals_{"".join([str(i) for i in true_lable])}.h')
    # make_Qpairnet_params(batch_size=windows.shape[0], input_width=50, stride_size=1,
    #                      input_signals=windows, gesN=gesN, path=model_path, true_label=true_lable,
    #                      header_name=f'./include/Qpairnet_params{gesN}_{channel}_{"".join([str(i) for i in true_lable])}.h')
    test_dir = 'Qgesature_signals_test'
    train_dir = 'Qgesature_signals'
    acc = []
    # acc_16 = test_gemmini(main_operation='conv2matmul', ai_application='pairnet16', dir_path=test_dir)
    acc_32 = test_gemmini(main_operation='conv2matmul', ai_application='pairnet32', dir_path=test_dir)
    acc_64 = test_gemmini(main_operation='conv2matmul', ai_application='pairnet64', dir_path=test_dir)
    # acc.append(acc_16)
    # accuracy : 0.8018018018018018
    acc.append(acc_32)
    # accuracy : 0.7815924032140248
    acc.append(acc_64)
    print(acc)


if __name__ == '__main__':
    main()
