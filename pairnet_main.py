from gesture_signals import *
from PairNet_params import *
from Oap_params import make_oapNetALLQ_params
from library import *
import subprocess as sp
from pycm import *
import os, glob
import matplotlib.pyplot as plt


def run_gemmini(main_file='Qpairnet_gemmini'):
    program_path = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/bareMetalC/'
    build_path = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/'
    run_path = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/'
    riscv_file = main_file + '-baremetal'
    save_txt = f'{main_file}.txt'
    command = f'spike --extension=gemmini {riscv_file} > {save_txt}'
    os.chdir(program_path)
    if not os.path.exists(program_path + main_file + '.c'):
        print("Cannot find the main file...")
    os.chdir(build_path)
    if os.path.exists(build_path + 'build'):
        build = sp.Popen("source /home/sam/chipyard/env.sh  && rm -r build  && ./build.sh",
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
    os.chdir('/home/sam/CLionProjects/gemmini_projects/')
    with open(path, 'r') as f:
        line = f.readlines()
        # print(line)
    res = line[4].split()
    result = []
    for i in range(2, len(res)):
        result.append(eval(res[i]))
    return result


def test(test_dir, gesN, save_path):
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
            try:
                for txt in glob.glob(test_dir + file_label + "/*.txt"):
                    print(txt)
                    windows = make_window_siginals(txt)
                    make_Qsiginals(windows)
                    make_pairNetALLQ_params(batch_size=windows.shape[0], input_width=50, stride_size=1,
                                            input_signals=windows, gesN=gesN, path=model_path, true_label=txt_label,
                                            len_label=len(txt_label))
                    pre = run_gemmini()
                    print(pre)
                    print(txt_label)
                    for i in range(len(pre)):
                        if pre[i] == -1:
                            break
                        else:
                            pre_result.append(pre[i])
                            gt_result.append(txt_label[i])
            except ValueError:
                print("error")
                pass
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
    path = 'Oap/test/1100920_test_(J&W&D&j&in0)/9-8-4/TD20180927-110149_(Wen)_H50_N3_K9-8-4.txt'
    # path = 'Oap/test/1100920_test_(J&W&D&j&in0)/2-1-6-5/TD20181001-233625_(Wen)_H50_N4_K2-1-6-5.txt'
    # path = 'Oap/test/1100920_test_(J&W&D&j&in0)/6-5/TD20180921-214456_(Wen)_H50_N2_K6-5.txt'
    gesN = 12
    model_path = "PairNet/model/pairnet_model16_12_20220216.h5"
    windows = make_window_siginals(path)
    make_Qsiginals(windows)
    make_pairNetALLQ_params(batch_size=windows.shape[0], input_width=50, stride_size=1,
                            input_signals=windows, gesN=gesN, path=model_path, true_label=[9, 8, 4], len_label=3)
    # test('PairNet/1071109_test_1-2-3-4_New12_test/', gesN, f'result_cfm_quantized{gesN}.png')