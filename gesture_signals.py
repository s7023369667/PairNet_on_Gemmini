import os
import numpy as np
import tensorflow_addons as tfa
from keras.models import load_model
from library import *
import datetime
from library import *
import shutil


def make_Qsiginals_optimal(windows, header_name=None):
    """Quantize gesture signals"""
    # # give general s1, z1
    s1, z1 = 0.9189412337744948, -1
    Q_windows = Quantization(windows, s1, z1)
    # Q_windows = np.array(Q_windows, dtype=np.int32)
    # Q_windows = np.clip(Q_windows - z1, a_min=-128, a_max=127)
    f = open(header_name, "w+")
    f.write(f"//{datetime.datetime.now()}\n")
    f.write("#ifndef QGESTURE_SIGINALS_H\n")
    f.write("#define QGESTURE_SIGINALS_H\n")
    f.write(f"#define BATCH_SIZE {Q_windows.shape[0]}\n")
    f.write(
        f"const elem_t gesture_signals[{Q_windows.shape[0]}][{Q_windows.shape[2]}][{Q_windows.shape[3]}]=")
    f.write("\n{")
    for i in range(Q_windows.shape[0]):
        f.write("{")
        for j in range(Q_windows.shape[1]):
            for k in range(Q_windows.shape[2]):
                f.write("{")
                for l in range(Q_windows.shape[3]):
                    if l != Q_windows.shape[3] - 1:
                        f.write(f"{Q_windows[i][j][k][l]},")
                    else:
                        f.write(f"{Q_windows[i][j][k][l]}")
                if k != Q_windows.shape[2] - 1:
                    f.write("},")
                else:
                    f.write("}")
        if i != Q_windows.shape[0] - 1:
            f.write("},\n")
        else:
            f.write("}\n")
    f.write("};\n")

    f.write("#endif")
    f.close()
    # saveDir = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/include/siginals_Qdataset/'
    # dst = f'{saveDir}{header_name[10:]}'
    # shutil.move(header_name, dst)


def make_all(dir_path):
    """make all Quantized signals"""
    for label in os.listdir(dir_path):
        path = os.path.join(dir_path, label)
        for txt in os.listdir(path):
            print(txt)
            windows = make_window_siginals(os.path.join(path, txt))
            header_name = f'./include/Qgesture_signals_minus/Quantized_{txt.split("_")[0]}_{label}.h'
            make_Qsiginals_optimal(windows, header_name)


if __name__ == '__main__':
    # dir_path = './OapNet/train/train_raw/1071101_Johny[5]&Wen[5]_train_New12(J&W)/'
    # make_all(dir_path)
    path = './OapNet/test/1100920_test_(J&W&D&j&in0)/9-8-4/TD20180927-110100_(Wen)_H50_N3_K9-8-4.txt'
    windows = make_window_siginals(path)
    make_Qsiginals_optimal(windows, './include/Qgesture_signals_984_test.h')
    # make_Qsiginals_optimal(windows, './include/Qgesture_signals_optimal_1.h')
