import tensorflow_addons as tfa
from keras.models import load_model
from Oap.train.Oap_GD_loss import OaP_GD_loss
from library import *
import datetime
from library import *
import shutil


def make_siginals(windows, header_name='./include/gesture_signals.h'):
    f = open(header_name, "w+")
    f.write(f"//{datetime.datetime.now()}\n")
    f.write("#ifndef GESTURE_SIGINALS_H\n")
    f.write("#define GESTURE_SIGINALS_H\n\n")
    f.write(
        f"static const double gesture_signals[{windows.shape[0]}][{windows.shape[1]}][{windows.shape[2]}][{windows.shape[3]}]=")
    f.write("\n{")
    for i in range(windows.shape[0]):
        f.write("{")
        for j in range(windows.shape[1]):
            f.write("{")
            for k in range(windows.shape[2]):
                f.write("{")
                for l in range(windows.shape[3]):
                    if l != windows.shape[3] - 1:
                        f.write(f"{windows[i][j][k][l]},")
                    else:
                        f.write(f"{windows[i][j][k][l]}")
                if k != windows.shape[2] - 1:
                    f.write("},\n")
                else:
                    f.write("}")
            f.write("}")
        if i != windows.shape[0] - 1:
            f.write("},\n")
        else:
            f.write("}\n")
    f.write("};\n")
    f.write("#endif")
    f.close()


def make_Qsiginals(windows, header_name='./include/Qgesture_signals.h'):
    """Quantize gesture signals"""
    best_minn, best_maxx = optimal_MinMax(windows)
    scale, zeroPoint = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
    Q_windows = Quantization(windows, scale, zeroPoint)
    f = open(header_name, "w+")
    f.write(f"//{datetime.datetime.now()}\n")
    f.write("#ifndef QGESTURE_SIGINALS_H\n")
    f.write("#define QGESTURE_SIGINALS_H\n")
    f.write(f'const double scale_signals = {scale};\n')
    f.write(f'const elem_t zeroPoint_signals = {zeroPoint};\n')
    f.write(
        f"static const elem_t gesture_signals[{Q_windows.shape[0]}][{Q_windows.shape[1]}][{Q_windows.shape[2]}][{Q_windows.shape[3]}]=")
    f.write("\n{")
    for i in range(Q_windows.shape[0]):
        f.write("{")
        for j in range(Q_windows.shape[1]):
            f.write("{")
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
            f.write("}")
        if i != Q_windows.shape[0] - 1:
            f.write("},\n")
        else:
            f.write("}\n")
    f.write("};\n")
    f.write("#endif")
    f.close()
    dst = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/include/Qgesture_signals.h'
    shutil.copyfile('./include/Qgesture_signals.h', dst)


if __name__ == '__main__':
    path = 'Oap/test/1100920_test_(J&W&D&j&in0)/3-7-5-11/TD20181001-152951_(Johny)_H50_N4_K3-7-5-11.txt'
    # windows = make_window_siginals(path)
    # make_siginals_headers(windows)
