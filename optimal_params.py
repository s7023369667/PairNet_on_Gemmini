import datetime
import os
from tensorflow.keras.models import load_model
from tqdm._tqdm import trange
from library import *
from library import reshape_kernel


def get_rawData(dir_path, cnt, gesN):
    total_windows = []
    c = 0
    flag = False
    for label in os.listdir(dir_path):
        if eval(label) <= gesN:
            path = os.path.join(dir_path, label)
            for txt in os.listdir(path):
                print(txt)
                if c < cnt:
                    windows = make_window_siginals(os.path.join(path, txt))
                    total_windows.append(np.array(windows))
                    c += 1
                else:
                    flag = True
                    break
        if flag:
            break
    total_windows = np.array(total_windows)
    return total_windows


def get_sAndz(fp_feature):
    best_minn, best_maxx = optimal_MinMax(fp_feature)
    s, z = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
    return s, z


def get_layer_factor(window_size, model_path, total_features):
    S1, Z1 = [[] * i for i in range(6)], [[] * i for i in range(6)]
    S4, Z4 = [[] * i for i in range(6)], [[] * i for i in range(6)]
    model = load_model(model_path)
    for f in trange(len(total_features)):
        s1, z1 = get_sAndz(total_features[f])
        layer_cnt = 0
        feature = total_features[f]
        input_width = window_size
        stride = 1
        for layer in model.layers:
            # print(layer.name)
            if layer.get_weights():
                if 'conv1d' in layer.name:
                    conv = np.array(layer.get_weights())
                elif 'batch_normalization' in layer.name:
                    BN = np.array(layer.get_weights())
                    """configures"""
                    filters = conv.shape[1]
                    out_channels = conv.shape[3]
                    kernel_size = filters
                    do_padding = False
                    padding_front, padding_back, padding_size = 0, 0, 0
                    if do_padding:
                        padding_back = 1
                        padding_front = 1 if kernel_size % 2 != 0 else 0
                    padding_size = padding_back + padding_front
                    output_width = (input_width - kernel_size + padding_size) // stride + 1
                    """ConBN Bias"""
                    convBN, bias = Folding_Conv_BN(conv, BN)
                    convBN_bias = np.repeat([bias], output_width, axis=0)
                    convBN_bias = np.repeat([convBN_bias], feature.shape[0], axis=0)
                    convBN_bias = convBN_bias.reshape([feature.shape[0], 1, output_width, out_channels])
                    convBN_biasCorr = bias_Correction(convBN_bias)
                    """Calculate Conv1d"""
                    result = tf.nn.conv1d(feature, convBN[0], stride=stride, padding='VALID') + convBN_biasCorr
                    s3, z3 = get_sAndz(result)
                    result = tf.clip_by_value(result, clip_value_min=0,
                                              clip_value_max=np.max(result))
                    s4, z4 = cal_scaleZeroPoint(r_max=np.max(result), r_min=0, q_max=127, q_min=z3)
                    feature = result
                    S1[layer_cnt].append(s1), S4[layer_cnt].append(s4)
                    Z1[layer_cnt].append(z1), Z4[layer_cnt].append(z4)
                    input_width = output_width
                    stride = 2
                    layer_cnt += 1
                    s1, z1 = s4, z4
                elif 'dense' in layer.name:
                    # Dense
                    dense, dense_bias = np.array(layer.get_weights()[0]), np.array(layer.get_weights()[1])
                    # dense_biasCorr = bias_Correction(dense_bias)
                    result = np.matmul(feature, dense) + dense_bias
                    feature = result
                    s3, z3 = get_sAndz(feature)
                    S1[layer_cnt].append(s1), S4[layer_cnt].append(s3)
                    Z1[layer_cnt].append(z1), Z4[layer_cnt].append(z3)
                elif 'global_average_pooling1d' in layer.name:
                    # GAP
                    gap_result = tf.reshape(feature, [total_features[f].shape[0], 3, out_channels])
                    gap_result = tf.keras.layers.GlobalAveragePooling1D()(gap_result)
                    feature = gap_result

    S1, Z1, S4, Z4 = np.array(S1), np.array(Z1), np.array(S4), np.array(Z4)
    return S1, Z1, S4, Z4


def make_header(gesN, window_size, model_path, S1, Z1, S4, Z4, header_name=None):
    input_width, stride = window_size, 1
    file = open(header_name, "w+")
    file.write(f"//{datetime.datetime.now()}\n")
    file.write(f"#ifndef QPAIR_NET_PARAMS_H\n")
    file.write("#define QPAIR_NET_PARAMS_H\n\n")
    file.write("#include <stdint.h>\n")
    file.write("#include <stdbool.h>\n\n")
    file.write(f"#define GESN {gesN}\n")
    file.write("struct Conv1d_Params {\n\tint input_width;\n\tint in_channels;\n\t"
               "int out_channels;\n\tint kernel_size;\n\tint stride_size;\n\t""int padding_front;\n\t"
               "int padding_back;\n\tint output_width;\n\tdouble out_scale;\n\tdouble s1;\n\tdouble s2;\n\t"
               "double sb;\n\tdouble s4;\n\tint z1;\n\tint z2;\n\tint zb;\n\tint z4;\n\t};\n")
    file.write("struct Dense_Params {\n\t//I=BATCH_SIZE;\n\tint K;\n\tint J;\n\tdouble out_scale;\n\tdouble "
               "s1;\n\tdouble s2;\n\tdouble sb;\n\tdouble s3;\n\tint z1;\n\tint z2;\n\tint zb;\n\tint z3;\n\t};\n")
    layer_cnt = 0
    model = load_model(model_path)
    for layer in model.layers:
        print(layer.name)
        print(np.array(layer.get_weights()).shape)
        if layer.get_weights():
            if 'conv1d' in layer.name:
                conv = np.array(layer.get_weights())
            elif 'batch_normalization' in layer.name:
                BN = np.array(layer.get_weights())
                convBN, bias = Folding_Conv_BN(conv, BN)
                convBN_biasCorr = bias_Correction(bias)
                convBN_biasCorr = np.squeeze(convBN_biasCorr)
                wrapper = convBN.shape[0]
                filters = convBN.shape[1]
                in_channels = convBN.shape[2]
                out_channels = convBN.shape[3]
                kernel_size = filters
                do_padding = False
                padding_front, padding_back, padding_size = 0, 0, 0
                if do_padding:
                    padding_back = 1
                    padding_front = 1 if kernel_size % 2 != 0 else 0
                padding_size = padding_back + padding_front
                output_width = (input_width - kernel_size + padding_size) // stride + 1
                s2, z2 = get_sAndz(convBN)
                sb, zb = get_sAndz(convBN_biasCorr)
                s1, z1 = np.mean(S1[layer_cnt]), int(np.mean(Z1[layer_cnt]))
                s4, z4 = np.mean(S4[layer_cnt]), int(np.mean(Z4[layer_cnt]))
                """write QConv_BN"""
                QConv_BN = Quantization(convBN, s2, z2)
                file.write(
                    f"static const elem_t QConv_BN{layer.name[-1]}[{filters}][{in_channels}][{out_channels}]=\n")
                file.write('{')
                for i in range(wrapper):
                    for j in range(filters):
                        file.write('{')
                        for k in range(in_channels):
                            file.write('{')
                            for l in range(out_channels):
                                if l != (out_channels - 1):
                                    file.write(f"{QConv_BN[i][j][k][l]},")
                                else:
                                    file.write(f"{QConv_BN[i][j][k][l]}")
                            if k != (in_channels - 1):
                                file.write('},')
                            else:
                                file.write('}')
                        if j != (filters - 1):
                            file.write('},')
                        else:
                            file.write('}')
                file.write('};\n')
                """write QConv_BN_mc2"""
                QConv_BN_mc2 = np.array(QConv_BN, dtype=np.int32)
                # # pre-processing weight quantization
                QConv_BN_mc2 = np.clip(QConv_BN_mc2 - z2, a_min=-128, a_max=127)
                QConv_BN_mc2 = reshape_kernel(QConv_BN_mc2)
                file.write(
                    f"static const elem_t QConv_BN_mc2_{layer.name[-1]}[{kernel_size * in_channels}][{out_channels}]=\n")
                file.write('{')
                for i in range(QConv_BN_mc2.shape[0]):
                    file.write('{')
                    for j in range(QConv_BN_mc2.shape[1]):
                        if not j == (QConv_BN_mc2.shape[1] - 1):
                            file.write(f'{QConv_BN_mc2[i][j]},')
                        else:
                            file.write(f'{QConv_BN_mc2[i][j]}')
                    if not i == (QConv_BN_mc2.shape[0] - 1):
                        file.write('},')
                    else:
                        file.write('}')
                file.write('};\n')
                """write bias """
                QConv_BN_bias = Quantization(convBN_biasCorr, sb, zb)
                QConv_BN_bias = ((sb / (s1 * s2)) * (QConv_BN_bias - zb)) + (z4 / ((s1 * s2) / s4))
                QConv_BN_bias = np.round(QConv_BN_bias)
                QConv_BN_bias = np.array(QConv_BN_bias, dtype=np.int32)
                file.write(
                    f"static const acc_t QConv_BN_bias{layer.name[-1]}[{out_channels}] = ")
                file.write("\n{")
                for l in range(out_channels):
                    if l != out_channels - 1:
                        file.write(f"{QConv_BN_bias[l]},")
                    else:
                        file.write(f"{QConv_BN_bias[l]}")
                file.write("};\n")

                file.write(f'const struct Conv1d_Params QConv_BN{layer.name[-1]}_params = {{'
                           f'.input_width={input_width}, .in_channels={in_channels},'f'.out_channels = {out_channels},'
                           f'.kernel_size ={kernel_size},.stride_size={stride},'f'.padding_front= {padding_front},'
                           f'.padding_back= {padding_back},.output_width={output_width}'f',.out_scale={(s1 * s2) / s4},'
                           f'.s1={s1},.z1={z1},.s2={s2},.z2={z2},.sb={sb},.zb={zb},.s4={s4},'f'.z4={z4}}};\n')
                input_width = output_width
                stride = 2
                layer_cnt += 1
            elif 'dense' in layer.name:
                dense, dense_bias = np.array(layer.get_weights()[0]), np.array(layer.get_weights()[1])
                s2, z2 = get_sAndz(dense)
                sb, zb = get_sAndz(dense_bias)
                dense_biasCorr = bias_Correction(dense_bias)
                """write QDense_params"""
                QDense = Quantization(dense, s2, z2)
                file.write(f"static const elem_t QDense_params[{QDense.shape[0]}][{QDense.shape[1]}] = \n")
                file.write('{')
                for i in range(QDense.shape[0]):
                    file.write('{')
                    for j in range(QDense.shape[1]):
                        if not j == (QDense.shape[1] - 1):
                            file.write(f'{QDense[i][j]},')
                        else:
                            file.write(f'{QDense[i][j]}')
                    if not i == (QDense.shape[0] - 1):
                        file.write('},')
                    else:
                        file.write('}')
                file.write('};\n')
                """write QDense_bias"""
                QDense_bias = Quantization(dense_biasCorr, sb, zb)
                s1, z1 = np.mean(S1[layer_cnt]), int(np.mean(Z1[layer_cnt]))
                s4, z4 = np.mean(S4[layer_cnt]), int(np.mean(Z4[layer_cnt]))
                QDense_bias = ((sb / (s1 * s2)) * (QDense_bias - zb)) + (z4 / ((s1 * s2) / s4))
                QDense_bias = np.round(QDense_bias)
                QDense_bias = np.array(QDense_bias, dtype=np.int32)
                file.write(f"static const acc_t QDense_bias[{QDense_bias.shape[0]}] = \n")
                file.write('{')
                for i in range(QDense_bias.shape[0]):
                    if not i == (QDense_bias.shape[0] - 1):
                        file.write(f'{QDense_bias[i]},')
                    else:
                        file.write(f'{QDense_bias[i]}')
                file.write('};\n')
                file.write(f'const struct Dense_Params Dense{layer.name[-1]}_params = {{'
                           f'.K={out_channels},'f'.J = {gesN},'f'.out_scale={(s1 * s2) / s4},'
                           f'.s1={s1},.z1={z1},.s2={s2},.z2={z2},.sb={sb},.zb={zb},.s3={s4},'f'.z3={z4}}};\n')
    file.write('#endif\n')
    file.close()
    # saveDir = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/include/'
    # dst = saveDir + header_name.split('/')[-1]
    # shutil.move(header_name, dst)


def main():
    train_dir = './OapNet/train/train_raw/1071101_Johny[5]&Wen[5]_train_New12(J&W)/'
    gesN = 12
    channel = 64
    data_windows = get_rawData(train_dir, 2000, gesN)
    model_path = 'PairNet/model/pairnet_model64_12_20220503.h5'

    S1, Z1, S4, Z4 = get_layer_factor(window_size=50, model_path=model_path, total_features=data_windows)
    make_header(gesN=gesN, window_size=50, S1=S1, Z1=Z1, S4=S4, Z4=Z4,
                model_path=model_path, header_name=f'./include/Qpairnet_params{gesN}_{channel}_optimal.h')


if __name__ == '__main__':
    main()
