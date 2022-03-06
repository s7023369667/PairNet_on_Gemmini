import datetime
from tensorflow.keras.models import load_model
import numpy as np
from library import *
import shutil
from gesture_signals import make_Qsiginals
from Qconv1d import reshape_kernel, reshape_feature, pre_compute_bias, Qconv1d


def make_pairNetQDEQ_params(batch_size, input_width, stride_size, gesN, header_name='./include/pairnet_params.h'):
    """feed into PairNet_QDEQ_main.c"""
    f = open(header_name, "w+")
    f.write(f"//{datetime.datetime.now()}\n")
    f.write("#ifndef PAIR_NET_PARAMS_H\n")
    f.write("#define PAIR_NET_PARAMS_H\n\n")
    f.write("#include <stdint.h>\n")
    f.write("#include <stdbool.h>\n\n")
    f.write("struct Conv1d_Params {\n\tint batch_size;\n\tint input_width;\n\tint in_channels;\n\t"
            "int out_channels;\n\tint kernel_size;\n\tint stride_size;\n\t""int padding_front;\n\t"
            "int padding_back;\n\tint output_width;\n\tdouble out_scale;\n\t};\n")
    model_pairNet = load_model("PairNet/model/model_PairNet_paper.h5")
    for layer in model_pairNet.layers:
        print(layer.name)
        print(np.array(layer.get_weights()).shape)
        # print(layer.get_weights())
        shape = ''
        for i in range(len(np.array(layer.get_weights()).shape)):
            shape += f"[{np.array(layer.get_weights()).shape[i]}]"
        if layer.get_weights():
            if 'batch_normalization' in layer.name:
                f.write(f" static const double {layer.name}{shape} = \n")
                row = np.array(layer.get_weights()).shape[0]
                col = np.array(layer.get_weights()).shape[1]
                f.write('{')
                for i in range(row):
                    f.write('{')
                    for j in range(col):
                        if j != (col - 1):
                            f.write(f"{layer.get_weights()[i][j]},")
                        else:
                            f.write(f"{layer.get_weights()[i][j]}")
                    if i != (row - 1):
                        f.write('},')
                    else:
                        f.write('}')
                f.write('};\n')
            elif 'conv1d' in layer.name:
                wrapper = np.array(layer.get_weights()).shape[0]
                filters = np.array(layer.get_weights()).shape[1]
                in_channels = np.array(layer.get_weights()).shape[2]
                out_channels = np.array(layer.get_weights()).shape[3]
                kernel_size = filters
                do_padding = False
                padding_front, padding_back, padding_size = 0, 0, 0
                if do_padding:
                    padding_back = 1
                    if kernel_size % 2 != 0:
                        padding_front = 1
                    else:
                        padding_front = 0
                padding_size = padding_back + padding_front
                output_width = (input_width - kernel_size + padding_size) // stride_size + 1
                f.write(f" static const double {layer.name}{shape} = \n")
                f.write('{')
                for i in range(wrapper):
                    f.write('{')
                    for j in range(filters):
                        f.write('{')
                        for k in range(in_channels):
                            f.write('{')
                            for l in range(out_channels):
                                if l != (out_channels - 1):
                                    f.write(f"{layer.get_weights()[i][j][k][l]},")
                                else:
                                    f.write(f"{layer.get_weights()[i][j][k][l]}")
                            if k != (in_channels - 1):
                                f.write('},')
                            else:
                                f.write('}')
                        if j != (filters - 1):
                            f.write('},')
                        else:
                            f.write('}')
                    f.write('}')
                f.write('};\n')
                f.write(f' static const struct Conv1d_Params {layer.name}_params = {{'
                        f'.batch_size = {batch_size}, .input_width={input_width}, .in_channels={in_channels},'
                        f' .out_channels = {out_channels},.kernel_size ={kernel_size},.stride_size={stride_size},'
                        f'.padding_front= {padding_front},.padding_back= {padding_back},.output_width={output_width}}};\n')
                f.write(f'static double {layer.name}_out[{batch_size}][1][{output_width}][{out_channels}];\n')
                input_width = output_width
                stride_size = 2
            elif 'dense' in layer.name:
                params = layer.get_weights()[0]
                f.write(f" static const double {layer.name}_params[{len(params)}][{len(params[0])}] = \n")
                f.write('{')
                for i in range(len(params)):
                    f.write('{')
                    for j in range(len(params[i])):
                        if not j == (len(params[i]) - 1):
                            f.write(f'{params[i][j]},')
                        else:
                            f.write(f'{params[i][j]}')
                    if not i == (len(params) - 1):
                        f.write('},')
                    else:
                        f.write('}')
                f.write('};\n')
                bias = layer.get_weights()[1]
                f.write(f" static const double {layer.name}_bias[{len(bias)}] = \n")
                f.write('{')
                for i in range(len(bias)):
                    if not i == (len(bias) - 1):
                        f.write(f'{bias[i]},')
                    else:
                        f.write(f'{bias[i]}')
                f.write('};\n')
                f.write(f" static double dense_out[{batch_size}][{gesN}];\n")

        elif 'global_average_pooling1d' in layer.name:
            f.write(f" static double gap_out[{batch_size}][256];\n")

    f.write('#endif\n')
    f.close()


def make_pairNetALLQ_params(batch_size, input_width, stride_size, gesN, input_signals, path, true_label: list,
                            len_label, header_name='./include/Qpairnet_params.h'):
    """feed into PairNet_ALLQ_main.c"""
    f = open(header_name, "w+")
    f.write(f"//{datetime.datetime.now()}\n")
    f.write("#ifndef QPAIR_NET_PARAMS_H\n")
    f.write("#define QPAIR_NET_PARAMS_H\n\n")
    f.write("#include <stdint.h>\n")
    f.write("#include <stdbool.h>\n\n")
    f.write(f"#define GESN {gesN}\n")
    f.write(f'#define LEN_LABLE {len_label}\n')
    f.write("struct Conv1d_Params {\n\tint batch_size;\n\tint input_width;\n\tint in_channels;\n\t"
            "int out_channels;\n\tint kernel_size;\n\tint stride_size;\n\t""int padding_front;\n\t"
            "int padding_back;\n\tint output_width;\n\tdouble out_scale;\n\t};\n")
    gt = ''.join([str(i)+' ' for i in true_label])
    f.write(f'#define TRUE_LABEL \"{gt}\"\n')
    model_pairNet = load_model(path)
    fp_feature = input_signals
    best_minn, best_maxx = optimal_MinMax(fp_feature)
    s1, z1 = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
    Q_windows = Quantization(fp_feature, s1, z1)
    Q_feature = Q_windows
    for layer in model_pairNet.layers:
        print(layer.name)
        # print(np.array(layer.get_weights(), dtype=object).shape)
        if layer.get_weights():
            if 'conv1d' in layer.name:
                conv = np.array(layer.get_weights())
            elif 'batch_normalization' in layer.name:
                BN = np.array(layer.get_weights())
                wrapper = conv.shape[0]
                filters = conv.shape[1]
                in_channels = conv.shape[2]
                out_channels = conv.shape[3]
                kernel_size = filters
                do_padding = False
                padding_front, padding_back, padding_size = 0, 0, 0
                if do_padding:
                    padding_back = 1
                    padding_front = 1 if kernel_size % 2 != 0 else 0
                padding_size = padding_back + padding_front
                output_width = (input_width - kernel_size + padding_size) // stride_size + 1
                """Folding zeroPointBN into Conv1D"""
                convBN, convBN_bias = Folding_Conv_BN(batch_size, output_width, conv, BN)
                """Quantized weights"""
                best_minn, best_maxx = optimal_MinMax(convBN)
                s2_convBN, z2_convBN = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
                QConv_BN = Quantization(convBN, s2_convBN, z2_convBN)
                """Bias Corrected & Quantized Bias"""
                convBN_biasCorr = bias_Correction(convBN_bias)
                best_minn_bias, best_maxx_bias = optimal_MinMax(convBN_biasCorr)
                s2_convBN_bias, z2_convBN_bias = cal_scaleZeroPoint(r_max=best_maxx_bias, r_min=best_minn_bias)
                QConv_BN_bias = Quantization(convBN_biasCorr, s2_convBN_bias, z2_convBN_bias)
                """calculate each block result, Conv1d -> BN -> Relu """
                fp_result = tf.nn.conv1d(fp_feature, convBN[0], stride=stride_size, padding='VALID') + convBN_biasCorr
                best_minn_res, best_maxx_res = optimal_MinMax(fp_result)
                s3_convBN, z3_convBN = cal_scaleZeroPoint(r_max=best_maxx_res, r_min=best_minn_res)
                fp_result = tf.clip_by_value(fp_result, clip_value_min=0, clip_value_max=np.max(fp_result))
                fp_feature = fp_result
                s4_convBN, z4_convBN = cal_scaleZeroPoint(r_max=np.max(fp_result), r_min=0,
                                                          q_max=127, q_min=z3_convBN)

                f.write(f"const double downScalar{layer.name[-2:]} = {s1 * s2_convBN / s4_convBN};\n")
                f.write(f"const elem_t z3{layer.name[-2:]} = {z4_convBN};\n")
                f.write(f"const elem_t z4{layer.name[-2:]} = {z4_convBN};\n")
                """Quantization"""
                # reshape input_feature
                Q_reshaped_feature = reshape_feature(Q_feature, kernel_size, stride_size, output_width)
                # reshape kernel_weight
                Q_reshaped_kernel = reshape_kernel(QConv_BN)
                # pre-compute bias
                QConv_BN_bias = pre_compute_bias(Q_reshaped_feature, Q_reshaped_kernel, QConv_BN_bias[0][0],
                                                 kernel_size * in_channels, s1, z1, s2_convBN, z2_convBN,
                                                 s2_convBN_bias, z2_convBN_bias,s3_convBN, z3_convBN, s4_convBN,
                                                 z4_convBN)
                # compute result
                Q_result = Qconv1d(Q_reshaped_feature, Q_reshaped_kernel, QConv_BN_bias, output_width, kernel_size * in_channels,
                                   out_channels, s1 * s2_convBN / s4_convBN, z3_convBN, z4_convBN)
                Q_feature = Q_result
                """write QConv_BN"""
                f.write(
                    f"const elem_t QConv_BN{layer.name[-2:]}[{wrapper}][{filters}][{in_channels}][{out_channels}]=\n")
                f.write('{')
                for i in range(wrapper):
                    f.write('{')
                    for j in range(filters):
                        f.write('{')
                        for k in range(in_channels):
                            f.write('{')
                            for l in range(out_channels):
                                if l != (out_channels - 1):
                                    f.write(f"{QConv_BN[i][j][k][l]},")
                                else:
                                    f.write(f"{QConv_BN[i][j][k][l]}")
                            if k != (in_channels - 1):
                                f.write('},')
                            else:
                                f.write('}')
                        if j != (filters - 1):
                            f.write('},')
                        else:
                            f.write('}')
                    f.write('}')
                f.write('};\n')
                f.write(
                    f"const acc_t QConv_BN_bias{layer.name[-2:]}[{batch_size}][1][{output_width}][{out_channels}] = \n")
                f.write("\n{")
                for i in range(batch_size):
                    f.write("{")
                    for j in range(1):
                        f.write("{")
                        for k in range(output_width):
                            f.write("{")
                            for l in range(out_channels):
                                if l != out_channels - 1:
                                    f.write(f"{QConv_BN_bias[i][j][k][l]},")
                                else:
                                    f.write(f"{QConv_BN_bias[i][j][k][l]}")
                            if k != output_width - 1:
                                f.write("},")
                            else:
                                f.write("}")
                        f.write("}")
                    if i != batch_size - 1:
                        f.write("},\n")
                    else:
                        f.write("}\n")
                f.write("};\n")
                f.write(f'const struct Conv1d_Params QConv_BN{layer.name[-2:]}_params = {{'
                        f'.batch_size = {batch_size}, .input_width={input_width}, .in_channels={in_channels},'
                        f'.out_channels = {out_channels},.kernel_size ={kernel_size},.stride_size={stride_size},'
                        f'.padding_front= {padding_front},.padding_back= {padding_back},.output_width={output_width}}};\n')
                f.write(
                    f'static elem_t QConv_BN{layer.name[-2:]}_out[{batch_size}][1][{output_width}][{out_channels}];\n')
                input_width = output_width
                stride_size = 2
                s1, z1 = s4_convBN, z4_convBN

            elif 'dense' in layer.name:
                dense, dense_bias = np.array(layer.get_weights()[0]), np.array(layer.get_weights()[1])
                """Quantized weights"""
                best_minn, best_maxx = optimal_MinMax(dense)
                s2_dense, z2_dense = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
                QDense = Quantization(dense, s2_dense, z2_dense)
                """Bias Corrected & Quantized Bias"""
                dense_biasCorr = bias_Correction(dense_bias)
                best_minn_bias, best_maxx_bias = optimal_MinMax(dense_biasCorr)
                s2_dense_bias, z2_dense_bias = cal_scaleZeroPoint(r_max=best_maxx_bias, r_min=best_minn_bias)
                QDense_bias = Quantization(dense_biasCorr, s2_dense_bias, z2_dense_bias)
                """calculate dense result"""
                fp_result = np.matmul(fp_feature, dense) + dense_bias
                fp_feature = fp_result
                best_minn_res, best_maxx_res = optimal_MinMax(fp_result)
                s3_dense, z3_dense = cal_scaleZeroPoint(r_max=best_maxx_res, r_min=best_minn_res)
                f.write(f"const double downScalar_dense = {s1 * s2_dense / s3_dense};\n")
                f.write(f"const elem_t z2_dense = {z2_dense};\n")
                f.write(f"const double s3_dense = {s3_dense};\n")
                f.write(f"const elem_t z3_dense = {z3_dense};\n")
                """Quantization"""
                # pre-compute bias
                KS_inChannel = QDense.shape[0]
                QDense_bias = pre_compute_bias(Q_feature, QDense, QDense_bias, KS_inChannel, s1, z1, s2_dense, z2_dense,
                                               s2_dense_bias, z2_dense_bias, s3_dense, z3_dense, 0, 0, is_conv=False)
                Q_result = Qconv1d(Q_feature, QDense, QDense_bias, batch_size, KS_inChannel, gesN,
                                   s1 * s2_dense / s3_dense, 0, 0, is_conv=False)
                Q_feature = Q_result
                f.write(f"const elem_t QDense_params[{QDense.shape[0]}][{QDense.shape[1]}] = \n")
                f.write('{')
                for i in range(QDense.shape[0]):
                    f.write('{')
                    for j in range(QDense.shape[1]):
                        if not j == (QDense.shape[1] - 1):
                            f.write(f'{QDense[i][j]},')
                        else:
                            f.write(f'{QDense[i][j]}')
                    if not i == (QDense.shape[0] - 1):
                        f.write('},')
                    else:
                        f.write('}')
                f.write('};\n')
                f.write(f"const acc_t QDense_bias[{QDense_bias.shape[0]}][{QDense_bias.shape[1]}] = \n")
                f.write('{')
                for i in range(QDense_bias.shape[0]):
                    f.write('{')
                    for j in range(QDense_bias.shape[1]):
                        if not j == (QDense_bias.shape[1] - 1):
                            f.write(f'{QDense_bias[i][j]},')
                        else:
                            f.write(f'{QDense_bias[i][j]}')
                    if not i == (QDense_bias.shape[0] - 1):
                        f.write('},')
                    else:
                        f.write('}')
                f.write('};\n')
                f.write(f"elem_t QDense_out[{batch_size}][{gesN}];\n")

        elif 'global_average_pooling1d' in layer.name:
            fp_result = tf.reshape(fp_feature, [batch_size, 3, out_channels])
            fp_result = tf.keras.layers.GlobalAveragePooling1D()(fp_result)
            fp_feature = fp_result
            Q_result = tf.reshape(Q_feature, [batch_size, 3, out_channels])
            Q_result = tf.cast(Q_result, tf.float32)
            Q_result = tf.keras.layers.GlobalAveragePooling1D()(Q_result)
            Q_feature = tf.round(Q_result)
            f.write(f"elem_t QGap_out[{batch_size}][{out_channels}];\n")
    f.write(f"double deq_softmax_out[{batch_size}][{gesN}];\n")
    f.write('#endif\n')
    f.close()
    dst = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/include/Qpairnet_params.h'
    shutil.copyfile('./include/Qpairnet_params.h', dst)


def make_pairNet_mc2conv1d_params(batch_size, input_width, stride_size, gesN, input_signals, path, true_label: list,
                                  len_label, header_name='./include/Qpairnet_mc2conv1d_params.h'):
    """feed into 1d_with_ch.c"""
    f = open(header_name, "w+")
    f.write(f"//{datetime.datetime.now()}\n")
    f.write("#ifndef QPAIRNET_MC2CONV1D_PARAMS_H\n")
    f.write("#define QPAIRNET_MC2CONV1D_PARAMS_H\n\n")
    f.write("#include <stdint.h>\n")
    f.write("#include <stdbool.h>\n\n")
    f.write(f"#define GESN {gesN}\n")
    f.write(f'#define LEN_LABLE {len_label}\n')
    f.write("struct Conv1d_Params {\n\tint batch_size;\n\tint input_width;\n\tint in_channels;\n\t"
            "int out_channels;\n\tint kernel_size;\n\tint stride_size;\n\t""int padding_front;\n\t"
            "int padding_back;\n\tint output_width;\n\tdouble out_scale;\n\t};\n")
    gt = ''.join([str(i)+' ' for i in true_label])
    f.write(f'#define TRUE_LABEL \"{gt}\"\n')
    model_pairNet = load_model(path)
    fp_feature = input_signals
    best_minn, best_maxx = optimal_MinMax(fp_feature)
    s1, z1 = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
    Q_windows = Quantization(fp_feature, s1, z1)
    Q_feature = Q_windows
    for layer in model_pairNet.layers:
        print(layer.name)
        # print(np.array(layer.get_weights(), dtype=object).shape)
        if layer.get_weights():
            if 'conv1d' in layer.name:
                conv = np.array(layer.get_weights())
            elif 'batch_normalization' in layer.name:
                BN = np.array(layer.get_weights())
                wrapper = conv.shape[0]
                filters = conv.shape[1]
                in_channels = conv.shape[2]
                out_channels = conv.shape[3]
                kernel_size = filters
                do_padding = False
                padding_front, padding_back, padding_size = 0, 0, 0
                if do_padding:
                    padding_back = 1
                    padding_front = 1 if kernel_size % 2 != 0 else 0
                padding_size = padding_back + padding_front
                output_width = (input_width - kernel_size + padding_size) // stride_size + 1
                """Float"""
                # Folding zeroPointBN into Conv1D
                convBN, convBN_bias = Folding_Conv_BN(batch_size, output_width, conv, BN)
                # Quantized weights
                best_minn, best_maxx = optimal_MinMax(convBN)
                s2_convBN, z2_convBN = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
                QConv_BN = Quantization(convBN, s2_convBN, z2_convBN)
                # Bias Corrected & Quantized Bias
                convBN_biasCorr = bias_Correction(convBN_bias)
                best_minn_bias, best_maxx_bias = optimal_MinMax(convBN_biasCorr)
                s2_convBN_bias, z2_convBN_bias = cal_scaleZeroPoint(r_max=best_maxx_bias, r_min=best_minn_bias)
                QConv_BN_bias = Quantization(convBN_biasCorr, s2_convBN_bias, z2_convBN_bias)
                QConv_BN_bias = QConv_BN_bias.astype(np.int32)
                # calculate each block result, Conv1d -> BN -> Relu
                fp_result = tf.nn.conv1d(fp_feature, convBN[0], stride=stride_size, padding='VALID') + convBN_biasCorr
                best_minn_res, best_maxx_res = optimal_MinMax(fp_result)
                s3_convBN, z3_convBN = cal_scaleZeroPoint(r_max=best_maxx_res, r_min=best_minn_res)
                fp_result = tf.clip_by_value(fp_result, clip_value_min=0, clip_value_max=np.max(fp_result))
                fp_feature = fp_result
                s4_convBN, z4_convBN = cal_scaleZeroPoint(r_max=np.max(fp_result), r_min=0,
                                                          q_max=127, q_min=z3_convBN)
                f.write(f"const double downScalar{layer.name[-2:]} = {s1 * s2_convBN / s4_convBN};\n")
                f.write(f"const elem_t z3{layer.name[-2:]} = {z4_convBN};\n")
                f.write(f"const elem_t z4{layer.name[-2:]} = {z4_convBN};\n")
                """Quantization"""
                # reshape input_feature
                Q_reshaped_feature = reshape_feature(Q_feature, kernel_size, stride_size, output_width)
                # reshape kernel_weight
                QConv_BN = reshape_kernel(QConv_BN)
                # pre-compute bias
                QConv_BN_bias = pre_compute_bias(Q_reshaped_feature, QConv_BN, QConv_BN_bias[0][0],
                                                 kernel_size * in_channels,
                                                 s1, z1, s2_convBN, z2_convBN, s2_convBN_bias, z2_convBN_bias,
                                                 s3_convBN, z3_convBN, s4_convBN, z4_convBN)
                # compute result
                Q_result = Qconv1d(Q_reshaped_feature, QConv_BN, QConv_BN_bias, output_width, kernel_size * in_channels,
                                   out_channels, s1 * s2_convBN / s4_convBN, z3_convBN, z4_convBN)
                Q_feature = Q_result
                # write QConv_BN
                f.write(
                    f"const elem_t QConv_BN{layer.name[-2:]}[{kernel_size * in_channels}][{out_channels}]=\n")
                f.write('{')
                for i in range(QConv_BN.shape[0]):
                    f.write('{')
                    for j in range(QConv_BN.shape[1]):
                        if not j == (QConv_BN.shape[1] - 1):
                            f.write(f'{QConv_BN[i][j]},')
                        else:
                            f.write(f'{QConv_BN[i][j]}')
                    if not i == (QConv_BN.shape[0] - 1):
                        f.write('},')
                    else:
                        f.write('}')
                f.write('};\n')
                # write pre-computed QConv_BN_bias
                f.write(
                    f"const acc_t QConv_BN_bias{layer.name[-2:]}[{batch_size}][{output_width}][{out_channels}] = \n")
                f.write("\n{")
                for i in range(batch_size):
                    f.write("{")
                    for j in range(1):
                        for k in range(output_width):
                            f.write("{")
                            for l in range(out_channels):
                                if l != out_channels - 1:
                                    f.write(f"{QConv_BN_bias[i][j][k][l]},")
                                else:
                                    f.write(f"{QConv_BN_bias[i][j][k][l]}")
                            if k != output_width - 1:
                                f.write("},")
                            else:
                                f.write("}")
                    if i != batch_size - 1:
                        f.write("},\n")
                    else:
                        f.write("}\n")
                f.write("};\n")
                f.write(f'const struct Conv1d_Params QConv_BN{layer.name[-2:]}_params = {{'
                        f'.batch_size = {batch_size}, .input_width={input_width}, .in_channels={in_channels},'
                        f'.out_channels = {out_channels},.kernel_size ={kernel_size},.stride_size={stride_size},'
                        f'.padding_front= {padding_front},.padding_back= {padding_back},.output_width={output_width}}};\n')
                f.write(
                    f'static elem_t QConv_BN{layer.name[-2:]}_out[{batch_size}][{output_width}][{out_channels}];\n')
                input_width = output_width
                stride_size = 2
                s1, z1 = s4_convBN, z4_convBN
            elif 'dense' in layer.name:
                dense, dense_bias = np.array(layer.get_weights()[0]), np.array(layer.get_weights()[1])
                """Float"""
                # Quantized weights
                best_minn, best_maxx = optimal_MinMax(dense)
                s2_dense, z2_dense = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
                QDense = Quantization(dense, s2_dense, z2_dense)
                # Bias Corrected & Quantized Bias
                dense_biasCorr = bias_Correction(dense_bias)
                best_minn_bias, best_maxx_bias = optimal_MinMax(dense_biasCorr)
                s2_dense_bias, z2_dense_bias = cal_scaleZeroPoint(r_max=best_maxx_bias, r_min=best_minn_bias)
                QDense_bias = Quantization(dense_biasCorr, s2_dense_bias, z2_dense_bias)
                # calculate dense result
                fp_result = np.matmul(fp_feature, dense) + dense_bias
                fp_feature = fp_result
                best_minn_res, best_maxx_res = optimal_MinMax(fp_result)
                s3_dense, z3_dense = cal_scaleZeroPoint(r_max=best_maxx_res, r_min=best_minn_res)
                f.write(f"const double downScalar_dense = {s1 * s2_dense / s3_dense};\n")
                f.write(f"const elem_t z2_dense = {z2_dense};\n")
                f.write(f"const double s3_dense = {s3_dense};\n")
                f.write(f"const elem_t z3_dense = {z3_dense};\n")
                """Quantization"""
                # pre-compute bias
                KS_inChannel = QDense.shape[0]
                QDense_bias = pre_compute_bias(Q_feature, QDense, QDense_bias, KS_inChannel, s1, z1, s2_dense, z2_dense,
                                               s2_dense_bias, z2_dense_bias, s3_dense, z3_dense, 0, 0, is_conv=False)
                Q_result = Qconv1d(Q_feature, QDense, QDense_bias, batch_size, KS_inChannel, gesN,
                                   s1 * s2_dense / s3_dense, 0, 0, is_conv=False)
                Q_feature = Q_result
                f.write(f"const elem_t QDense_params[{QDense.shape[0]}][{QDense.shape[1]}] = \n")
                f.write('{')
                for i in range(QDense.shape[0]):
                    f.write('{')
                    for j in range(QDense.shape[1]):
                        if not j == (QDense.shape[1] - 1):
                            f.write(f'{QDense[i][j]},')
                        else:
                            f.write(f'{QDense[i][j]}')
                    if not i == (QDense.shape[0] - 1):
                        f.write('},')
                    else:
                        f.write('}')
                f.write('};\n')
                f.write(f"const acc_t QDense_bias[{QDense_bias.shape[0]}][{QDense_bias.shape[1]}] = \n")
                f.write('{')
                for i in range(QDense_bias.shape[0]):
                    f.write('{')
                    for j in range(QDense_bias.shape[1]):
                        if not j == (QDense_bias.shape[1] - 1):
                            f.write(f'{QDense_bias[i][j]},')
                        else:
                            f.write(f'{QDense_bias[i][j]}')
                    if not i == (QDense_bias.shape[0] - 1):
                        f.write('},')
                    else:
                        f.write('}')
                f.write('};\n')
                f.write(f"elem_t QDense_out[{batch_size}][{gesN}];\n")

        elif 'global_average_pooling1d' in layer.name:
            fp_result = tf.reshape(fp_feature, [batch_size, 3, out_channels])
            fp_result = tf.keras.layers.GlobalAveragePooling1D()(fp_result)
            fp_feature = fp_result
            Q_result = tf.reshape(Q_feature, [batch_size, 3, out_channels])
            Q_result = tf.cast(Q_result, tf.float32)
            Q_result = tf.keras.layers.GlobalAveragePooling1D()(Q_result)
            Q_feature = tf.round(Q_result)
            f.write(f"elem_t QGap_out[{batch_size}][{out_channels}];\n")
    f.write(f"double deq_softmax_out[{batch_size}][{gesN}];\n")
    f.write('#endif\n')
    f.close()
    dst = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/include/Qpairnet_mc2conv1d_params.h'
    shutil.copyfile('./include/Qpairnet_mc2conv1d_params.h', dst)


