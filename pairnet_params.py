import datetime
from tensorflow.keras.models import load_model
import numpy as np
from library import *
import shutil
from Qconv1d import reshape_kernel, reshape_feature, pre_compute_bias, Qconv1d


def make_Qpairnet_params(batch_size, input_width, stride_size, gesN, input_signals, path, true_label: list,
                         header_name='./include/Qpairnet_params_32.h'):
    """feed into PairNet_ALLQ_main.c  & mc2_conv1d_main"""
    f = open(header_name, "w+")
    f.write(f"//{datetime.datetime.now()}\n")
    f.write(f"#ifndef QPAIR_NET_PARAMS_H\n")
    f.write("#define QPAIR_NET_PARAMS_H\n\n")
    f.write("#include <stdint.h>\n")
    f.write("#include <stdbool.h>\n\n")
    f.write(f"#define GESN {gesN}\n")
    f.write(f'#define LEN_LABLE {len(true_label)}\n')
    f.write("struct Conv1d_Params {\n\tint batch_size;\n\tint input_width;\n\tint in_channels;\n\t"
            "int out_channels;\n\tint kernel_size;\n\tint stride_size;\n\t""int padding_front;\n\t"
            "int padding_back;\n\tint output_width;\n\tdouble out_scale;\n\t};\n")
    for label in true_label:
        if label >= gesN:
            print(f"{label} > {gesN}")
            raise ValueError
    gt = ''.join([str(i) + ' ' for i in true_label])
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
                convBN, bias = Folding_Conv_BN(conv, BN)
                convBN_bias = np.repeat([bias], output_width, axis=0)
                convBN_bias = np.repeat([convBN_bias], batch_size, axis=0)
                convBN_bias = convBN_bias.reshape([batch_size, 1, output_width, out_channels])
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

                f.write(f"static const double downScalar{layer.name[-2:]} = {s1 * s2_convBN / s4_convBN};\n")
                f.write(f"static const elem_t z3{layer.name[-2:]} = {z3_convBN};\n")
                f.write(f"static const elem_t z4{layer.name[-2:]} = {z4_convBN};\n")
                """Quantization"""
                # reshape input_feature
                QReshaped_feature = reshape_feature(Q_feature, kernel_size, stride_size, output_width)
                # reshape kernel_weight
                QReshaped_kernel = reshape_kernel(QConv_BN)
                # pre-compute bias
                QConv_BN_bias = pre_compute_bias(QReshaped_feature, QReshaped_kernel, QConv_BN_bias[0][0],
                                                 kernel_size * in_channels, s1, z1, s2_convBN, z2_convBN,
                                                 s2_convBN_bias, z2_convBN_bias, s3_convBN, z3_convBN, s4_convBN,
                                                 z4_convBN)
                # compute result
                Q_result = Qconv1d(QReshaped_feature, QReshaped_kernel, QConv_BN_bias, output_width,
                                   kernel_size * in_channels, out_channels, s1 * s2_convBN / s4_convBN,
                                   z3_convBN, z4_convBN)
                Q_feature = Q_result
                """write QConv_BN"""
                f.write(
                    f"static const elem_t QConv_BN{layer.name[-2:]}[{filters}][{in_channels}][{out_channels}] row_align(1)=\n")
                f.write('{')
                for i in range(wrapper):
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
                f.write('};\n')
                """write QConv_BN_mc2"""
                f.write(
                    f"static const elem_t QConv_BN_mc2{layer.name[-2:]}[{kernel_size * in_channels}][{out_channels}] row_align(1)=\n")
                f.write('{')
                for i in range(QReshaped_kernel.shape[0]):
                    f.write('{')
                    for j in range(QReshaped_kernel.shape[1]):
                        if not j == (QReshaped_kernel.shape[1] - 1):
                            f.write(f'{QReshaped_kernel[i][j]},')
                        else:
                            f.write(f'{QReshaped_kernel[i][j]}')
                    if not i == (QReshaped_kernel.shape[0] - 1):
                        f.write('},')
                    else:
                        f.write('}')
                f.write('};\n')
                # write pre-computed QConv_BN_bias
                f.write(
                    f"static const acc_t QConv_BN_bias{layer.name[-2:]}[{batch_size}][{output_width}][{out_channels}] row_align_acc(1) = \n")
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
                    f'static elem_t QConv_BN{layer.name[-2:]}_out[{batch_size}][{output_width}][{out_channels}] row_align(1);\n')
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
                f.write(f"static const double downScalar_dense = {s1 * s2_dense / s3_dense};\n")
                f.write(f"static const elem_t z2_dense = {z2_dense};\n")
                f.write(f"static const double s3_dense = {s3_dense};\n")
                f.write(f"static const elem_t z3_dense = {z3_dense};\n")
                """Quantization"""
                # pre-compute bias
                KS_inChannel = QDense.shape[0]
                QDense_bias = pre_compute_bias(Q_feature, QDense, QDense_bias, KS_inChannel, s1, z1, s2_dense, z2_dense,
                                               s2_dense_bias, z2_dense_bias, s3_dense, z3_dense, 0, 0, is_conv=False)
                Q_result = Qconv1d(Q_feature, QDense, QDense_bias, batch_size, KS_inChannel, gesN,
                                   s1 * s2_dense / s3_dense, 0, 0, is_conv=False)
                Q_feature = Q_result
                f.write(f"static const elem_t QDense_params[{QDense.shape[0]}][{QDense.shape[1]}] = \n")
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
                f.write(f"static const acc_t QDense_bias[{QDense_bias.shape[0]}][{QDense_bias.shape[1]}] = \n")
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
                f.write(f"static elem_t QDense_out[{batch_size}][{gesN}];\n")

        elif 'global_average_pooling1d' in layer.name:
            fp_result = tf.reshape(fp_feature, [batch_size, 3, out_channels])
            fp_result = tf.keras.layers.GlobalAveragePooling1D()(fp_result)
            fp_feature = fp_result
            Q_result = tf.reshape(Q_feature, [batch_size, 3, out_channels])
            Q_result = tf.cast(Q_result, tf.float32)
            Q_result = tf.keras.layers.GlobalAveragePooling1D()(Q_result)
            Q_feature = tf.round(Q_result)
            f.write(f"static elem_t QGap_out[{batch_size}][{out_channels}];\n")
    # f.write(f"double deq_softmax_out[{batch_size}][{gesN}];\n")
    f.write('#endif\n')
    f.close()
    saveDir = '/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests/include/'
    dst = saveDir + header_name.split('/')[-1]
    shutil.move(header_name, dst)


if __name__ == '__main__':
    gesN = 12
    channel = 16
    model_path = './PairNet/model/model_QconvBN16_9.h5'

