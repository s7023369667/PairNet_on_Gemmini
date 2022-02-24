import datetime
from tensorflow.keras.models import load_model
from Oap.train.Oap_GD_loss import OaP_GD_loss
import numpy as np
import tensorflow_addons as tfa
from library import *


def make_oapNetALLQ_params(batch_size, input_width, gesN, input_signals, stride_size=1,
                           header_name='./include/Qoap_params.h'):
    """feed into OapNet_ALLQ_main.c"""
    f = open(header_name, "w+")
    f.write(f"//{datetime.datetime.now()}\n")
    f.write("#ifndef QOAP_PARAMETERS_H\n")
    f.write("#define QOAP_PARAMETERS_H\n\n")
    f.write("//#include <include/gemmini_params.h>\n")
    f.write("//#include <include/gemmini.h>\n")
    f.write('//#include "include/gemmini_nn.h"\n')
    f.write("#include <stdint.h>\n")
    f.write("#include <stdbool.h>\n\n")
    f.write("struct Conv1d_Params {\n\tint batch_size;\n\tint input_width;\n\tint in_channels;\n\t"
            "int out_channels;\n\tint kernel_size;\n\tint stride_size;\n\t""int padding_front;\n\t"
            "int padding_back;\n\tint output_width;\n\tdouble out_scale;\n\t};\n")
    model_OaP = load_model('Oap/model_h5/best_OaP_SAM_adjust_branch_s1_1205.h5',
                           custom_objects={'OaP_GD_loss': OaP_GD_loss,
                                           'GroupNormalization': tfa.layers.GroupNormalization})
    fp_feature = input_signals
    group = 6
    for layer in model_OaP.layers:
        print(layer.name)
        print(np.array(layer.get_weights()).shape)
        # print(np.array(layer.get_weights()))
        if layer.get_weights():
            if 'conv' in layer.name:
                do_padding = 'SAME'
                if layer.name == 'conv2_1' or layer.name == 'conv3_1':
                    do_padding = 'VALID'
                conv = np.array(layer.get_weights())
                wrapper = conv.shape[0]
                filters = conv.shape[1]
                in_channels = conv.shape[2]
                out_channels = conv.shape[3]
                kernel_size = filters
                padding_front, padding_back, padding_size = 0, 0, 0
                if do_padding == 'SAME':
                    padding_back = 1
                    if kernel_size % 2 != 0:
                        padding_front = 1
                    else:
                        padding_front = 0
                padding_size = padding_back + padding_front
                output_width = (input_width - kernel_size + padding_size) // stride_size + 1
                print(output_width)
                """Quantized weights"""
                best_minn, best_maxx = optimal_MinMax(conv)
                s2_conv, z2_conv = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
                QConv = Quantization(conv, s2_conv, z2_conv)
                f.write(f"const double s2_conv{layer.name[-3:]} = {s2_conv};\n")
                f.write(f"const elem_t z2_conv{layer.name[-3:]} = {z2_conv};\n")
                """calculate and quantized, Conv1d """
                fp_result = tf.nn.conv1d(fp_feature, conv[0], stride=stride_size, padding=do_padding)

                best_minn_res, best_maxx_res = optimal_MinMax(fp_result)
                s3_conv, z3_conv = cal_scaleZeroPoint(r_max=best_maxx_res, r_min=best_minn_res)
                f.write(f"const double s3_conv{layer.name[-3:]} = {s3_conv};\n")
                f.write(f"const elem_t z3_conv{layer.name[-3:]} = {z3_conv};\n")
                f.write(
                    f"const elem_t QConv{layer.name[-2:]}[{wrapper}][{filters}][{in_channels}][{out_channels}]=\n")
                f.write('{')
                for i in range(wrapper):
                    f.write('{')
                    for j in range(filters):
                        f.write('{')
                        for k in range(in_channels):
                            f.write('{')
                            for l in range(out_channels):
                                if l != (out_channels - 1):
                                    f.write(f"{QConv[i][j][k][l]},")
                                else:
                                    f.write(f"{QConv[i][j][k][l]}")
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
                f.write(f'static const struct QConv_Params {layer.name}_params = {{'
                        f'.batch_size = {batch_size}, .input_width={input_width}, .in_channels={in_channels},'
                        f'.out_channels = {out_channels},.kernel_size ={kernel_size},.stride_size={stride_size},'
                        f'.padding_front= {padding_front},.padding_back= {padding_back},.output_width={output_width}}};\n')
                f.write(f'static double QConv{layer.name[-3:]}_out[{batch_size}][1][{output_width}][{out_channels}];\n')

                input_width = output_width
                fp_feature = fp_result
        elif 'Gp' in layer.name:
            gp = np.array(layer.get_weights())
            row = gp.shape[0]
            col = gp.shape[1]
            """Quantized weights"""
            best_minn, best_maxx = optimal_MinMax(gp)
            s2_gp, z2_gp = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
            QGp = Quantization(gp, s2_gp, z2_gp)
            """calculate GP & Relu result"""
            fp_result = GroupNormalization(fp_feature, gp[0], gp[1], group)
            best_minn_res, best_maxx_res = optimal_MinMax(fp_result)
            s3_gp, z3_gp = cal_scaleZeroPoint(r_max=best_maxx_res, r_min=best_minn_res)
            f.write(f"const double s3_gp{layer.name[-3:]} = {s3_gp};\n")
            f.write(f"const elem_t z3_gp{layer.name[-3:]} = {z3_gp};\n")
            fp_result = tf.clip_by_value(fp_feature, clip_value_min=0, clip_value_max=np.max(fp_feature))
            fp_feature = fp_result
            s4_relu, z4_relu = cal_scaleZeroPoint(r_max=np.max(fp_result), r_min=0,
                                                  q_max=127, q_min=z3_gp)
            f.write(f"const double s4_relu{layer.name[-3:]} = {s4_relu};\n")
            f.write(f"const elem_t z4_relu{layer.name[-3:]} = {z4_relu};\n")
            f.write(f"const elem_t QGP_{layer.name[-3:]}[2][{QGp.shape[1]}] = \n")
            f.write('{')
            for i in range(row):
                f.write('{')
                for j in range(col):
                    if j != (col - 1):
                        f.write(f"{QGp[i][j]},")
                    else:
                        f.write(f"{QGp[i][j]}")
                if i != (row - 1):
                    f.write('},')
                else:
                    f.write('}')
            f.write('};\n')
            group = 16
            # # DO Average Pooling
            if layer.name == 'Gp1_6':
                print(fp_feature.shape[0], fp_feature.shape[1], fp_feature.shape[2], fp_feature[3])
                fp_feature = tf.reshape(fp_feature, [batch_size, 50, 64])
                fp_result = tf.nn.avg_pool1d(fp_feature, 4, 2, padding="VALID")
                fp_feature = fp_result

        elif 'output' in layer.name:
            dense, dense_bias = np.array(layer.get_weights()[0]), np.array(layer.get_weights()[1])
            """Quantized weights"""
            best_minn, best_maxx = optimal_MinMax(dense)
            s2_dense, z2_dense = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
            QDense = Quantization(dense, s2_dense, z2_dense)
            f.write(f"const double s2_dense{layer.name[-2:]} = {s2_dense};\n")
            f.write(f"const elem_t z2_dense{layer.name[-2:]} = {z2_dense};\n")
            """Bias Corrected & Quantized Bias"""
            dense_biasCorr = bias_Correction(dense_bias)
            best_minn_bias, best_maxx_bias = optimal_MinMax(dense_biasCorr)
            s2_dense_bias, z2_dense_bias = cal_scaleZeroPoint(r_max=best_maxx_bias, r_min=best_minn_bias)
            QDense_bias = Quantization(dense_biasCorr, s2_dense_bias, z2_dense_bias)
            f.write(f"const double s2_dense_bias{layer.name[-2:]} = {s2_dense_bias};\n")
            f.write(f"const elem_t z2_dense_bias{layer.name[-2:]} = {z2_dense_bias};\n")
            """calculate dense result"""
            fp_result = np.matmul(fp_feature, dense)
            fp_result = fp_result + dense_bias
            fp_feature = fp_result
            best_minn_res, best_maxx_res = optimal_MinMax(fp_result)
            s3_dense, z3_dense = cal_scaleZeroPoint(r_max=best_maxx_res, r_min=best_minn_res)
            f.write(f"const double s3_dense{layer.name[-2:]} = {s3_dense};\n")
            f.write(f"const elem_t z3_dense{layer.name[-2:]} = {z3_dense};\n")
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
            f.write(f"const acc_t QDense_bias[{QDense_bias.shape[0]}] = \n")
            f.write('{')
            for i in range(QDense_bias.shape[0]):
                if not i == (len(QDense_bias) - 1):
                    f.write(f'{QDense_bias[i]},')
                else:
                    f.write(f'{QDense_bias[i]}')
            f.write('};\n')
            f.write(f"elem_t QDense_out[{batch_size}][{gesN}];\n")


            # f.write(f"elem_t Qap_out[{batch_size}][{output_width}][{out_channels}];\n")

    f.write(f"double deq_softmax_out[{batch_size}][{gesN}];\n")

    f.write('#endif\n')
    f.close()
