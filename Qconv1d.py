from library import *
from tensorflow.keras.models import load_model


def round_near_even(x: float):
    float_x = x
    int_x = int(float_x)

    if float_x < 0:
        next_x = int(float_x) - 1
    else:
        next_x = int(float_x) + 1
    remain = abs(float_x - int_x)

    if remain < 0.5:
        result = int_x
    else:
        if remain > 0.5:
            result = next_x
        else:
            if int_x % 2 == 0:
                result = int_x
            else:
                result = next_x
    return result


def QRelu_clip(x: np.ndarray, use_relu: bool, z3=None, z4=None):
    if use_relu:
        x = np.where(x < z3, z4, x)
        x = np.clip(x, z4, 127)
    else:
        x = np.clip(x, -128, 127)
    return x


def reshape_kernel(kernel: np.ndarray):
    kernel_size = kernel.shape[1]
    in_channels = kernel.shape[2]
    out_channels = kernel.shape[3]
    return np.reshape(kernel, (kernel_size * in_channels, out_channels))


def reshape_feature(input_feature: np.ndarray, kernel_size, stride_size, out_width):
    batch_size = input_feature.shape[0]
    in_channels = input_feature.shape[3]
    reshape_featre = []
    for idx in range(batch_size):
        reshape_f = np.zeros((out_width, kernel_size * in_channels))
        flatten_f = input_feature[idx][0].flatten()
        start = 0
        for i in range(out_width):
            for j in range(kernel_size * in_channels):
                reshape_f[i][j] = flatten_f[(start + j)]
            start += stride_size * in_channels
        reshape_featre.append([reshape_f])
    return np.array(reshape_featre)


def Qconv1d(reshaped_feature: np.ndarray, reshaped_kernel: np.ndarray, kernel_bias: np.ndarray, out_width, KS_inChannel,
            out_channels, down_scalar, z3, z4, is_conv=True):
    if is_conv:
        batch_size = reshaped_feature.shape[0]
        out_feature = np.zeros((batch_size, 1, out_width, out_channels))
        for idx in range(batch_size):
            for i in range(out_width):
                tmp_res = 0
                for j in range(out_channels):
                    for k in range(KS_inChannel):
                        tmp_res += reshaped_feature[idx][0][i][k] * reshaped_kernel[k][j]
                    out_feature[idx][0][i][j] = round_near_even((down_scalar * (tmp_res + kernel_bias[idx][0][i][j])))
                    tmp_res = 0
        out_feature = QRelu_clip(out_feature, z3=z3, z4=z4, use_relu=True)
    else:
        out_feature = np.zeros((out_width, out_channels))
        for i in range(out_width):
            tmp_res = 0
            for j in range(out_channels):
                for k in range(KS_inChannel):
                    tmp_res += reshaped_feature[i][k] * reshaped_kernel[k][j]
                out_feature[i][j] = round_near_even((down_scalar * (tmp_res + kernel_bias[i][j])))
                tmp_res = 0
        out_feature = QRelu_clip(out_feature, use_relu=False)
    return out_feature


def pre_compute_bias(input_feature: np.ndarray, kernel: np.ndarray, bias: np.ndarray, KS_inChannel,
                     s1, z1, s2, z2, s2_b, z2_b, s3, z3, relu_s4, relu_z4, is_conv=True):
    batch_size = input_feature.shape[0]

    if is_conv:
        out_width = bias.shape[0]
        out_channels = bias.shape[1]
        total_bias = np.zeros((batch_size, 1, out_width, out_channels))
        for idx in range(batch_size):
            for i in range(out_width):
                tmp_bias = 0
                for j in range(out_channels):
                    for k in range(KS_inChannel):
                        tmp_bias += ((z1 * z2) - (kernel[k][j] * z1) - (input_feature[idx][0][i][k] * z2))
                    total_bias[idx][0][i][j] = round(
                        tmp_bias + ((s2_b / (s1 * s2)) * (bias[i][j] - z2_b)) + (z3 / ((s1 * s2) / relu_s4)))
                    tmp_bias = 0
        total_bias = total_bias.astype(np.int32)
        return total_bias
    else:
        out_width = batch_size
        out_channels = bias.shape[0]
        res_bias = np.zeros((out_width, out_channels))
        for i in range(out_width):
            tmp_bias = 0
            for j in range(out_channels):
                for k in range(KS_inChannel):
                    tmp_bias += ((z1 * z2) - (kernel[k][j] * z1) - (input_feature[i][k] * z2))
                res_bias[i][j] = np.round(tmp_bias + ((s2_b / (s1 * s2)) * (bias[j] - z2_b)) + (z3 / ((s1 * s2) / s3)))
                tmp_bias = 0
        res_bias = res_bias.astype(np.int32)
        return res_bias


if __name__ == '__main__':
    path = 'Oap/test/1100920_test_(J&W&D&j&in0)/9-8-4/TD20180927-110149_(Wen)_H50_N3_K9-8-4.txt'
    model_pairNet = load_model("PairNet/model/pairnet_model16_9_20220216.h5")
    for layer in model_pairNet.layers:
        if 'cov' in layer.name:
            weight = np.array(layer.get_weights())
        break
    windows = make_window_siginals(path)
    reshape_feature = reshape_feature(windows, kernel_size=3, stride_size=1, out_width=48)
    reshape_kernel = reshape_kernel(weight)
