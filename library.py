import math
import numpy as np
import tensorflow as tf


def BatchNormalization(x: np.ndarray, params: np.ndarray, eps=1e-3):
    """
    # moving_means = moving_meacns_params * momentum + means * (1 - momentum)
    # moving_var = moving_var_params * momentum + var * (1 - momentum)
    """
    gamma, betta = params[0], params[1]
    moving_means, moving_var = params[2], params[3]
    y = gamma * ((x - moving_means) / tf.sqrt(moving_var + eps)) + betta
    # res = tf.nn.batch_normalization(x, moving_means, moving_var, betta, gamma, eps)

    return y


def Folding_Conv_BN(conv1d_wieghts: np.ndarray, BN: np.ndarray, eps=1e-3):
    """Folding BN weights into Conv1d-weights
     *  conv1d_wieghts : (1, kernel_size, in_channel, out_channel)
     *  Y_bn = gamma * (y - moving_means)/(sqrt(moving_var + eps)) + beta
     *  r_hat = gamma / sqrt(moving_var + eps)
     *  W_hat = r_hat * W
     *  bias_hat = r_hat * (bias - moving_means) + beta
     """
    conv1d_bias = np.zeros((conv1d_wieghts.shape[3],))
    gamma = BN[0].reshape((1, 1, 1, BN[0].shape[0]))
    beta = BN[1]
    mean = BN[2]
    variance = BN[3].reshape((1, 1, 1, BN[3].shape[0]))
    new_weights = conv1d_wieghts * gamma / np.sqrt(variance + eps)
    new_bias = beta + (conv1d_bias - mean) * gamma / np.sqrt(variance + eps)
    return new_weights, new_bias


def test_BN_folding(batch_size, out_width, windows, conv_weight, bn_params):
    weight_fold, weight_fold_bias = Folding_Conv_BN(conv_weight, bn_params)
    res_fold = tf.nn.conv1d(windows, weight_fold[0], stride=1, padding='VALID') + weight_fold_bias

    res = tf.nn.conv1d(windows, conv_weight[0], stride=1, padding='VALID')
    res = BatchNormalization(res, bn_params)


def GroupNormalization(x: np.ndarray, gamma, beta, G, eps=0.00001):
    N, H, W, C = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])
    means, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
    x = (x - means) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, H, W, C])
    return x * gamma + beta


def test_gp(model, path):
    windows = make_window_siginals(path)
    weight = []
    for layer in model.layers:
        if layer.name == 'Gp':
            weight = layer.get_weights()
            break
    gp = GroupNormalization(windows, weight[0], weight[1], 6)


def test_ap(fp_feature: np.ndarray):
    batch_size = fp_feature.shape[0]
    output_width = fp_feature.shape[1]
    out_channels = fp_feature.shape[2]
    fp_feature = tf.reshape(fp_feature, [batch_size, output_width, out_channels])
    fp_result = tf.nn.avg_pool1d(fp_feature, 4, 2, padding="VALID")
    print(fp_result)


def make_window_siginals(txt_path):
    queue = []
    window_size = 50

    with open(txt_path, 'r') as f:
        window = []
        pairNet_samples = f.readlines()[:-50]
        for line in pairNet_samples:
            window.append(list(map(eval, line.split())))
            if len(window) == window_size:
                queue.append(np.array(window[:], dtype=np.float32).reshape((1, 50, 6)))  # window clones
                window.pop(0)
    return np.array(queue)


def make_window_siginals_opt(txt_path):
    queue = []
    window_size = 50
    with open(txt_path, 'r') as f:
        window = []
        pairNet_samples = f.readlines()
        label = pairNet_samples[-1].split()
        for line in pairNet_samples[:-1]:
            window.append(list(map(eval, line.split())))
            if len(window) == window_size:
                queue.append(np.array(window[:], dtype=np.float32).reshape((1, 50, 6)))
                window.pop(0)
    return np.array(queue), label


def cal_scaleZeroPoint(r_max, r_min, q_max=127, q_min=-128):
    scale = (r_max - r_min) / (q_max - q_min)
    zeroPoint = q_max - (r_max / scale)
    zeroPoint = np.clip(zeroPoint, q_min, q_max)
    zeroPoint = int(zeroPoint)
    return scale, zeroPoint


def Quantization(r: np.ndarray, scale, zeroPoint):
    q = np.array(np.clip(np.round(r / scale + zeroPoint), -128, 127), dtype=np.int8)
    return q


def Dequantization(q: np.ndarray, scale, zeroPoint):
    r = np.array(scale * (q - zeroPoint), dtype=np.float32)
    return r


def cosineSimilarity(P: np.ndarray, Q: np.ndarray):
    dot = np.sum(p * q for p, q in zip(P, Q))
    norm_p = np.sum(p * p for p in P) ** 0.5
    norm_q = np.sum(q * q for q in Q) ** 0.5
    cos_sim = dot / ((norm_p * norm_q) + 1e-5)
    # cosine_similarity([P], [Q])
    return cos_sim


def stastics_data(x: np.ndarray, isfloat=True):
    x = x.flatten()
    stastics = {}
    for i in range(len(x)):
        tmp = x[i]
        if isfloat:
            tmp = np.round(x[i], 3)
        stastics[tmp] = stastics.get(tmp, 0) + 1

    indices = sorted(stastics.keys())
    counts = [stastics[k] for k in indices]
    return indices, counts


def bias_Correction(bias: np.ndarray):
    """optional"""
    best_minn, best_maxx = optimal_MinMax(bias)
    scale_bias, zeroPoint_bias = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
    Qbias = Quantization(bias, scale_bias, zeroPoint_bias)
    DEQbias = Dequantization(Qbias, scale_bias, zeroPoint_bias)
    error_bias = bias - DEQbias
    bias_Corr = bias - error_bias
    return bias_Corr


def optimal_MinMax(x: np.ndarray):
    best_cosine_sim = -1
    best_minn, best_maxx = float('inf'), float('-inf')
    means, std = np.mean(x), np.std(x)
    region = np.arange(2, 3.1, 0.1)
    for i in range(len(region)):
        minn, maxx = means - region[i] * std, means + region[i] * std
        scale, zero_point = cal_scaleZeroPoint(r_min=minn, r_max=maxx)
        Q_x = Quantization(x, scale, zero_point)
        DE_Q_x = Dequantization(Q_x, scale, zero_point)
        cosine_sim = cosineSimilarity(x, DE_Q_x)
        if np.mean(cosine_sim) > best_cosine_sim:
            best_minn, best_maxx = minn, maxx
            best_cosine_sim = np.mean(cosine_sim)
            # print(best_cosine_sim)
    return best_minn, best_maxx


def approximate_M(M: float):
    """M = S1 * S2 / S4 , could be approximated to a fixed-point-number(m0) with bit shift(n) """
    m0, n = math.frexp(M)
    return m0, n


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
