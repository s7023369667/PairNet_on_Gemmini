import math
import matplotlib.pyplot as plt
import numpy as np
from library import *
from tensorflow.keras.models import load_model


def plt_siginal(windows):
    best_minn, best_maxx = optimal_MinMax(windows)
    scale, zeroPoint = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
    Qwindows_optimal = Quantization(windows, scale, zeroPoint)
    DEQwindows_optimal = Dequantization(Qwindows_optimal, scale, zeroPoint)
    indices_optimal, counts_optimal = stastics_data(DEQwindows_optimal)

    indices, counts = stastics_data(windows)
    cosine_weights = cosineSimilarity(DEQwindows_optimal, windows)

    plt.plot(indices, counts)
    # plt.plot(indices_optimal, counts_optimal, 'tomato')
    plt.ylabel('Counts')
    # plt.legend(["gestures", "Optimal_Deq(Q(gestures))"])
    plt.title(f'Cosine Similarity = {np.mean(cosine_weights)}')

    plt.show()


def plt_weights(model):
    windows = make_window_siginals(path)
    weight1, weight2, weight3, weight4, weight5 = [], [], [], [], []
    bn_params1, bn_params2, bn_params3, bn_params4, bn_params5 = [], [], [], [], []
    for layer in model.layers:
        if layer.name == 'conv1d_5':
            weight1 = np.array(layer.get_weights())

        if layer.name == 'batch_normalization_5':
            bn_params1 = np.array(layer.get_weights())
            break

    weight1_folding, weight1_bias_folding = Folding_Conv_BN(weight1, bn_params1)

    # weights
    indices_folding, counts_folding = stastics_data(weight1_folding)
    best_minn, best_maxx = optimal_MinMax(weight1_folding)
    scale, zeroPoint = cal_scaleZeroPoint(r_min=best_minn, r_max=best_maxx)
    Qweight1_folding = Quantization(weight1_folding, scale, zeroPoint)
    DEQweight1_folding = Dequantization(Qweight1_folding, scale, zeroPoint)
    DEQindices_weight_folding, DEQcounts_weight_folding = stastics_data(DEQweight1_folding)
    cosine_weights = cosineSimilarity(weight1_folding, DEQweight1_folding)
    ##bias
    indices_folding_bias, counts_folding_bias = stastics_data(weight1_bias_folding)
    weight1_biasCorr_folding = bias_Correction(weight1_bias_folding)
    best_minn, best_maxx = optimal_MinMax(weight1_biasCorr_folding)
    scale_biasCorr, zeroPoint_biasCorrs = cal_scaleZeroPoint(r_min=best_minn, r_max=best_maxx)
    Qweight1_biasCorr_folding = Quantization(weight1_biasCorr_folding, scale_biasCorr, zeroPoint_biasCorrs)
    DEQweight1_biasCorr_folding = Dequantization(Qweight1_biasCorr_folding, scale_biasCorr, zeroPoint_biasCorrs)
    DEQindices_folding_biasCorr, DEQcounts_folding_biasCorr = stastics_data(DEQweight1_biasCorr_folding)
    cosine_bias = cosineSimilarity(weight1_bias_folding, DEQweight1_biasCorr_folding)
    plt.figure(figsize=(8, 5))
    # plt.plot(indices_folding, counts_folding)
    plt.plot(DEQindices_weight_folding, DEQcounts_weight_folding, marker='o', label='point')
    # plt.plot()
    plt.xlabel('Weight Values', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.legend(["Weight After Quantization"], fontsize=10)
    plt.title(f'Cosine Similarity = {np.mean(cosine_weights)}', fontsize=12)
    plt.show()
    # plt.savefig("L5weight_quantization.png")
    # plt.close()

    # plt.figure(figsize=(8, 5))
    # plt.plot(indices_folding_bias, counts_folding_bias)
    # plt.plot(DEQindices_folding_biasCorr, DEQcounts_folding_biasCorr, 'tomato')
    # plt.xlabel('Weight Values', fontsize=12)
    # plt.ylabel('Counts', fontsize=12)
    # plt.legend(["Bias Before Quantization", "Bias After Quantization"], fontsize=12)
    # plt.title(f'Cosine Similarity = {np.mean(cosine_bias)}', fontsize=12)
    # plt.savefig("L5bias_quantization.png")
    # plt.close()

    # plt.plot(indices_folding_bias, counts_folding_bias)
    # plt.plot(DEQindices_folding_biasCorr, DEQcounts_folding_biasCorr, 'tomato')
    # plt.ylabel('Counts')
    # plt.legend(["ConvBN_bias", "Deq(Q(ConvBN_biasCorr))"])
    # plt.xlabel('Values')
    # plt.title(f'Cosine Similarity = {np.mean(cosine_bias)}')
    # plt.show()


def test_conv(x, w_folding, w_folding_bias, stride=2):
    # # weight
    best_minn, best_maxx = optimal_MinMax(w_folding)
    scale_w, zeroPoint_w = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
    Qw_folding = Quantization(w_folding, scale_w, zeroPoint_w)
    DEQw_folding = Dequantization(Qw_folding, scale_w, zeroPoint_w)
    # # bias
    w_folding_biasCorr = bias_Correction(w_folding_bias)
    best_minn_bias, best_maxx_bias = optimal_MinMax(w_folding_biasCorr)
    s2_convBN_bias, z2_convBN_bias = cal_scaleZeroPoint(r_max=best_maxx_bias, r_min=best_minn_bias)
    Qw_folding_biasCorr = Quantization(w_folding_biasCorr, s2_convBN_bias, z2_convBN_bias)
    DEQw_folding_biasCorr = Dequantization(Qw_folding_biasCorr, s2_convBN_bias, z2_convBN_bias)

    res_q_deq = tf.nn.conv1d(x, DEQw_folding[0], stride=stride, padding='VALID') + DEQw_folding_biasCorr
    best_maxx_res, best_minn_res = optimal_MinMax(res_q_deq)
    s3_convBN, z3_convBN = cal_scaleZeroPoint(r_max=best_maxx_res, r_min=best_minn_res)
    res_q_deq = tf.clip_by_value(res_q_deq, clip_value_min=0, clip_value_max=np.max(res_q_deq))
    s4_convBN, z4_convBN = cal_scaleZeroPoint(r_max=np.max(res_q_deq), r_min=0, q_max=127, q_min=z3_convBN)

    Qres_q_deq = Quantization(res_q_deq, s4_convBN, z4_convBN)
    DEQres_q_deq = Dequantization(Qres_q_deq, s4_convBN, z4_convBN)

    return DEQres_q_deq


def test_dense(x, dense, dense_bias):
    # # weight
    best_minn, best_maxx = optimal_MinMax(dense)
    scale_w, zeroPoint_w = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
    Qdense = Quantization(dense, scale_w, zeroPoint_w)
    DEQQdense = Dequantization(Qdense, scale_w, zeroPoint_w)
    # # bias
    dense_bias_biasCorr = bias_Correction(dense_bias)

    res = np.matmul(x, dense) + dense_bias_biasCorr
    best_maxx_res, best_minn_res = optimal_MinMax(res)
    s3_convBN, z3_convBN = cal_scaleZeroPoint(r_max=best_maxx_res, r_min=best_minn_res)

    Qres = Quantization(res, s3_convBN, z3_convBN)
    DEQres = Dequantization(Qres, s3_convBN, z3_convBN)

    return DEQres


def compute_missing(model, windows):
    similarities = []
    weight1, weight2, weight3, weight4, weight5 = [], [], [], [], []
    bn_params1, bn_params2, bn_params3, bn_params4, bn_params5 = [], [], [], [], []
    dense_params, dense_bias = [], []
    for layer in model.layers:
        if layer.name == 'conv1d_1':
            weight1 = np.array(layer.get_weights())
        if layer.name == 'conv1d_2':
            weight2 = np.array(layer.get_weights())
        if layer.name == 'conv1d_3':
            weight3 = np.array(layer.get_weights())
        if layer.name == 'conv1d_4':
            weight4 = np.array(layer.get_weights())
        if layer.name == 'conv1d_5':
            weight5 = np.array(layer.get_weights())
        if layer.name == 'batch_normalization_1':
            bn_params1 = np.array(layer.get_weights())
        if layer.name == 'batch_normalization_2':
            bn_params2 = np.array(layer.get_weights())
        if layer.name == 'batch_normalization_3':
            bn_params3 = np.array(layer.get_weights())
        if layer.name == 'batch_normalization_4':
            bn_params4 = np.array(layer.get_weights())
        if layer.name == 'batch_normalization_5':
            bn_params5 = np.array(layer.get_weights())
        if layer.name == 'dense_1':
            dense_params = np.array(layer.get_weights()[0])
            dense_bias = np.array(layer.get_weights()[1])
    """conv_BN_relu1"""
    w_folding, w_folding_bias = Folding_Conv_BN(weight1, bn_params1)
    res1 = tf.nn.conv1d(windows, w_folding[0], stride=1, padding='VALID') + w_folding_bias
    res1 = tf.nn.relu(res1)
    """conv_BN_relu2"""
    w2_folding, w2_folding_bias = Folding_Conv_BN(weight2, bn_params2)
    res2 = tf.nn.conv1d(res1, w2_folding[0], stride=2, padding='VALID') + w2_folding_bias
    res2 = tf.nn.relu(res2)

    """conv_BN_relu3"""
    w3_folding, w3_folding_bias = Folding_Conv_BN(weight3, bn_params3)
    res3 = tf.nn.conv1d(res2, w3_folding[0], stride=2, padding='VALID') + w3_folding_bias
    res3 = tf.nn.relu(res3)
    """conv_BN_relu4"""
    w4_folding, w4_folding_bias = Folding_Conv_BN(weight4, bn_params4)
    res4 = tf.nn.conv1d(res3, w4_folding[0], stride=2, padding='VALID') + w4_folding_bias
    res4 = tf.nn.relu(res4)
    """conv_BN_relu5"""
    w5_folding, w5_folding_bias = Folding_Conv_BN(weight5, bn_params5)
    res5 = tf.nn.conv1d(res4, w5_folding[0], stride=2, padding='VALID') + w5_folding_bias
    res5 = tf.nn.relu(res5)
    """dense"""
    gp = tf.reshape(res5, [res5.shape[0], res5.shape[2], res5.shape[3]])
    gp = tf.keras.layers.GlobalAveragePooling1D()(gp)
    res_dense = np.matmul(gp, dense_params) + dense_bias

    """DEQ(Q())"""
    DEQres_q_deq = test_conv(windows, w_folding, w_folding_bias, stride=1)
    cos_Similarity = cosineSimilarity(res1, DEQres_q_deq)
    similarities.append(np.mean(cos_Similarity))

    DEQres_q_deq = test_conv(DEQres_q_deq, w2_folding, w2_folding_bias)
    cos_Similarity = cosineSimilarity(res2, DEQres_q_deq)
    similarities.append(np.mean(cos_Similarity))

    DEQres_q_deq = test_conv(DEQres_q_deq, w3_folding, w3_folding_bias)
    cos_Similarity = cosineSimilarity(res3, DEQres_q_deq)
    similarities.append(np.mean(cos_Similarity))

    DEQres_q_deq = test_conv(DEQres_q_deq, w4_folding, w4_folding_bias)
    cos_Similarity = cosineSimilarity(res4, DEQres_q_deq)
    similarities.append(np.mean(cos_Similarity))

    DEQres_q_deq = test_conv(DEQres_q_deq, w5_folding, w5_folding_bias)
    cos_Similarity = cosineSimilarity(res5, DEQres_q_deq)
    similarities.append(np.mean(cos_Similarity))

    DEQres_q_deq = tf.reshape(DEQres_q_deq, [res5.shape[0], res5.shape[2], res5.shape[3]])
    DEQres_q_deq = tf.keras.layers.GlobalAveragePooling1D()(DEQres_q_deq)
    DEQres_q_deq = test_dense(DEQres_q_deq, dense_params, dense_bias)
    cos_Similarity = cosineSimilarity(res_dense, DEQres_q_deq)
    similarities.append(np.mean(cos_Similarity))
    print(similarities)





if __name__ == '__main__':
    path = 'OapNet/test/1100920_test_(J&W&D&j&in0)/3-7-5-11/TD20181001-152951_(Johny)_H50_N4_K3-7-5-11.txt'
    model_pairNet = load_model("PairNet/model/pairnet_model16_12_20220503.h5")
    print(dir(model_pairNet))
    for layer in model_pairNet.layers:
        print(layer.name)
        print(np.array(layer.get_weights()).shape)
        print(layer.get_weights())
    # windows = make_window_siginals(path)
    # compute_missing(model_pairNet, windows)
    # plt_weights(model_pairNet)
    # plt_siginal(windows)
