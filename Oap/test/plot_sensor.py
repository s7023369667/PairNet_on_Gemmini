import math
import numpy as np
import matplotlib.pyplot as plt
from library import *



def plot_sample(txt_path, pre_ges_region=None):
    sample = []
    with open(txt_path) as test_case:
        for line in test_case:
            s = line.split(' ')  # 每一行以' '分割
            sample.append(list(map(float, s[0:6])))  # n,6
    sample = np.array(sample)
    best_minn, best_maxx = optimal_MinMax(sample)
    #
    # # scale, zeroPoint = cal_scaleZeroPoint(r_max=np.max(sample), r_min=np.min(sample))
    # scale, zeroPoint = cal_scaleZeroPoint(r_max=best_maxx, r_min=best_minn)
    # Qsample = Quantization(sample, scale, zeroPoint)
    # DEQsample = Dequantization(Qsample, scale, zeroPoint)
    # cosine_sim = cosineSimilarity(sample, DEQsample)
    # print(np.mean(cosine_sim))
    # sample = DEQsample
    #
    sensor = [[0.0] * len(sample) for i in range(6)]
    w = [[0.0] * len(sample) for i in range(3)]
    wt = [[0.0] * len(sample) for i in range(3)]
    g = [[0.0] * len(sample) for i in range(3)]
    v = [[0.0] * len(sample) for i in range(3)]
    aaa = [0.0 for i in range(len(sample))]
    tmp = 0
    PI = math.pi
    for i in range(len(sample)):
        for j in range(3):
            sensor[j][i] += sample[i][j]
            sample[i][j + 3] = sample[i][j + 3] * PI / 180
            sensor[j + 3][i] += sample[i][j + 3]
            w[j][i] += sample[i][j + 3]
    f = 50  # 取樣頻率
    for i in range(1, len(sample)):
        for j in range(3):
            wt[j][i] += (0.95 * (w[j][i - 1] + w[j][i]) / 2) / f + wt[j][i - 1]
    tmpg = [0.0 for i in range(3)]
    for i in range(5):
        tmpg[0] += sensor[0][i]
        tmpg[1] += sensor[1][i]
        tmpg[2] += sensor[2][i]
    tmpg[0] = tmpg[0] / 5
    tmpg[1] = tmpg[1] / 5
    tmpg[2] = tmpg[2] / 5
    for i in range(1, len(sample)):
        a = -1 * wt[0][i]
        b = -1 * wt[1][i]
        r = -1 * wt[2][i]
        g[0][i] = (math.cos(r) * math.cos(b) * tmpg[0]) + \
                  (-1 * math.sin(r) * math.cos(a) + math.cos(r) * math.sin(b) * math.sin(a)) * tmpg[1] + \
                  (math.sin(r) * math.sin(a) + math.cos(r) * math.sin(b) * math.cos(a)) * tmpg[2]
        g[1][i] = (math.sin(r) * math.cos(b) * tmpg[0]) + \
                  (math.cos(r) * math.cos(a) + math.sin(r) * math.sin(b) * math.sin(a)) * tmpg[1] + \
                  (-1 * math.cos(r) * math.sin(a) + math.sin(r) * math.sin(b) * math.cos(a)) * tmpg[2]
        g[2][i] = (-1 * math.sin(b) * tmpg[0]) + \
                  (math.cos(b) * math.sin(a)) * tmpg[1] + \
                  (math.cos(b) * math.cos(a)) * tmpg[2]

        for j in range(3):
            v[j][i] = sensor[j][i] - g[j][i]
        tmp = v[0][i] * v[0][i] + v[1][i] * v[1][i] + v[2][i] * v[2][i]
        aaa[i] = math.sqrt(tmp)

    tmp = 0
    for i in range(5):
        tmp += aaa[i + 1]
    tmp /= 5
    for i in range(7, len(sample)):
        if abs(aaa[i] - tmp) > 1.11:
            print(i)
            PS = i
            break
    tmp = 0
    for i in range(10):
        tmp += aaa[len(sample) - 1 - i]
    tmp /= 10
    for i in range(len(sample) - 12, 0, -1):
        if abs(aaa[i] - tmp) > 1.11:
            print(i)
            PF = i
            break

    x = range(len(sample))
    y = []
    for i in range(3):
        for j in range(len(sensor[i])):
            y.append(sensor[i][j])
    """ACC"""
    plt.figure(figsize=(20, 5))
    plt.ylabel('Gyro(deg/s), G-Sensor(m/s^2)')
    plt.xlabel('Samples')
    # acc
    plt.plot(x, sensor[0])
    plt.plot(x, sensor[1])
    plt.plot(x, sensor[2])
    # Gyroscope
    plt.plot(x, sensor[3])
    plt.plot(x, sensor[4])
    plt.plot(x, sensor[5])
    # plt.title('DEQ(Q(Original Sensor Data))')
    # plt.axvline(x=PS, color="red", linestyle='--')
    # plt.axvline(x=PF, color="red", linestyle='--')
    # plt.axvline(x=pre_ges_region[0][0] + PS + 1, color="orange", linestyle='--')
    # plt.axvline(x=pre_ges_region[0][1], color="orange", linestyle='--')
    # xx = [pre_ges_region[0][0] + PS + 1, pre_ges_region[0][1], pre_ges_region[0][1], pre_ges_region[0][0] + PS + 1]
    # yy = [np.min(y), np.min(y), np.max(y), np.max(y)]
    # plt.fill(xx, yy, color='orange', alpha=0.3)
    # # plt.axvline(x=pre_ges_region[1][0], color="blue", linestyle='--')
    # # plt.axvline(x=pre_ges_region[1][1], color="blue", linestyle='--')
    # xx = [pre_ges_region[1][0], pre_ges_region[1][1], pre_ges_region[1][1], pre_ges_region[1][0]]
    # plt.fill(xx, yy, color='skyblue', alpha=0.3)
    # plt.axvline(x=PS, color="red", linestyle='--')
    # plt.axvline(x=PF, color="red", linestyle='--')
    # plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    path = './1100920_test_(J&W&D&j&in0)/11-8-10/TD20181107-195426_(Wen)_H50_N3_K11-8-10.txt'
    plot_sample(path)
