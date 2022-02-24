import numpy as np
import random, math, os
from keras.utils.np_utils import to_categorical

# TODO 資料增量 & 標記關鍵區間PKI和SKI
"""
本檔為[資料增量]以及[標記關鍵區間]
有各式各樣尚未使用的函數沒有要用資料增量就不用看
標記關鍵區間請至函數Data_Aug
資料增量的部分也是至函數Data_Aug
調整完後請儲存並執行OaP_train訓練模型
"""


def GD(x, u, idx):  # 高斯分布
    if idx < 5:
        return math.exp(-0.5 * math.pow(x - u, 2) / 5)  # var=5 自訂
    else:
        return math.exp(-0.5 * math.pow(x - u, 2) / 5)  # var=5 自訂


def Interval(x, u, idx, len):  # 以u為中點 x為變數 展開半徑len的區間 區間內為1 超過len的GD
    if abs(x - u) <= len:
        return 1
    elif x - u < 0:
        return GD(x, u - len, idx)  # 現在的中點為u-len
    else:
        return GD(x, u + len, idx)  # 現在的中點為u+len


def Ran(data, rana, ranb, ranr):  # a旋轉X軸 b旋轉Y軸 r旋轉Z軸
    # TODO 每個sample點隨機選轉
    tmp = [[0.0] * 6 for i in range(len(data))]
    tmp[0][3] = data[0][3]
    tmp[0][4] = data[0][4]
    tmp[0][5] = data[0][5]
    tmpa = 0
    tmpb = 0
    tmpr = 0
    PI = math.pi
    for i in range(0, len(tmp)):
        a = tmpa + random.uniform(-rana, rana)
        b = tmpb + random.uniform(-ranb, ranb)
        r = tmpr + random.uniform(-ranr, ranr)
        tmp[i][0] = (math.cos(r) * math.cos(b) * data[i][0]) + \
                    (-1 * math.sin(r) * math.cos(a) + math.cos(r) * math.sin(b) * math.sin(a)) * data[i][1] + \
                    (math.sin(r) * math.sin(a) + math.cos(r) * math.sin(b) * math.cos(a)) * data[i][2]
        tmp[i][1] = (math.sin(r) * math.cos(b) * data[i][0]) + \
                    (math.cos(r) * math.cos(a) + math.sin(r) * math.sin(b) * math.sin(a)) * data[i][1] + \
                    (-1 * math.cos(r) * math.sin(a) + math.sin(r) * math.sin(b) * math.cos(a)) * data[i][2]
        tmp[i][2] = (-1 * math.sin(b) * data[i][0]) + \
                    (math.cos(b) * math.sin(a)) * data[i][1] + \
                    (math.cos(b) * math.cos(a)) * data[i][2]
        if (i == 0):
            tmpa = a
            tmpb = b
            tmpr = r
            continue
        dela = a - tmpa
        delb = b - tmpb
        delr = r - tmpr
        # 角度變化= t * (w(t-1)+w(t))/2
        # 角*2/t -w(t-1) = w(t)  t=0.02 (f=50)
        tmp[i][3] = (dela * 100) - tmp[i - 1][3]
        tmp[i][4] = (delb * 100) - tmp[i - 1][4]
        tmp[i][5] = (delr * 100) - tmp[i - 1][5]
        for j in range(3, 6):
            if tmp[i][j] > PI:
                tmp[i][j] -= 2 * PI
            elif tmp[i][j] < -PI:
                tmp[i][j] += 2 * PI
        tmpa = a
        tmpb = b
        tmpr = r
    return tmp


def Rotate(data, a, b, r):  # a旋轉X軸 b旋轉Y軸 r旋轉Z軸
    # TODO 統一轉一個角度
    tmp = [[0.0] * 6 for i in range(len(data))]
    for i in range(0, len(tmp)):
        tmp[i][0] = (math.cos(r) * math.cos(b) * data[i][0]) + \
                    (-1 * math.sin(r) * math.cos(a) + math.cos(r) * math.sin(b) * math.sin(a)) * data[i][1] + \
                    (math.sin(r) * math.sin(a) + math.cos(r) * math.sin(b) * math.cos(a)) * data[i][2]
        tmp[i][1] = (math.sin(r) * math.cos(b) * data[i][0]) + \
                    (math.cos(r) * math.cos(a) + math.sin(r) * math.sin(b) * math.sin(a)) * data[i][1] + \
                    (-1 * math.cos(r) * math.sin(a) + math.sin(r) * math.sin(b) * math.cos(a)) * data[i][2]
        tmp[i][2] = (-1 * math.sin(b) * data[i][0]) + \
                    (math.cos(b) * math.sin(a)) * data[i][1] + \
                    (math.cos(b) * math.cos(a)) * data[i][2]
        tmp[i][3] = data[i][3]
        tmp[i][4] = data[i][4]
        tmp[i][5] = data[i][5]
    return tmp


def Rotate_t(data, a, b, r):  # a旋轉X軸 b旋轉Y軸 r旋轉Z軸
    # TODO 隨時間選轉成我要的角度
    l = len(data) - 1
    tmpa = a / l
    tmpb = b / l
    tmpr = r / l
    tmp = [[0.0] * 6 for i in range(l + 1)]
    for i in range(0, l + 1):
        a = i * tmpa
        b = i * tmpb
        r = i * tmpr
        tmp[i][0] = (math.cos(r) * math.cos(b) * data[i][0]) + \
                    (-1 * math.sin(r) * math.cos(a) + math.cos(r) * math.sin(b) * math.sin(a)) * data[i][1] + \
                    (math.sin(r) * math.sin(a) + math.cos(r) * math.sin(b) * math.cos(a)) * data[i][2]
        tmp[i][1] = (math.sin(r) * math.cos(b) * data[i][0]) + \
                    (math.cos(r) * math.cos(a) + math.sin(r) * math.sin(b) * math.sin(a)) * data[i][1] + \
                    (-1 * math.cos(r) * math.sin(a) + math.sin(r) * math.sin(b) * math.cos(a)) * data[i][2]
        tmp[i][2] = (-1 * math.sin(b) * data[i][0]) + \
                    (math.cos(b) * math.sin(a)) * data[i][1] + \
                    (math.cos(b) * math.cos(a)) * data[i][2]
        if i == 0:
            tmp[i][3] = data[i][3]
            tmp[i][4] = data[i][4]
            tmp[i][5] = data[i][5]
            continue
        tmp[i][3] = data[i][3] + 50 * tmpa
        tmp[i][4] = data[i][4] + 50 * tmpb
        tmp[i][5] = data[i][5] + 50 * tmpr
    return tmp


def linear(data, f, mx, my, mz, cx, cy, cz, t1, t2):  # t1到t2 data增加m(t-t1)+c
    # 可能沒用
    tmp = np.array([[0.0] * 6] * len(data))
    tmp[:][:] = data[:][:]
    for i in range(t1, t2):
        tmp[i][0] += mx * (i - t1) / f + cx
        tmp[i][1] += my * (i - t1) / f + cy
        tmp[i][2] += mz * (i - t1) / f + cz
    return tmp


def Sine(data, Ax, Ay, Az, fx, fy, fz, t1, t2):  # t1到t2 data增加
    # 可能沒用
    if (Ax == 0) & (Ay == 0) & (Az == 0):
        return data

    tmp = np.array([[0.0] * 6] * len(data))
    tmp[:][:] = data[:][:]
    A = np.sqrt(Ax * Ax + Ay * Ay + Az * Az)
    Ax /= A
    Ay /= A
    Az /= A
    Pi = math.pi
    l = t2 - t1
    for i in range(t1, t2):
        tmp[i][0] += Ax * math.sin(2 * Pi * fx * (i - t1) / l)
        tmp[i][1] += Ay * math.sin(2 * Pi * fy * (i - t1) / l)
        tmp[i][2] += Az * math.sin(2 * Pi * fz * (i - t1) / l)
    return tmp


def FT(data, f, A, HL):  # 對頻率f以上或以下*A倍
    data = np.array(data)
    sft = np.array([[0.0] * len(data)] * 6)
    re = np.array([[0.0] * 6] * len(data))
    for i in range(6):
        sft[i, :] = data[:, i]
    sft = np.fft.fft(sft)
    sft = np.fft.fftshift(sft)
    MP = int(len(sft[0]) / 2)

    if HL == 0:
        sft[:, : (MP - f)] *= A  # 處理高頻
        sft[:, (MP + f):] *= A
    elif HL == 1:
        sft[:, (MP - f):(MP + f)] *= A  # 處理低頻
    else:
        return data
    sft = np.fft.ifftshift(sft)
    sift = np.fft.ifft(sft)
    for i in range(6):
        re[:, i] = sift[i, :]
    return re


def Data_Aug(file, WindowSize, gesN, stride_size):
    train = []
    train_label = []
    PI = math.pi
    deg = PI / 180
    RR = deg / 2
    Ro = 30 * deg
    Ro2 = 2 * Ro
    length = WindowSize
    '''增量處理'''
    for txt in os.listdir(file):
        qqq = []
        if ".txt" in txt:
            with open(os.path.join(file, txt), 'r') as fr:
                for line in fr:
                    s = line.split()  # 每一行以' '分割
                    qqq.append(list(map(eval, s)))
            fr.close()
        train.append(qqq[:-1])
        train_label.append(qqq[-1])

        PS = qqq[-1][0]
        PF = qqq[-1][1]
        Radius = (PF - PS) / 2

        '''使用函數增量'''
        '''tmp = FT(qqq[:-1],int(len(qqq)/10),3,0)
        train.append(tmp)
        train_label.append(qqq[-1])

        tmp = FT(qqq[:-1], int(len(qqq) / 10), 0, 0)
        train.append(tmp)
        train_label.append(qqq[-1])

        tmp = FT(qqq[:-1], int(len(qqq) / 20), 1.2, 1)
        train.append(tmp)
        train_label.append(qqq[-1])

        tmp = FT(qqq[:-1], int(len(qqq) / 20), 1.5, 1)
        train.append(tmp)
        train_label.append(qqq[-1])'''

        """for i in range(3):
            for j in range(3):
                for k in range(3):
                    '''tmp = linear(qqq[:-1],50,
                                 (i - 1) * 1.2, (j - 1) * 1.2, (k - 1) * 1.2,
                                 (i - 1) * 0.6, (j - 1) * 0.6, (k - 1) * 0.6,
                                 0,
                                 int(SP + 0.2 * Radius)
                                 )
                    tmp = linear(tmp,50,
                                 (i - 1) * 1.2, (j - 1) * 1.2, (k - 1) * 1.2,
                                 (i - 1) * 0.6, (j - 1) * 0.6, (k - 1) * 0.6,
                                 int(FP - 0.2 * Radius),
                                 len(tmp)
                                 )
                    train.append(tmp)
                    train_label.append(qqq[-1])'''
                    '''tmp = Sine(qqq[:-1],
                               (i - 1) * 1.6, (j - 1) * 1.6, (k - 1) * 1.6,
                               3.0, 3.0,  3.0,
                               int(SP - 0.2 * Radius),
                               int(FP + 0.2 * Radius)
                              )
                    tmp = Sine(tmp,
                               (i - 1) * 1.6, (j - 1) * 1.6, (k - 1) * 1.6,
                               1.0, 1.0, 1.0,
                               int(SP - 0.2 * Radius),
                               int(FP + 0.2 * Radius)
                               )
                    tmp = Sine(tmp,
                               (i - 1) * 1.6, (j - 1) * 1.6, (k - 1) * 1.6,
                               4.0, 4.0, 4.0,
                               int(SP - 0.2 * Radius),
                               int(FP + 0.2 * Radius)
                               )
                    tmp = Sine(tmp,
                               (i - 1) * 0.6, (j - 1) * 0.6, (k - 1) * 0.6,
                               7.0, 7.0, 7.0,
                               int(SP - 0.2 * Radius),
                               int(FP + 0.2 * Radius)
                               )
                    tmp = Sine(tmp,
                               (i - 1) * 0.6, (j - 1) * 0.6, (k - 1) * 0.6,
                               11.0, 11.0, 11.0,
                               int(SP - 0.2 * Radius),
                               int(FP + 0.2 * Radius)
                               )
                    train.append(tmp)
                    train_label.append(qqq[-1])'''

                    '''tmp = Sine(qqq[:-1],
                               (i - 1) * 6.0, (j - 1) * 6.0, (k - 1) * 6.0,
                               1.0, 1.0, 1.0,
                               int(SP - 0.2 * Radius),
                               int(SP + 0.8 * Radius)
                               )
                    tmp = Sine(tmp,
                               (i - 1) * 6.0, (j - 1) * 6.0, (k - 1) * 6.0,
                               1.0, 1.0, 1.0,
                               int(FP -0.8 * Radius),
                               int(FP +0.2 * Radius)
                              )
                    train.append(tmp)
                    train_label.append(qqq[-1])'''"""

    '''增量處理完'''
    '''
    標記調整 
    
    基本上只要調整S F R 以及輸入至Interval()內的半徑長
    
    '''
    # TODO 標記關鍵區間PKI和SKI
    x, y, tmp = [], [], []
    dim = 6
    out_dim = gesN + 1
    for i in range(len(train)):
        '''for j in range(len(train[i])): #弧度比例調整
            train[i][j][3]*=3
            train[i][j][4] *= 3
            train[i][j][5] *= 3'''
        label_MI = to_categorical(np.array([0.0] * (len(train[i]))), out_dim)  # len(train[i])),2
        label_PKI = to_categorical(np.array([0.0] * (len(train[i]))), out_dim)
        label_SKI = to_categorical(np.array([0.0] * (len(train[i]))), out_dim)
        PS = train_label[i][0]
        PF = train_label[i][1]
        idx = int(train_label[i][2])
        MP = int((PS + PF) / 2)
        Radius = (PF - PS) / 2

        if idx <= (11 - gesN):
            # 如果是背景手勢跳過此檔
            continue
        else:
            idx -= (11 - gesN)
            pk = int(Radius * 0.4)  # 從中心點往前推至第一關鍵中心
            sk = int(Radius * 0.4)  # 從中心點往後推至第二關鍵中心
            R = int(Radius * 1.0)  # Regulation範圍
            for j in range(len(label_MI)):  # label interval
                # j=sample點,idx=手勢種類
                label_MI[j][idx] = Interval(j, MP, idx, R)
                label_MI[j][0] = 1 - label_MI[j][idx]
                # label_PKI = PKI
                label_PKI[j][idx] = Interval(j, MP - pk, idx, int(Radius * 0.10))  # key interval region
                label_PKI[j][0] = 1 - label_PKI[j][idx]
                # label_PF = SKI
                label_SKI[j][idx] = Interval(j, MP + sk, idx, int(Radius * 0.10))  # key interval region
                label_SKI[j][0] = 1 - label_SKI[j][idx]

        l = len(train[i])
        for j in range(0, l - length + 1, stride_size):  # 50 windows 切割 ,stride_size自訂(決定資料數)
            x.append(train[i][j:j + length])  # f(x[50]) -> y
            tmp = list(label_MI[j + (int)(length / 2)])
            tmp.extend(list(label_PKI[j + (int)(length / 2)]))
            tmp.extend(list(label_SKI[j + (int)(length / 2)]))
            # tmp.append(train_label[i][-1])
            # tmp.append(idx)
            y.append(tmp)  # GD*3+len+第幾種手勢 6*3 #+ 1 + 1
    '''標記調整完'''
    '''洗順序'''
    c = list(zip(x, y))

    random.shuffle(c)
    x, y = zip(*c)

    x = np.array(x).reshape((len(x), length, dim)).astype(float)
    y = np.array(y).reshape((len(y), 3 * out_dim)).astype(float)  # 2*3

    # 切割 75% 訓練、25% 驗證
    l = int(len(x) * 0.75)
    # 95%
    # l = int(len(x) * 0.95)
    x_train = x[:l]
    x_test = x[l:]
    y_train = y[:l]
    y_test = y[l:]

    # random
    c = list(zip(x_train, y_train))
    random.shuffle(c)
    x_train, y_train = zip(*c)

    x_train = np.array(x_train).reshape((len(x_train), length, dim))
    x_test = np.array(x_test).reshape((len(x_test), length, dim))
    y_train = np.array(y_train).reshape((len(y_train), 3 * out_dim))  # 2*3
    y_test = np.array(y_test).reshape(len(y_test), 3 * out_dim)  # 2*3

    return x_train, y_train, x_test, y_test
