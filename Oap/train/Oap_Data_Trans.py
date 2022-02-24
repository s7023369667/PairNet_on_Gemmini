import os, glob
import math
import numpy as np

# TODO ：標記單一手勢的起點終點
"""
從手機上收集的資料為未標記起點終點的原資料
請注意原資料以及轉換後的轉速度單位不同
本檔會將原資料轉換成有標記起點終點的檔案
依據[資料加速度的變化量]從資料前方開始找起點
依據[資料加速度的變化量]從資料後方開始找終點
在原資料前加上24個資料點(給Sliding window對齊用)   
###輸入長50，輸出長1，假設輸出時間點是t，那sliding window取t-24到t+25的sample
去掉終點+30後的資料(減少資料量)
在最後一行加[起點位置,終點位置,手勢種類]
"""
'''
設定原始資料的資料夾路徑
'''
# train_data_path = "./train/1071101_Johny[5]&Wen[5]_train_New12(J&W)"
train_data_path = '../float_data[-1,1]'
"""
設定轉換後檔案的檔案路徑
之後程式會在檔案名後加'數字.txt'
"""
trans_path = train_data_path + '_trans'
if not os.path.exists(trans_path):
    os.mkdir(trans_path)


# 讀檔案位址與label
def get_files_OaP(data_path):
    # TODO Get file_name from given path output as labeling data
    clean_files_path = []  # a list to store all text's [dir_path, label]
    n = 25  # 每個手勢取幾筆
    for label in os.listdir(data_path):
        label_dir = data_path + '/' + label
        print(label_dir)
        N = 0

        for txt in glob.glob(label_dir + '/*.txt'):
            clean_files_path.append(txt)
            N += 1
            if (N == n):
                break
    print(clean_files_path)
    return clean_files_path


# 開檔&讀取至list
def get_samples_OaP(train_file, trans_path):
    '''
        簡單來說這邊是計算加速度變化，為了避免中心點不會位移，但只要有旋轉會導致重力影響到加速度的變化，所以用了wt計算轉動量，
        並將sample點0的加速度值假設為重力加速度方向，然後根據wt轉動重力加速度的方向，然後與原資料相減就是手機揮動的加速度。
    '''
    # TODO Get training samples
    gesN = 0  # 前面去除幾個手勢
    PI = math.pi
    deg = PI / 180
    f = 50  # 取樣頻率
    Num = 0

    for file in train_file:  # i為clean_files內之一筆[ ,]
        fileName = file.split('/')[-3]
        label = file.split('/')[-2]
        Num += 1
        train_data = []
        train_label = []
        try:
            with open(file) as txt_file:
                sample, tmp = [], []
                for line in txt_file:
                    s = line.split(' ')  # 每一行以' '分割
                    sample.append(list(map(float, s[0:6])))
                sample = np.array(sample)
                sensor = [[0.0] * len(sample) for i in range(6)]
                w = [[0.0] * len(sample) for i in range(3)]
                wt = [[0.0] * len(sample) for i in range(3)]
                g = [[0.0] * len(sample) for i in range(3)]
                Acc = [[0.0] * len(sample) for i in range(3)]
                aaa = [0.0 for i in range(len(sample))]
                for i in range(len(sample)):
                    for j in range(3):
                        sensor[j][i] += sample[i][j]
                        sample[i][j + 3] = sample[i][j + 3] * deg
                        sensor[j + 3][i] += sample[i][j + 3]
                        w[j][i] += sample[i][j + 3]
                for i in range(1, len(sample)):  # w積分=wt
                    for j in range(3):
                        wt[j][i] += (0.95 * (w[j][i - 1] + w[j][i]) / 2) / f + wt[j][i - 1]  # 梯形面積
                tmpg = [0.0 for i in range(3)]
                for i in range(5):  # 初始重力
                    tmpg[0] += sensor[0][i]
                    tmpg[1] += sensor[1][i]
                    tmpg[2] += sensor[2][i]
                tmpg[0] = tmpg[0] / 5
                tmpg[1] = tmpg[1] / 5
                tmpg[2] = tmpg[2] / 5
                for i in range(1, len(sample)):  # 旋轉矩陣
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
                        Acc[j][i] = sensor[j][i] - g[j][i]
                    tmp = Acc[0][i] * Acc[0][i] + Acc[1][i] * Acc[1][i] + Acc[2][i] * Acc[2][i]
                    aaa[i] = math.sqrt(tmp)
                tmp = 0
                for i in range(5):
                    tmp += aaa[i + 1]
                tmp /= 5
                for i in range(7, len(sample)):  # 去除重力 找加速度變化
                    if abs(aaa[i] - tmp) > 1.11:
                        PF = i
                        break
                tmp = 0
                for i in range(10):
                    tmp += aaa[len(sample) - 1 - i]
                tmp /= 10
                for i in range(len(sample) - 12, 0, -1):
                    if abs(aaa[i] - tmp) > 1.11:
                        PS = i
                        break
                """25hz"""
                '''tmp=[]
                for i in range(len(sample)):
                    if(i%2==0):
                        tmp.append(sample[i])
                tmp=tmp[0:2]*6+tmp
                PS=(int)(SP/2)
                PS+=2
                PF = (int)(PF / 2)
                PF+=2
                seq=tmp[:PF+20]
                seq=np.array(seq)'''
                """50hz"""
                sample = list(sample)
                ##TODO 前面補24個資料 讓輸出和原本等長
                sample = sample[0:2] * 12 + sample  # 補
                PS += 24  # 補後位移
                PF += 24
                seq = sample[:PF + 30]
                seq = np.array(seq)
                tmp = []
                tmp.append(PS)
                tmp.append(PF)
                tmp.append((int(label)) - gesN)  # 中點的手勢種類
                train_data.append(seq)
                train_label.append(tmp)
        except:
            print("Error file path:", file)
            # os.remove(file)
            # pass
        qq = ""
        for i in range(len(train_data)):
            for j in range(len(train_data[i])):
                qq = qq + " ".join(map(str, train_data[i][j]))
                qq = qq + '\n'
        OutNum = str(Num)
        savefile_path = trans_path + '/' + fileName + f'_{label}_' + OutNum + '.txt'
        with open(savefile_path, 'w') as fw:
            fw.write(qq)
            for i in range(len(train_label)):
                fw.write(" ".join(map(str, train_label[i])))


if __name__ == '__main__':  # main
    clean_files_path = get_files_OaP(train_data_path)
    get_samples_OaP(clean_files_path, trans_path)