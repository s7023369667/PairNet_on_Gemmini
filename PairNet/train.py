from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from preprocessing_utils import *
from preprocessing import *
import time
from model_pairnet import *


def get_files(path, gesN):
    """
        Get file_name from given path
    """
    clean_files = []
    # .DS_Store 來自於Mac OS上，用來儲存這個文件夾顯示的屬性
    # Image - 資料視覺化的圖片
    for subdir in listdir(path):
        if eval(subdir) > gesN - 1:
            ##gesN = 9 + 1
            continue
        if '.DS_Store' in subdir or 'Image' in subdir or 'label' in subdir:
            continue
        for i in listdir(path + '/' + subdir):
            if '.DS_Store' in i or 'Image' in i or 'label' in i:
                continue
            clean_files.append((path + '/' + subdir + '/' + i, subdir))
    # print('file - ', clean_files)
    return clean_files


def process_data(file_name, train, train_label):
    """
        Process samples (處理單一個檔案)
    """
    # raw - class 'str'
    raw = None
    # print('0 -', file_name[0])
    with open(file_name[0]) as raw_file:
        raw = raw_file.read()
    seq = []
    l = []
    dim = 6

    '''
      原本收集到的訓練手勢中，最後都會接 50筆idle 用的資料
      加上最後有一行 '' ，所以總共要扣掉 51行
    '''
    raw_list = raw.split('\n')
    raw_list = raw_list[:-51]

    # 去掉目錄的路徑，回傳文件名稱
    # train_all_11/1  ->  1
    labels = os.path.basename(os.path.split(file_name[0])[0])
    # Input gesture is 'str' type, use '\n' to split text
    for i in raw_list:
        if i.split(' ')[-1] != '0':  # train_all_11_by107
            try:
                l.append(int(labels))
            except:
                print('Train Label Append Error')
                pass

            try:
                seq.append(list(map(float, i.split(' '))))
            except:
                print('train set Append Error')
                pass

    seq = list(filter(None, seq))
    train.append(seq)
    train_label.append(list(filter(None, l)))
    return train, train_label


def get_samples(train_file):
    """
        Get training / validation samples
    """
    train_data = []
    train_label = []

    # global_count = 0
    for i in train_file:
        # i (class 'tupe') -> ('train_all_11/1/SensorData_2017_11_23_152125.txt', '1')
        train_data, train_label = process_data(i, train_data, train_label)

    return train_data, train_label


def preprocessing(train, train_label, gesN):
    x, y = [], []
    length = 50
    dim = 6
    out_dim = gesN

    for i in range(len(train)):  # train裡面放的是所有資料(扣掉最後50筆)
        if len(train[i]) != len(train_label[i]) or len(train[i]) == 0:
            continue

        # over-lapping 如同測試集的做法 -> 實驗結過不理想
        l = len(train[i])

        for j in range(0, (l - length + 1), length):
            x.append(train[i][j:j + length])
            y.append(to_categorical(train_label[i][j:j + length], out_dim))

    c = list(zip(x, y))
    random.shuffle(c)
    x, y = zip(*c)

    x = np.array(x).reshape((len(x), length, dim)).astype(float)
    y = np.array(y).reshape((len(y), length, out_dim)).astype(int)

    """
      切割成 訓練集 / 驗證集
    """
    val_split_len = int(len(x) * 0.75)

    x_train = x[:val_split_len]  # (val_split_len) %
    y_train = y[:val_split_len]

    x_test = x[val_split_len:]  # (1 - val_split_len) %
    y_test = y[val_split_len:]

    return x_train, y_train[:, -1, :], x_test, y_test[:, -1, :]


def train():
    Model_Name_Now_Time = time.strftime("%Y%m%d", time.localtime())
    window_size = 50
    gesN = 12
    channel = 16
    Model_HDF5_name = f'./model/pairnet_noRelu_model{channel}_{gesN}_{Model_Name_Now_Time}.h5'

    training_path = '../Oap/train/train_raw/1071101_Johny[5]&Wen[5]_train_New12(J&W)'
    train, train_label = get_samples(get_files(training_path, gesN))

    x_train, y_train, x_val, y_val = preprocessing(train, train_label,gesN)

    # model = build_model(window_size, 6, gesN, channel)
    model = build_model(window_size, 6, gesN, channel)
    model.summary()
    lr = ReduceLROnPlateau(patience=5, factor=0.4, min_delta=0.0001, min_lr=0.00001, verbose=1)
    c = ModelCheckpoint(Model_HDF5_name, monitor='val_accuracy', verbose=0, save_best_only=True, period=1)
    model.fit(x_train, y_train, batch_size=32, epochs=150, validation_data=(x_val, y_val), callbacks=[c, lr], verbose=2)
    K.clear_session()


if __name__ == '__main__':
    train()
