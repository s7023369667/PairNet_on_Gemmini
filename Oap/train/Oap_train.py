from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from Oap.train.Oap_Data_Aug import *
from Oap.model_Oap import *
import matplotlib.pyplot as plt

# model 名稱與位址
User = 'SAM'
# model_name = "OaP_" + User + "_in0_s3_1209"
model_name = "PairNet_" + User + "_0103"

output_dir = "../model_h5/"
history_plot_dir = "../model_h5"
# (trans)訓練集位址
# trans_data_path = "../data/training_data/1100312_in0_123_trans"
trans_data_path = 'trans_all'


# --- show_train_history function --- #
def show_train_history(in_history):
    # plot train set accuracy / loss function value ( determined by what parameter 'train' you pass )
    # The type of train_history.history is dictionary (a special data type in Python)

    plt.subplot(2, 1, 1)
    plt.title("model accuracy")
    plt.plot(in_history.history["output_1_accuracy"])
    plt.plot(in_history.history["output_2_accuracy"])
    plt.plot(in_history.history["output_3_accuracy"])
    plt.plot(in_history.history["val_output_1_accuracy"])
    plt.plot(in_history.history["val_output_2_accuracy"])
    plt.plot(in_history.history["val_output_3_accuracy"])
    plt.ylabel("acc")
    plt.xlabel("Epoch")
    # Places a legend on the place you set by loc
    plt.legend(['Regularization_train', 'PKI_train', 'SKI_train', 'Regularization_validation', 'PKI_validation',
                'SKI_validation'], loc='lower right')

    plt.subplot(2, 1, 2)
    plt.title("model loss")
    plt.plot(in_history.history["output_1_loss"])
    plt.plot(in_history.history["output_2_loss"])
    plt.plot(in_history.history["output_3_loss"])
    plt.plot(in_history.history["val_output_1_loss"])
    plt.plot(in_history.history["val_output_2_loss"])
    plt.plot(in_history.history["val_output_3_loss"])
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    # Places a legend on the place you set by loc
    plt.legend(['Regularization_train', 'PKI_train', 'SKI_train', 'Regularization_validation', 'PKI_validation',
                'SKI_validation'], loc='upper right')

    plt.tight_layout()

    plt.savefig(history_plot_dir + '/' + model_name + '.png')
    plt.show()


# ---------- main ---------- #
if __name__ == "__main__":
    gesN = 2  # anchor手勢號碼大於(11-gesN)是前景
    WindowSize = 50
    stride_size = 1  # 在取訓練資料時 移動Sliding window的步長 設大一點可以減少資料量
    # read train file & pre_processing
    # Data_Aug from OaP_Data_Aug
    x_train_OaP, y_train_OaP, x_test_OaP, y_test_OaP = Data_Aug(trans_data_path, WindowSize, gesN, stride_size)
    # train_OaP, train_label_OaP = get_samples_OaP(get_files_OaP(train_data_path))
    # x_train_OaP, y_train_OaP, x_test_OaP, y_test_OaP = PrePro_OaP(train_OaP, train_label_OaP)
    y_train_MP = y_train_OaP[:, :(gesN + 1)]
    # print(y_train_MP.shape)
    y_train_SP = y_train_OaP[:, (gesN + 1):2 * (gesN + 1)]
    # print(y_train_SP.shape)
    y_train_FP = y_train_OaP[:, 2 * (gesN + 1):3 * (gesN + 1)]
    # print(y_train_FP.shape)
    # y_train_len = y_train_OaP[:,-2]
    # y_train_ges = y_train_OaP[:,-1]
    # y_train_ges = to_categorical(y_train_ges, 2)
    y_test_MP = y_test_OaP[:, :(gesN + 1)]
    y_test_SP = y_test_OaP[:, (gesN + 1):2 * (gesN + 1)]
    y_test_FP = y_test_OaP[:, 2 * (gesN + 1):3 * (gesN + 1)]
    # y_test_len = y_test_OaP[:,-2]
    # y_test_ges = y_test_OaP[:,-1]
    # y_test_ges=to_categorical(y_test_ges, 2)
    #
    # print(x_train_OaP.shape)  # Train
    # print(y_train_OaP.shape)  # Train label
    # print(x_test_OaP.shape)  # validate
    # print(y_test_OaP.shape)  # validate label

    # model build & fit
    model = build_model(WindowSize, 6, (gesN + 1))  # (50, 6, gesN+1)
    lr = ReduceLROnPlateau(patience=5, factor=0.4, min_delta=0.0001, min_lr=0.00001, verbose=1)
    c = ModelCheckpoint(output_dir + '/best_' + model_name + ".h5", verbose=0, save_best_only=True, period=1)

    train_history = model.fit(x_train_OaP, [y_train_MP, y_train_SP, y_train_FP], batch_size=64, epochs=20, verbose=2,
                              callbacks=[c, lr], validation_data=(x_test_OaP, [y_test_MP, y_test_SP, y_test_FP]))
    model.save(output_dir + '/' + model_name + ".h5")

    # plot fit history
    # show_train_history(train_history)
