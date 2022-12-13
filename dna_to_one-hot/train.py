from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import time
import keras
from data_pre import data_preprocess
from my_lstm_with_att import bi_lstm_att
from lstm_config import Config

config = Config()


# config = LSTM_CONFIG()
# config = BILSTM_CONFIG()
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.process_time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.process_time() - self.epoch_time_start)


def train_and_eval(model, X_train, y_train, X_test, y_test):
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    # sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
    # with tf.Session(config=sess_config) as sess:
    #     # 初始化变量值
    #     sess.run(tf.global_variables_initializer())
    #     current_step = 0
    # CREATE CALLBACKS
    save_path = "test_model"
    model_name = "ir_bi_lstm"
    # checkpoint = callbacks.ModelCheckpoint(save_path + model_name + "_random_" + ".h5",
    #                                        # save_path + model_name + "_random_" + str(fold_var) + ".h5"
    #                                        monitor='val_loss', verbose=1,
    #                                        save_best_only=True, mode='min')
    time_callback = TimeHistory()
    history = model.fit(X_train, y_train,
                        epochs=config.epoch,
                        # callbacks=[checkpoint, time_callback],
                        validation_data=(X_test, y_test), batch_size=config.batch_size)
    return history


def paint(history):
    plt.figure(figsize=(10, 5))
    plt.suptitle('Dataset X3 for BI-LSTM-ATT')

    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'], label='train acc')  # acc最新版keras已经无法使用
    plt.plot(history.history['val_acc'], label='val acc')  # val_acc最新版keras已经无法使用
    plt.title('Accuracy')  # 图名
    plt.ylabel('Accuracy')  # 纵坐标名
    plt.xlabel('Epoch')  # 横坐标名
    # plt.legend(['Train', 'Test'])  # 角标及其位置
    plt.legend()
    # 如果不想画在同一副图上，可关闭再打开
    # 绘制训练 & 验证的损失值
    # plt.figure(figsize=(4, 3))
    plt.subplot(1, 2, 2)

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.legend()
    plt.show()


def main():
    # kf = KFold(n_splits=10, shuffle=True)
    # train_path = r'C:\Users\lie\Desktop\trans-for_seqs\text_classifier\data\mydata\seq_train.txt'
    # test_path = r'C:\Users\lie\Desktop\trans-for_seqs\text_classifier\data\mydata\seq_test.txt'
    # X_train, y_train = data_preprocess(train_path)
    # print('X_train shape', X_train.shape)
    X, y = data_preprocess()
    print(len(X), len(y))
    # X = X[:3657]
    # print('X_test shape', X_train.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size,
                                                        random_state=config.random_state)
    # model = model_cycle()
    # model = my_lstm_model(config)
    model = bi_lstm_att(config)
    # model = my_bilstm()
    history = train_and_eval(model, X_train, y_train, X_test, y_test)
    paint(history)
    model.save('bi_lstm_att_X3.h5')


if __name__ == '__main__':
    s = time.time()
    main()
    print('it costs: ', str((time.time() - s) / 60) + ' mins')
