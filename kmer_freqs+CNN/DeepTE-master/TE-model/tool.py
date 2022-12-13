from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, matthews_corrcoef
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from config import Config
config = Config()

config = Config()


def getTrainData(df):
    # df = pd.read_csv(filename)
    print(df.columns)

    # C开头的列代表稀疏特征，I开头的列代表的是稠密特征
    dense_features_col = [col for col in df.columns]

    # 这个文件里面存储了稀疏特征的最大范围，用于设置Embedding的输入维度
    # fea_col = np.load(feafile, allow_pickle=True)
    # sparse_features_col = []
    # for f in fea_col[1]:
    #     sparse_features_col.append(f['feat_num'])
    #
    # data, labels = df.drop(columns='Label').values, df['Label'].values

    return dense_features_col


def DataPreprocess():
    patho = pd.read_csv(config.patho_path, header=None)
    names = []
    for i in range(patho.shape[1]):
        names.append('F' + str(i))
    patho.columns = names
    nonpatho = pd.read_csv(config.nonpatho_path, header=None)
    nonpatho.columns = names
    print('read csv over')

    train_df = pd.concat((patho, nonpatho))
    print(train_df.shape)
    labels = np.concatenate((np.ones(patho.shape[0]), np.zeros(nonpatho.shape[0])))
    # labels = np.concatenate((np.ones(patho.shape[0]), np.zeros(nonpatho.shape[0])))

    print(len(labels))
    # 这一步再用scaler中的均值和方差来转换data，使data数据标准化
    scaler = StandardScaler().fit(train_df)
    scaled = scaler.transform(train_df)
    X = scaled
    # X = train_df

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=config.test_size,
                                                        random_state=config.random_state)
    print(y_train[0:5])
    return train_df, X_train, X_test, y_train, y_test


def testDataPreprocess():
    test_path = config.test_path
    patho = pd.read_csv(test_path)
    names = []
    for i in range(patho.shape[1]):
        names.append('F' + str(i))
    patho.columns = names
    # nonpatho = pd.read_csv(config.nonpatho_path)
    # nonpatho.columns = names
    print('read csv over')

    # train_df = pd.concat((patho, nonpatho))
    print(patho.shape)
    # labels = np.concatenate((np.zeros(patho.shape[0]), np.ones(nonpatho.shape[0])))
    labels = (np.ones(patho.shape[0]))

    print(len(labels))
    scaler = StandardScaler().fit(patho)
    # # np.savetxt(r'D:\bio4\pathtest20\2\data2\scaler.csv', scaler, delimiter=",")  # 不能正常十进制储存
    #
    # scaler.transform(data)
    # 这一步再用scaler中的均值和方差来转换data，使data数据标准化
    scaled = scaler.transform(patho)
    X = scaled
    df_X = patho
    # 进行数据合并，为了同时对train和test数据进行预处理
    # data_df = pd.concat((train_df, test_df))

    # del data_df['Id']
    #
    # print(data_df.columns)
    # X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=414)
    # print(y_train[0:5])
    return df_X, X, labels





def paint(train_loss, test_acc, test_roc, test_f1):
    # acc = history.history['accuracy']  # 训练集准确率
    # val_acc = history.history['val_accuracy']  # 测试集准确率
    # loss = history.history['loss']  # 训练集损失
    # val_loss = history.history['val_loss']  # 测试集损失
    #  打印acc和loss，采用一个图进行显示。
    #  将acc打印出来。

    plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)  # 将图像分为一行两列，将其显示在第一列
    # # plt.plot(train_acc, label='Training Accuracy')
    # plt.plot(test_acc, label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()

    plt.subplot(1, 2, 1)  # 将其显示在第二列
    plt.plot(train_loss, label='Training Loss')
    # plt.plot(test_loss, label='Validation Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)  # 将其显示在第三列
    plt.plot(test_roc, label='Roc-auc score', color='black', linestyle='dotted')
    plt.plot(test_f1, label='F1 score')
    plt.plot(test_acc, label='Accuracy')

    # plt.plot(test_loss, label='Validation Loss')
    plt.title('Validation eval')
    plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    plt.legend()
    plt.show()


def roc(y_ture, y_pred_probs):
    y_test = y_ture.cpu()
    # y_pred_probs = y_pred_probs.cpu()
    y_pred_probs = y_pred_probs.cpu().detach().numpy()
    roc = roc_auc_score(y_test, y_pred_probs)
    print('ROC:', roc)
    return roc


def F1(y_ture, y_pred):
    y_test = y_ture.cpu()
    y_pred = y_pred.cpu()
    f1 = f1_score(y_test, y_pred)
    print('F1 Score: %f' % f1)

    return f1


def mcc(y_ture, y_pred):
    y_test = y_ture.cpu()
    y_pred = y_pred.cpu()
    mcc = matthews_corrcoef(y_test, y_pred)
    print('MCC: %f' % mcc)
    return mcc


def acc(y_ture, y_pred):
    y_test = y_ture.cpu()
    y_pred = y_pred.cpu()

    acc = accuracy_score(y_test, y_pred)
    print('ACC: %f' % acc)
    return acc


def paint_curve(y_test, y_pred_probs, gc_y_test, gc_y_pred_probs):
    # auc
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    fpr1, tpr1, threshold1 = roc_curve(y_test, y_pred_probs[0])
    fpr2, tpr2, threshold2 = roc_curve(y_test, y_pred_probs[1])
    fpr3, tpr3, threshold3 = roc_curve(y_test, y_pred_probs[2])
    fpr4, tpr4, threshold4 = roc_curve(y_test, y_pred_probs[3])
    fpr5, tpr5, threshold5 = roc_curve(y_test, y_pred_probs[4])
    fpr6, tpr6, threshold6 = roc_curve(gc_y_test, gc_y_pred_probs)
    fpr7, tpr7, threshold7 = roc_curve(y_test, y_pred_probs[5])
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    roc_auc3 = auc(fpr3, tpr3)
    roc_auc4 = auc(fpr4, tpr4)
    roc_auc5 = auc(fpr5, tpr5)
    roc_auc6 = auc(fpr6, tpr6)
    roc_auc7 = auc(fpr7, tpr7)

    plt.title('Validation ROC Curve')

    plt.plot(fpr1, tpr1, label='LR(area=%0.3f)' % roc_auc1)
    plt.plot(fpr2, tpr2, label='SVM(area=%0.3f)' % roc_auc2)
    plt.plot(fpr3, tpr3, label='RF(area=%0.3f)' % roc_auc3)
    plt.plot(fpr4, tpr4, label='XGBoost(area=%0.3f)' % roc_auc4)
    plt.plot(fpr5, tpr5, label='AdaBoost(area=%0.3f)' % roc_auc5)
    plt.plot(fpr6, tpr6, label='EC-DFR(area=%0.3f)' % roc_auc6)
    plt.plot(fpr7, tpr7, label='ResDeepCross(area=%0.3f)' % roc_auc7)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()

    # precision and recall
    plt.subplot(1, 2, 2)
    plt.title('Validation Precision/Recall Curve')  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')

    # y_true和y_scores分别是gt label和predict score

    precision1, recall1, thresholds1 = precision_recall_curve(y_test, y_pred_probs[0])
    precision2, recall2, thresholds2 = precision_recall_curve(y_test, y_pred_probs[1])
    precision3, recall3, thresholds3 = precision_recall_curve(y_test, y_pred_probs[2])
    precision4, recall4, thresholds4 = precision_recall_curve(y_test, y_pred_probs[3])
    precision5, recall5, thresholds5 = precision_recall_curve(y_test, y_pred_probs[4])
    precision6, recall6, thresholds6 = precision_recall_curve(gc_y_test, gc_y_pred_probs)
    precision7, recall7, thresholds7 = precision_recall_curve(y_test, y_pred_probs[5])
    area1 = auc(recall1, precision1)
    area2 = auc(recall2, precision2)
    area3 = auc(recall3, precision3)
    area4 = auc(recall4, precision4)
    area5 = auc(recall5, precision5)
    area6 = auc(recall6, precision6)
    area7 = auc(recall7, precision7)

    plt.plot(precision1, recall1, label='LR(area=%0.3f)' % area1)
    plt.plot(precision2, recall2, label='SVM(area=%0.3f)' % area2)
    plt.plot(precision3, recall3, label='RF(area=%0.3f)' % area3)
    plt.plot(precision4, recall4, label='XGBoost(area=%0.3f)' % area4)
    plt.plot(precision5, recall5, label='AdaBoost(area=%0.3f)' % area5)
    plt.plot(precision6, recall6, label='EC-DFR(area=%0.3f)' % area6)
    plt.plot(precision7, recall7, label='DEEP CROSS(area=%0.3f)' % area7)
    plt.legend()
    plt.show()
    # plt.savefig('p-r.png')


