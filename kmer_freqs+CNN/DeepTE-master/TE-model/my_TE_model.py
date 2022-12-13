# 4. Define model architecture
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import StandardScaler

from config import Config
from keras.utils import np_utils
from tool import DataPreprocess

config = Config()


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


def deep_te():
    model = Sequential()

    model.add(Conv2D(100, (1, 3), activation='relu', input_shape=(1, 10952, 1)))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(150, (1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(225, (1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    ##You can add a dropout layer to overcome the problem of overfitting to some extent. Dropout randomly turns off
    # a fraction of neurons during the training process, reducing the dependency on the training set by some amount.
    # How many fractions of neurons you want to turn off is decided by a hyperparameter, which can be tuned accordingly.
    # This way, turning off some neurons will not allow the network to memorize the training data since not all the neurons
    # will be active at the same time and the inactive neurons will not be able to learn anything.
    # This way, turning off some neurons will not allow the network to memorize the training data
    # since not all the neurons will be active at the same time and the inactive neurons will not be able to learn anything.

    model.add(Dense(2, activation='softmax'))
    # model.add(Dense(2, activation='relu'))
    model.summary()
    return model


# since 4 classes ##the output have four unit
##Your output's are integers for class labels. Sigmoid logistic function outputs values in range (0,1).
# The output of the softmax is also in range (0,1), but the softmax function adds another constraint on outputs:-
# the sum of outputs must be 1. Therefore the output of softmax can be interpreted as probability of the input
# for each class.
def train_te():
    model = deep_te()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ###########################
    # 6. Fit model on training data

    train_df, X_train, X_test, y_train, y_test = DataPreprocess()
    # df_X, val_patho, val_label = testDataPreprocess()

    X_train = X_train.reshape(X_train.shape[0], 1, 10952, 1)  ##shape[0]
    # print(X_train)
    print(X_train.shape)
    # indicates sample number
    # print('X_train is ')
    # print(X_train)
    X_test = X_test.reshape(X_test.shape[0], 1, 10952, 1)  ##kmer == 3 so it would be 64
    # val_patho = val_patho.reshape(val_patho.shape[0], 1, 10952, 1)  ##kmer == 3 so it would be 64
    # print(X_test)
    print(X_test.shape)
    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')
    # 3. Preprocess class labels; i.e. convert 1-dimensional class arrays to 3-dimensional class matrices
    Y_train_one_hot = np_utils.to_categorical(y_train, 2)  # four labels
    Y_test_one_hot = np_utils.to_categorical(y_test, 2)  #
    # val_label = np_utils.to_categorical(val_label, 2)  #

    model.fit(X_train, Y_train_one_hot, validation_data=(X_test, Y_test_one_hot),
              batch_size=128, epochs=35,
              verbose=1)  ##epoch An epoch is an iteration over the entire x and y data provided
    ##batch size if we have 1000 samples, we set batch size to 100, so we will
    ##run 100 first and then the second 100, so this will help us to reduce the
    ##the memory we use
    # model.fit(X_train, y_train, validation_data=(X_test, y_test),
    #           batch_size=1, epochs=10,
    #           verbose=1)
    ###########################

    y_pred = []
    y_pred_probs = []
    y_pred_porb = model.predict(X_test)
    # print(y_pred)

    for probs in y_pred_porb:
        y_pred_probs.append(probs[1])
        if probs[1] > 0.5:
            y_pred.append(1)

        else:
            y_pred.append(0)

    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)
    print('acc:', acc)
    print('mcc:', mcc)
    print('f1:', f1)
    print('auc:', auc)

    pd.DataFrame({'y_pred_porb': y_pred_probs}).to_csv('TE_pred_porb.csv')
    # print(y_pred_probs)
    # val_pred = model.predict(val_patho)
    # val_y_pred = []
    # for probs in val_pred:
    #     # y_pred_probs.append(probs[1])
    #     if probs[1] > 0.5:
    #         val_y_pred.append(1)
    #
    #     else:
    #         val_y_pred.append(0)
    # val_acc = accuracy_score(val_label, val_y_pred)
    # print('DeepTE val acc:', val_acc)

    # predicted_classes = np.argmax(np.round(predicted_classes), axis=1)


if __name__ == '__main__':
    train_te()
