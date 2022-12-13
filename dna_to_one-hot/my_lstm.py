import tensorflow as tf
from keras.models import Model
from keras.layers import Dropout, BatchNormalization, Input, Add, Concatenate
from keras.layers import Dense, Conv2D, MaxPooling2D, LSTM, Reshape
import keras.backend as K


def Precision(y_true, y_pred):
    """精确率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
    precision = tp / (pp + K.epsilon())
    return precision


def Recall(y_true, y_pred):
    """召回率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives
    recall = tp / (pp + K.epsilon())
    return recall


def F1(y_true, y_pred):
    """F1-score"""
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1


def my_lstm_model(config):
    inputs = Input(shape=(config.seq_length, 4, 1))

    x = Conv2D(64, kernel_size=(4, 4),
               activation='relu',
               padding='same')(inputs)
    # print()
    # padding='same')(inputs)
    print('after conv', x.shape)
    x = MaxPooling2D((2, 1), padding='same')(x)
    print('after MaxPooling', x.shape)

    x = BatchNormalization()(x)
    x = Dropout(config.drop_out)(x)
    # x = Dense(1000, activation='relu')(x)
    print('first x.shape', x.shape)

    # parallel line 1
    fx1 = Conv2D(64, kernel_size=(4, 1),
                 activation='relu',
                 padding='same')(x)
    fx1 = BatchNormalization()(fx1)
    fx1 = Dropout(config.drop_out)(fx1)
    print('fx1 x.shape', fx1.shape)

    # fx1 = Conv2D(config.filters, kernel_size=(3, 1),
    #              activation='relu',
    #              padding='same')(fx1)
    # fx1 = MaxPooling2D((2, 1), padding='same')(fx1)
    # fx1 = BatchNormalization()(fx1)
    # fx1 = Dropout(config.drop_out)(fx1)

    # parallel line 2
    fx2 = Conv2D(64, kernel_size=(12, 1),
                 activation='relu',
                 padding='same')(x)
    fx2 = BatchNormalization()(fx2)
    fx2 = Dropout(config.drop_out)(fx2)
    print('fx2 x.shape', fx2.shape)

    # fx2 = Conv2D(config.filters, kernel_size=(21, 1),
    #              activation='relu',
    #              padding='same')(fx2)
    # fx2 = MaxPooling2D((2, 1), padding='same')(fx2)
    # fx2 = BatchNormalization()(fx2)
    # fx2 = Dropout(config.drop_out)(fx2)

    # Add
    x1 = Concatenate(axis=(-3))([fx1, fx2])
    x1 = MaxPooling2D((2, 1), padding='same')(x1)

    # x1 = Add()([fx1, fx2])
    print('concat shape:', x1.shape)
    # x1.reshape
    x = Add()([x, x1])
    print('added.shape', x.shape)
    x = MaxPooling2D((2, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(config.drop_out)(x)
    print((K.int_shape(x)[1], K.int_shape(x)[3]))
    x = Reshape((K.int_shape(x)[1], K.int_shape(x)[2] * K.int_shape(x)[3]))(x)
    # x = LSTM(20, return_sequences=False, input_shape=(x.shape[1],))(x)
    print('lstm input shape', x.shape)
    x = LSTM(config.lstm_units, return_sequences=False)(x)
    # x = Dropout(config.drop_out)(x)

    # ?? outputs = Dense(1, activation='linear')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    network = Model(inputs, outputs)
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
    network.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy', Recall, Precision, F1])
    network.summary()

    return network
