import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES']="0"

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, BatchNormalization, Activation, Flatten
from tensorflow.keras.layers import Input, concatenate


import matplotlib.pyplot as plt
plt.switch_backend('agg')

#seqLength = 600  # TIS and polyA is 600, splice is 398
#gsr = 'TIS' # TIS/ PAS / PAS_AATAAA / PAS_miniData
#organism = 'hs' # hs / mm /  bt / dm





#-------------------------------model framework--------------------------------------------   
def model (seqLength):
    input_sequence = Input(shape=(seqLength, 4))
    towerA_1 = Conv1D(filters=29, kernel_size=1, padding='same', kernel_initializer='he_normal')(input_sequence)
    towerA_1 = BatchNormalization()(towerA_1)
    towerA_1 = Activation('relu')(towerA_1)
    towerA_2 = Conv1D(filters=121, kernel_size=3, padding='same', kernel_initializer='he_normal')(input_sequence)
    towerA_2 = BatchNormalization()(towerA_2)
    towerA_2 = Activation('relu')(towerA_2)
    towerA_3 = Conv1D(filters=467, kernel_size=5, padding='same', kernel_initializer='he_normal')(input_sequence)
    towerA_3 = BatchNormalization()(towerA_3)
    towerA_3 = Activation('relu')(towerA_3)
    output = concatenate([towerA_1, towerA_2, towerA_3], axis=-1)
    output = MaxPooling1D(pool_size=3, padding='same')(output)
    output = Dropout(rate=0.42198224)(output)

    towerB_1 = Conv1D(filters=216, kernel_size=1, padding='same', kernel_initializer='he_normal')(output)
    towerB_1 = BatchNormalization()(towerB_1)
    towerB_1 = Activation('relu')(towerB_1)
    towerB_2 = Conv1D(filters=237, kernel_size=3, padding='same', kernel_initializer='he_normal')(output)
    towerB_2 = BatchNormalization()(towerB_2)
    towerB_2 = Activation('relu')(towerB_2)
    towerB_3 = Conv1D(filters=517, kernel_size=5, padding='same', kernel_initializer='he_normal')(output)
    towerB_3 = BatchNormalization()(towerB_3)
    towerB_3 = Activation('relu')(towerB_3)
    towerB_4 = Conv1D(filters=458, kernel_size=7, padding='same', kernel_initializer='he_normal')(output)
    towerB_4 = BatchNormalization()(towerB_4)
    towerB_4 = Activation('relu')(towerB_4)
    output = concatenate([towerB_1, towerB_2, towerB_3, towerB_4], axis=-1)
    output = MaxPooling1D(pool_size=3, padding='same')(output)
    output = Dropout(rate=0.53868208)(output)

    output = Conv1D(filters=64, kernel_size=1, kernel_initializer='he_normal')(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    output = LSTM(units=123, return_sequences=True)(output)
    output = Dropout(rate=0.57608335)(output)
    output = LSTM(units=391, return_sequences=True)(output)
    output = Dropout(rate=0.49034301)(output)
    output = Flatten()(output)
    output = Dense(units=1, activation='sigmoid')(output)

    model = Model(input_sequence, output)

    return model

