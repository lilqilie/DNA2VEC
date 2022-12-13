'''
requirement
tensorflow2.0
scikit-learn
'''
import os
import numpy as np
#np.random.seed(1337)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES']="0"

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, BatchNormalization, Activation, Flatten
from tensorflow.keras.layers import Input, concatenate

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score

import argparse

import matplotlib.pyplot as plt
plt.switch_backend('agg')

#seqLength = 600  # TIS and polyA is 600, splice is 398
#gsr = 'TIS' # TIS/ PAS / PAS_AATAAA / PAS_miniData
#organism = 'hs' # hs / mm /  bt / dm


#------------------------- Arguments Parser ------------------------------------------
parser = argparse.ArgumentParser(description='Parameters to be used for DeepGenGrep training.')
parser.add_argument('-gsr', '--GSR', help='Genome signal region: TIS, PAS,PAS_AATAAA, Splice_acc, Splice_don', required=True)
parser.add_argument('-org', '--Organism', help='Organism name', required=True)
parser.add_argument('-len', '--SeqLength', type=int, default=600, help='The length of input sequence', required=True)

args = parser.parse_args()

gsr = args.GSR
organism = args.Organism
seqLength = args.SeqLength


#-----------------------print input arguments------------------------------------------------
print('gsr:',gsr)
print('organism:',organism)
print('seqLength:',seqLength)


#-------------------------------translate input sequence to encoding matrix-------------------------
def seq_matrix(seq_list, label):
    tensor = np.zeros((len(seq_list), seqLength, 4))
    for i in range(len(seq_list)):
        seq = seq_list[i]
        j = 0
        for s in seq:
            if s == 'A':
                tensor[i][j] = [1, 0, 0, 0]
            if s == 'T':
                tensor[i][j] = [0, 1, 0, 0]
            if s == 'C':
                tensor[i][j] = [0, 0, 1, 0]
            if s == 'G':
                tensor[i][j] = [0, 0, 0, 1]

            j += 1
    if label == 1:
        y = np.ones((len(seq_list), 1), dtype=np.int)
    if label == 0:
        y = np.zeros((len(seq_list), 1), dtype=np.int)
    return tensor, y


#-------------------------------model framework--------------------------------------------   
def DeepGenGrep():
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

# main function

if os.path.isfile('Data/'+gsr+'/process/'+organism+'_x.npy') & os.path.isfile('Data/'+gsr+'/process/'+organism+'_y.npy'):
    x_seq = np.load('Data/'+gsr+'/process/'+organism+'_x.npy')
    y_seq = np.load('Data/'+gsr+'/process/'+organism+'_y.npy')
else:
    pos_file = 'Data/'+gsr+'/'+organism+'_pos_'+gsr+'.txt'
    neg_file = 'Data/'+gsr+'/'+organism+'_neg_'+gsr+'.txt'

    pos_seq = np.loadtxt(pos_file, dtype=np.str)
    for i in range(len(pos_seq)):
        pos_seq[i] = pos_seq[i][:seqLength]
    neg_seq = np.loadtxt(neg_file, dtype=np.str)
    for i in range(len(neg_seq)):
        neg_seq[i] = neg_seq[i][:seqLength]

    print(str(len(pos_seq)) + ' positive test data loaded...')
    print(str(len(neg_seq)) + ' negative test data loaded...')

    pos_seq_x, pos_seq_y = seq_matrix(pos_seq, label=1)
    neg_seq_x, neg_seq_y = seq_matrix(neg_seq, label=0)
    x_seq = np.concatenate((pos_seq_x, neg_seq_x), axis=0)
    y_seq = np.concatenate((pos_seq_y, neg_seq_y), axis=0)

    np.save('Data/'+gsr+'/process/'+organism+'_x.npy', x_seq)
    np.save('Data/'+gsr+'/process/'+organism+'_y.npy', y_seq)

print('shuffle data...\n')
index = np.arange(len(x_seq))
np.random.shuffle(index)
x_seq = x_seq[index]
y_seq = y_seq[index]
    
x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq, test_size=0.25, random_state=1337, stratify=y_seq)

test_label = y_test

y_train = y_train.ravel().tolist()
y_test = y_test.ravel().tolist()
y_train = np.array(y_train)
y_test = np.array(y_test)

np.save('Data/'+gsr+'/process/'+organism+'_x_test.npy', x_test)
np.save('Data/'+gsr+'/process/'+organism+'_y_test.npy', y_test)

print('training samples: ', x_train.shape[0])
print('testing samples: ', x_test.shape[0])

print('Building model...')

model = DeepGenGrep()
#------------------------------------------ training the model---------------------------------------------
print('Compiling model...')
model.compile(loss='binary_crossentropy', #binary_crossentropy / categorical_crossentropy
              optimizer='nadam',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='./log/DeepGenGrep')
checkpoint = ModelCheckpoint(filepath=f"Model/{gsr}/{organism}_bestModel.hdf5",
                             monitor='val_accuracy',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=30,
                               mode='max')

callback_lists = [checkpoint, early_stopping]
print('-'*100)
hist = model.fit(x_train, y_train,
                 batch_size=64,
                 epochs=150,
                 verbose=2,
                 callbacks=callback_lists,
                 validation_split=0.2)

#-----------------------------------evaluate model-----------------------------------------------------------
model.load_weights(f"Model/{gsr}/{organism}_bestModel.hdf5")
scores = model.evaluate(x=x_test, y=y_test, batch_size=64, verbose=2)
predict_probs = model.predict(x_test, batch_size=64, verbose=2)
predict_classes = predict_probs.round()

#---------------------------------------save Evaluation Results ---------------------------------------------
cm = confusion_matrix(test_label, predict_classes)
total=sum(sum(cm))
accuracy=(cm[0,0]+cm[1,1])/total
sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])
precision = precision_score(test_label, predict_classes)
recall = recall_score(test_label, predict_classes)
f1Score = f1_score(test_label, predict_classes)

average_precision = average_precision_score(test_label, predict_classes)
AUC=roc_auc_score(test_label, predict_classes)

FileNameTrainingR = f'Results/{gsr}/{organism}_TrainingResults.txt'
Result_file=open(FileNameTrainingR,'w')
Result_file.write("Accuracy by calculated by confusion_matrix: %.2f%%\n" %(accuracy*100))
Result_file.write("%s: %.2f%% calculated by the best model\n" % (model.metrics_names[1], scores[1]*100))
Result_file.write("Sensitivity: %.2f%%\n" %( sensitivity*100))
Result_file.write("Specificity: %.2f%%\n" %(specificity*100))
Result_file.write("precision: %.2f%%\n" %(precision*100))
Result_file.write("recall: %.2f%%\n" %(recall*100))
Result_file.write("f1_score: %.2f%%\n" %(f1Score*100))
Result_file.write('AUPR: '+str(average_precision))
Result_file.write('\nAUC: '+str(AUC))
Result_file.close()