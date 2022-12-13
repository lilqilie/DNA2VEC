import os

import numpy as np
import pandas as pd

from lstm_config import Config

config = Config()


def dnaOneHot(sequence):
    seq_array = np.array(list(sequence))
    # seq_array = np.array(sequence)
    code = {"A": [0], "C": [1], "G": [2], "T": [3], "N": [4], "B": [4], "R": [4], "K": [4], "W": [4], "Y": [4],
            "S": [4], "M": [4], "D": [4], "H": [4], "V": [4],
            "a": [0], "c": [1], "g": [2], "t": [3], "n": [4]}
    onehot_encoded_seq = []
    for char in seq_array:
        onehot_encoded = np.zeros(5)
        onehot_encoded[code[char]] = 1
        # try:
        #     onehot_encoded[code[char]] = 1
        # except KeyError:
        #     onehot_encoded[[4]] = 1
        onehot_encoded_seq.append(onehot_encoded[0:4])
    return onehot_encoded_seq


def one_hot(seq):
    one_hot_seq = np.zeros((4, len(seq)))
    for i in range(len(seq)):
        if seq[i] == "A":
            one_hot_seq[0][i] = 1
        if seq[i] == "C":
            one_hot_seq[1][i] = 1
        if seq[i] == "G":
            one_hot_seq[2][i] = 1
        if seq[i] == "T":
            one_hot_seq[3][i] = 1
    return one_hot_seq


def read_data(data_path):
    """
    读取数据
    :return: 返回分词后的文本内容和标签，inputs = [[]], labels = []
    """
    inputs = []
    labels = []
    with open(data_path, "r", encoding="utf8") as fr:
        i = 0
        for line in fr.readlines():
            try:
                text, label = line.strip().split("<SEP>")
                # inputs.append(text.strip().split(" "))
                seq = text.replace(' ', '')[0:1500]
                # print(len(seq))
                # print(len(seq))
                if len(seq) != 1500:
                    i += 1
                    continue

                inputs.append(seq)
                labels.append(label)
            except:
                continue
    print("不等于1500的数量", i + 1)
    return inputs, labels


def make_data(label_file, out_file):
    # concat

    df = pd.read_csv(label_file, sep=';', names=['1', '2', '3', '4', 'labels', '5'])
    print(df)
    # print(df.labels.to_list())
    df.replace('t__Animal', '0', inplace=True)
    df.replace('t__Zoonotic', '1', inplace=True)
    # df.replace(' t_Multi', '0', inplace=True)
    print(df)

    df['labels'].to_csv(out_file, index=None)

    print(df['labels'].value_counts())
    concat_data()


def data_preprocess(name):
    path1 = os.path.abspath('./') + config.npy_file
    if os.path.exists(path1):
        print('exists')
        X = np.load(config.npy_file)
        print('X', X.shape)
        labels = pd.read_csv(config.qc_path).iloc[:30000, :]['labels']
        print('labels', labels.shape)
        print(labels.value_counts())
        labels = labels.tolist()
        if name == '1':
            X1 = np.concatenate((X[:5000], X[16000:21000]), axis=0)
            Y1 = labels[:5000] + labels[16000:21000]
            print('X1 shape:', X1.shape)
            return X1, Y1

        if name == '2':
            X2 = np.concatenate((X[5000:10000], X[21000:26000]), axis=0)
            Y2 = labels[5000:10000] + labels[21000:26000]
            print('X2 shape:', X2.shape)
            return X2, Y2

        if name == '3':
            X3 = np.concatenate((X[10000:15000], X[26000:30000]), axis=0)
            Y3 = labels[10000:15000] + labels[26000:30000]
            print('X3 shape:', X3.shape)
            return X3, Y3

        # print(X2.shape)
        # print(X3.shape)
        #
        # print(pd.value_counts(Y1))
        # print(pd.value_counts(Y2))
        # print(pd.value_counts(Y3))

        # else:
        #     print('dont have npy file, making file now...')
        #     X = []
        #     df = pd.read_csv(config.qc_path)
        #     print(df['labels'].value_counts())
        #     labels = df['labels'].to_list()
        #     seqs = df['seqs'].to_list()
        #     for seq in seqs:
        #         X.append(dnaOneHot(seq[:config.seq_length]))
        #     X = np.array(X)
        #     # print(X.shape)
        #     X = X.reshape((X.shape[0], config.seq_length, 4, 1))
        #     np.save(config.npy_file, X)
        #     print('saved')
        # return X, labels


def readfq(fp):  # this is a generator function 读取fasta文件
    last = None  # this is a buffer keeping the last unprocessed line
    while True:  # mimic closure; is it a bad idea?
        if not last:  # the first record or a record following a fastq
            for l in fp:  # search for the start of the next record
                if l[0] in '>@':  # fasta/q header line
                    last = l[:-1]  # save this line
                    break
        if not last: break
        name, seqs, last = last[1:].partition(" ")[0], [], None
        for l in fp:  # read the sequence
            if l[0] in '@+>':
                last = l[:-1]
                break
            seqs.append(l[:-1])
        if not last or last[0] != '+':  # this is a fasta record
            # yield name, ''.join(seqs)[0], None # yield a fasta record
            yield name, ''.join(seqs), None  # yield a fasta record
            if not last: break
        else:  # this is a fastq record
            seq, leng, seqs = ''.join(seqs), 0, []
            for l in fp:  # read the quality
                seqs.append(l[:-1])
                leng += len(l) - 1
                if leng >= len(seq):  # have read enough quality
                    last = None
                    yield name, seq, ''.join(seqs);  # yield a fastq record
                    break
            if last:  # reach EOF before reading enough quality
                yield name, seq, None  # yield a fasta record instead
                break


def get_seq(infile):
    fp = open(infile)
    i = 0
    seqs = []
    seq_lengths = []
    # labels = pd.read_csv('./data/ncbi_labels.csv')['labels'].tolist()
    # print('len(labels)', len(labels))
    for name, seq, _ in readfq(fp):
        # seq = seq + ',' + str(labels[i])
        i += 1
        # print('len seq:', len(seq))
        # if len(seq) < 210:
        #     print('di', i)
        #     continue
        seqs.append(seq)
        seq_lengths.append(len(seq))
    print('len(seqs)', len(seqs))
    bins = [1000, 1200, 1400, 1500, 1600, 1800, 2000, 5000]
    cats = pd.cut(seq_lengths, bins)

    print(pd.value_counts(cats))

    print(pd.cut(seq_lengths, bins, right=False))

    # if i % 100000 == 0:
    # if i < 100000:
    return seqs


def qualify():
    # seq_path = './data/ncbi_with_labels.csv'

    df = pd.read_csv(config.data_path)
    print(df.shape)
    # df = df.drop(df[df.labels == 2].index, axis=0)
    # print(df.shape)
    seqs = df['seqs'].to_list()
    filltered_seqs = []
    seq_lengths = []
    print('数据集共：', len(seqs))
    for i in range(len(seqs)):
        lens = len(seqs[i])
        seq_lengths.append(lens)
        if lens < config.seq_length or lens > 1600:
            print(i)
            filltered_seqs.append(i)
    print('original:', pd.DataFrame(seq_lengths).describe())

    print('数据集中序列长度小于%d bp条数：%d' % (config.seq_length, len(filltered_seqs)))
    df = df.drop(index=filltered_seqs, inplace=False)
    print('质控后:', df.shape)

    df.to_csv(config.qc_path)


def concat_data():
    seqs = get_seq(config.raw_fasta)
    seqs = pd.DataFrame({'seqs': seqs})
    labels = pd.read_csv(config.labels_path)
    df = pd.concat([seqs, labels], axis=1)
    df.to_csv(config.data_path)


def label_check(path):
    df = pd.read_csv(path)
    print(df['labels'].value_counts())


if __name__ == '__main__':
    # make_data('./raw_data/animal_zoonotic.txt', './pro_data/animal_zoo_labels.csv')
    # X, labels = data_preprocess(seq_path='./')
    # qualify()
    # path = './data/quality_ncbi_with_all_labels.csv'
    # path = './data/ncbi_all_labels.csv'
    # label_check(path)
    data_preprocess()
