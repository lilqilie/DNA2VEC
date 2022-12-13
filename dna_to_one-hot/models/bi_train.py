import json
import os
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_pre import data_preprocess

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import tensorflow as tf
# from data_helpers import TrainData, EvalData
from train_base import TrainerBase
from models import TextCnnModel, BiLstmModel, BiLstmAttenModel, RcnnModel, TransformerModel
from utils.metrics import get_binary_metrics, get_multi_metrics, mean

# def data_preprocess(path):
#     X = []
#     df = pd.read_csv(path)
#     df = df.iloc[1000:, :]
#     print(df['labels'].value_counts())
#     labels = df['labels'].to_list()
#     print(len(labels))
#     seqs = df['seqs'].to_list()
#     for seq in seqs:
#         X.append(dnaOneHot(seq[:config.seq_length]))
#     # print('X', len(X))
#
#     X = np.array(X)
#     # print(X.shape)
#     X = X.reshape((X.shape[0], config.seq_length, 4, 1))
#     return X, labels

class Trainer(TrainerBase):
    def __init__(self, config_path):
        super(Trainer, self).__init__()
        self.config_path = config_path
        with open(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config_path), "r") as fr:
            self.config = json.load(fr)

        # self.train_data_obj = None
        # self.eval_data_obj = None
        self.model = None
        # self.builder = tf.saved_model.builder.SavedModelBuilder("../pb_model/weibo/bilstm/savedModel")

        # 加载数据集
        X, y = data_preprocess(self.config['data_path'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                test_size=self.config['test_size'],
                                                                                random_state=self.config[
                                                                                    'random_state'])
        # self.load_data()
        # self.train_inputs, self.train_labels, label_to_idx = self.train_data_obj.gen_data()
        # print("train data size: {}".format(len(self.train_labels)))
        # self.vocab_size = self.train_data_obj.vocab_size
        # print("vocab size: {}".format(self.vocab_size))
        # self.word_vectors = self.train_data_obj.word_vectors
        # self.label_list = [value for key, value in label_to_idx.items()]
        #
        # self.eval_inputs, self.eval_labels = self.eval_data_obj.gen_data()
        # print("eval data size: {}".format(len(self.eval_labels)))
        # print("label numbers: ", len(self.label_list))
        # 初始化模型对象
        self.create_model()

    # def load_data(self):
    #     """
    #     创建数据对象
    #     :return:
    #     """
    #     # 生成训练集对象并生成训练数据
    #     self.train_data_obj = TrainData(self.config)
    #
    #     # 生成验证集对象和验证集数据
    #     self.eval_data_obj = EvalData(self.config)

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        if self.config["model_name"] == "textcnn":
            self.model = TextCnnModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        elif self.config["model_name"] == "bilstm":
            self.model = BiLstmModel(config=self.config)
        elif self.config["model_name"] == "bilstm_atten":
            self.model = BiLstmAttenModel(config=self.config, vocab_size=self.vocab_size,
                                          word_vectors=self.word_vectors)
        elif self.config["model_name"] == "rcnn":
            self.model = RcnnModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        elif self.config["model_name"] == "transformer":
            self.model = TransformerModel(config=self.config, vocab_size=self.vocab_size,
                                          word_vectors=self.word_vectors)

    ## get batch
    def next_batch(self, x, y, batch_size):
        """
        生成batch数据集
        :param x: 输入
        :param y: 标签
        :param batch_size: 批量的大小
        :return:
        """
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = np.array(x)[perm]
        y = np.array(y)[perm]

        num_batches = len(x) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = np.array(x[start: end], dtype="int64")
            batch_y = np.array(y[start: end], dtype="float32")

            yield dict(x=batch_x, y=batch_y)

    def train(self):
        """
        训练模型
        :return:
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            # 初始化变量值
            sess.run(tf.global_variables_initializer())
            current_step = 0

            # 创建train和eval的summary路径和写入对象
            train_summary_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                              self.config["output_path"] + "/summary/train")
            if not os.path.exists(train_summary_path):
                os.makedirs(train_summary_path)
            train_summary_writer = tf.summary.FileWriter(train_summary_path, sess.graph)

            eval_summary_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                             self.config["output_path"] + "/summary/eval")
            if not os.path.exists(eval_summary_path):
                os.makedirs(eval_summary_path)
            eval_summary_writer = tf.summary.FileWriter(eval_summary_path, sess.graph)

            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.next_batch(self.X_train, self.y_train,
                                             self.config["batch_size"]):
                    summary, loss, predictions = self.model.train(sess, batch, self.config["keep_prob"])
                    train_summary_writer.add_summary(summary)

                    if self.config["num_classes"] == 1:
                        acc, auc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batch["y"])
                        print(
                            "train: step: {}, loss: {}, acc: {}, auc: {}, recall: {}, precision: {}, f_beta: {}".format(
                                current_step, loss, acc, auc, recall, prec, f_beta))

                    current_step += 1
                    if self.eval_data_obj and current_step % self.config["checkpoint_every"] == 0:

                        eval_losses = []
                        eval_accs = []
                        eval_aucs = []
                        eval_recalls = []
                        eval_precs = []
                        eval_f_betas = []
                        for eval_batch in self.next_batch(self.X_test, self.y_test,
                                                          self.config["batch_size"]):
                            eval_summary, eval_loss, eval_predictions = self.model.eval(sess, eval_batch)
                            eval_summary_writer.add_summary(eval_summary)

                            eval_losses.append(eval_loss)
                            if self.config["num_classes"] == 1:
                                acc, auc, recall, prec, f_beta = get_binary_metrics(pred_y=eval_predictions,
                                                                                    true_y=eval_batch["y"])
                                eval_accs.append(acc)
                                eval_aucs.append(auc)
                                eval_recalls.append(recall)
                                eval_precs.append(prec)
                                eval_f_betas.append(f_beta)
                            elif self.config["num_classes"] > 1:
                                acc, recall, prec, f_beta = get_multi_metrics(pred_y=eval_predictions,
                                                                              true_y=eval_batch["y"],
                                                                              labels=self.label_list)
                                eval_accs.append(acc)
                                eval_recalls.append(recall)
                                eval_precs.append(prec)
                                eval_f_betas.append(f_beta)
                        print("\n")
                        print("eval:  loss: {}, acc: {}, auc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            mean(eval_losses), mean(eval_accs), mean(eval_aucs), mean(eval_recalls),
                            mean(eval_precs), mean(eval_f_betas)))
                        print("\n")

                        if self.config["ckpt_model_path"]:
                            save_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                                     self.config["ckpt_model_path"])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config_path", help="config path of model")
    # args = parser.parse_args()
    config_path = 'config/bilstm_config.json'
    # config_path = 'text_classifier/config/bilstm_config.json'
    trainer = Trainer(config_path)
    trainer.train()
