class Config:
    def __init__(self):
        # data
        # self.data_path = './data/quality_ncbi_with_all_labels.csv'
        self.data_path = './data/ncbi_with_labels.csv'
        self.seq_length = 1000
        self.test_size = 0.2
        self.random_state = 1
        # train
        self.batch_size = 256
        self.drop_out = 0.1
        self.epoch = 250
        self.learning_rate = 0.0005
        self.lstm_units = 32
        self.filters = 64

        # self.checkpoint_every = 100
        # self.eval_every = 100
