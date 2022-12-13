class Config:
    def __init__(self):
        # self.patho_path = r'D:\desktop\randomForest\data\plas_freq100.csv'
        # self.nonpatho_path = r'D:\desktop\randomForest\data\chrom_freq100.csv'
        # self.patho_path = r'D:\desktop\randomForest\data\k=8_patho_freq.csv'
        # self.nonpatho_path = r'D:\desktop\randomForest\data\k=8_non_freq.csv'
        self.patho_path = r'E:\python_project\randomForest\data\patho_freq1500.csv'
        # self.patho_path = r'D:\desktop\randomForest\data\k=8_patho_freq.csv'
        self.nonpatho_path = r'E:\python_project\randomForest\data\nonpatho_freq1500.csv'
        self.test_path = r'E:\combine\dataset\patho_test.csv'
        self.hidden_layers = [1024, 512, 256]
        # self.deep_layers = [10952, 4096, 2048, 1024]  # 设置Deep模块的隐层大小
        self.deep_layers = [8192, 4096, 2048, 1024]  # 设置Deep模块的隐层大小
        # self.deep_layers = [4096, 2048, 1024, 512]  # 设置Deep模块的隐层大小
        self.num_cross_layers = 10  # cross模块的层数
        self.out_layer_dims = 1024
        self.test_size = 0.2
        self.random_state = 1
        self.num_epoch = 200
        self.batch_size = 256
        self.Dropout = 0.1
        self.lr = 0.000001
        # self.lr = 0.0001
        self.l2_regularization = 0.00001
        self.device_id = 0
        self.use_cuda = True
        # self.use_cuda = False
        self.model_name = 'RDC.model'
        # self.model_name: 'dc.model'



if __name__ == '__main__':
    config = Config()
    name = config.model_name
    print(name)
