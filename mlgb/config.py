import torch
class Args:
    def __init__(self):
        self.method = 'origin'
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = 'QuickBird' # self.dataset = 'chikusei'  # 'chikusei' or 'xiongan'
        self.pretrained_model_path = None
        self.batch_size = 8
        self.end_epoch = 150
        self.init_lr = 2e-4
        self.ratio = 8
        # self.data_path = '/yehui/datasets/CAVE/'
        # self.gen_path = '/yehui/GuidedNet/dataset'
        self.PAN_path = '/yehui/datasets/QuickBird/PAN_1024/'
        self.MS_path = '/yehui/datasets/QuickBird/MS_256/'
        # self.data_path = '/yehui/GuidedNet/dataset/PaviaU/PaviaU.mat'
        # self.gen_path = '/yehui/GuidedNet/dataset/'
        self.patch_size = 16
        self.stride = 6
        self.gpu_id = '1'
        self.iteration_batch = 100
        self.num_workers = 0
        self.hsi_channels = 4
        self.pan_channels = 1
        self.msi_channels = 4
        self.seed = 42
        self.auto_patch = True
        self.enable_log = True # Enable or disable logging
        self.log_path = f'/yehui/ResNet/logs/{self.dataset}/{self.method}/'
        self.result_path = f'/yehui/ResNet/results/{self.dataset}/{self.method}/'
        self.pretrained_model_path = f'/yehui/ResNet/logs/{self.dataset}/{self.method}/2025_07_28_14_53_22/model/net_149epoch.pth'