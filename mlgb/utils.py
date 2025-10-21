import cv2
import numpy as np
import os
import torch
import scipy.io as sio
import scipy.interpolate as spi
import tensorly as tl
import logging
import glob

def gauss_kernel(row_size, col_size, sigma):
    kernel = cv2.getGaussianKernel(row_size, sigma)
    kernel = kernel * cv2.getGaussianKernel(col_size, sigma).T
    return kernel

def intersect(list1, list2):
    list1 = list(list1)
    elem = list(set(list1).intersection(set(list2)))
    elem.sort()
    res = np.zeros(len(elem))
    for i in range(0, len(elem)):
        res[i] = list1.index(elem[i])
    res = res.astype("int32")
    return res

def create_spec_resp(data_num, genPath):
    if data_num == 0:  # CAVE  31 X 3
        band = 31
        file = os.path.join(genPath, 'srf/D700.mat')  # 377-948
        mat = sio.loadmat(file)
        spec_rng = np.arange(400, 700 + 1, 10)
        spec_resp = mat['spec_resp']
        #print("spec_rng:", spec_rng)
        R = spec_resp[spec_rng - 377, 1:4].T
        
    if data_num == 1:  # harvard  31 X 3
        file = os.path.join(genPath, 'srf/D700.mat')  # 377-948
        mat = sio.loadmat(file)
        spec_rng = np.arange(420, 720 + 1, 10)
        spec_resp = mat['spec_resp']
        R = spec_resp[spec_rng - 377, 1:4].T

    if data_num == 2:
        band = 102  # paviaC
        file = os.path.join(genPath, 'srf/ikonos.mat')  # 350 : 5 : 1035
        mat = sio.loadmat(file)
        spec_rng = np.arange(430, 861)
        spec_resp = mat['spec_resp']
        ms_bands = range(1, 5)
        valid_ik_bands = intersect(spec_resp[:, 0], spec_rng)
        no_wa = len(valid_ik_bands)
        # Spline interpolation
        xx = np.linspace(1, no_wa, band)
        x = range(1, no_wa + 1)
        R = np.zeros([5, band])
        for i in range(0, 5):
            ipo3 = spi.splrep(x, spec_resp[valid_ik_bands, i + 1], k=3)
            R[i, :] = spi.splev(xx, ipo3)
        R = R[ms_bands, :]

    if data_num == 3:
        # PaviaU 103 X 4
        band = 103 
        file = os.path.join(genPath, 'srf/ikonos_SRF.mat')
        mat = sio.loadmat(file)
        spec_resp = mat['R']
        R  = spec_resp

    if data_num == 4:
        # Chikusei  128 X 4
        band = 128
        file = os.path.join(genPath, 'srf/ikonos.mat')
        mat = sio.loadmat(file)
        spec_rng = np.arange(375, 1015, 5)
        spec_resp = mat['spec_resp']
        R = spec_resp[(spec_rng - 350) // 5, 2:6].T
    c = 1 / np.sum(R, axis=1)
    R = np.multiply(R, c.reshape([c.size, 1]))
    return R


def create_hrms_lrhs(hs, B, R, ratio):
    hrms = tl.tenalg.mode_dot(hs, R, mode=2)
    # blur
    lrhs = cv2.filter2D(hs, -1, B, borderType=cv2.BORDER_REFLECT)
    # downsample
    lrhs = lrhs[0::ratio, 0::ratio, :]
    return hrms, lrhs

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n # 记录当前的样本数
        self.avg = self.sum / self.count # 计算当前的指标平均值
 
def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, model, optimizer):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))


import re
def get_best_model_name_from_log(log_path):
    """
    从train.log中自动获取Eval PSNR最高的epoch，并返回对应的模型文件名


    :param log_path: 日志文件路径
    :return: 最佳模型文件名（如 net_149epoch.pth）
    """



    best_psnr = -float('inf')
    best_epoch = None
    pattern = re.compile(r'Epoch\[(\d+)\].*Eval PSNR: ([\d\.]+)')

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                psnr = float(match.group(2))
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_epoch = epoch

    if best_epoch is not None:
        return f'net_{best_epoch}epoch.pth'
    else:
        raise ValueError("未找到有效的PSNR记录！")

def keep_only_best_model(model_dir, best_model_name):
    """
    只保留指定的最佳模型文件，删除其他 .pth 文件

    :param model_dir: 模型文件夹路径
    :param best_model_name: 最佳模型文件名（如 'net_149epoch.pth'）
    """
    best_model_path = os.path.join(model_dir, best_model_name)
    print(best_model_path)
    all_pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
    for file_path in all_pth_files:
        if os.path.abspath(file_path) != os.path.abspath(best_model_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
    print(f"Kept best model: {best_model_path}")

def save_best_pth(args):
    model_dir = os.path.dirname(args.model_path)                # 获得存放model的目录
    log_path = os.path.join(args.log_path, 'train.log')              # 添加后缀
    best_model_name = get_best_model_name_from_log(log_path)    # 从log中找到最佳的Epoch
    keep_only_best_model(model_dir, best_model_name)            # 保存最佳的pth
    print("success")