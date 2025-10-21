# -*- coding: utf-8 -*-
'''
@Author: Yehui
@██╗   ██╗███████╗██╗  ██╗██╗   ██╗██╗
@╚██╗ ██╔╝██╔════╝██║  ██║██║   ██║██║
@ ╚████╔╝ █████╗  ███████║██║   ██║██║
@  ╚██╔╝  ██╔══╝  ██╔══██║██║   ██║██║
@   ██║   ███████╗██║  ██║╚██████╔╝██║
@   ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝
@Date: 2025-06-05 15:08:08
@LastEditTime: 2025-09-26 15:26:19
@FilePath: /Global_Module/mlgb/datasets.py
@Copyright (c) 2025 by , All Rights Reserved.
'''
from unittest import TestLoader
import numpy as np
import os
import scipy.io as sio
from torch.utils.data import Dataset
import torch
from mlgb.utils import create_hrms_lrhs, create_spec_resp, gauss_kernel
from mlgb.config import Args
from torch.utils.data import DataLoader
import glob
import random
args = Args()

class CaveDataset(Dataset): 
    def __init__(self, data_path, gen_path, patch_size=16, stride=6, ratio=8, kerSize = 8, sigma = 2, type='train'):
        super(CaveDataset, self).__init__()
        self.data_path = data_path
        self.stride = stride
        self.rows = 64 # LR-HSI rows = HR-HSI rows / ratio
        self.cols = 64 # LR-HSI cols
        self.patch_size = patch_size
        self.kerSize = kerSize
        self.sigma = sigma
        self.stride = stride # stride for LR-HSI patch extraction
        self.ratio = ratio
        self.type = type   
        self.B = gauss_kernel(self.kerSize, self.kerSize, self.sigma)
        self.R = create_spec_resp(0, gen_path)

        # Generate samples and labels
        if self.type=='train':
            self.hsi_data, self.msi_data, self.label = self.generateTrain(self.patch_size, self.ratio, num_star=1, num_end=23, s=10,auto_patch = True)
        if self.type=='eval':
            self.hsi_data, self.msi_data, self.label = self.generateEval(self.patch_size, self.ratio, num_star=1, num_end=23, s=28,auto_patch = True)
        if self.type=='test':
            self.hsi_data, self.msi_data, self.label = self.generateTest(self.patch_size, self.ratio, num_star=28, num_end=33, s=1, auto_patch = True)

    def generateTrain(self, patch_size, ratio, num_star, num_end, s ,auto_patch=True):
        num_patch = ((self.rows - patch_size) // self.stride + 1) * ((self.cols - patch_size) // self.stride + 1)
        print('-------------------------------------------------')
        print('One Image patch number:', num_patch)
        if auto_patch:
            s = int(np.sqrt(num_patch))
        label_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 31), dtype=np.float32)
        hrmsi_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 3), dtype=np.float32)
        lrhsi_patch = np.zeros((s * s * (num_end - num_star), patch_size, patch_size, 31), dtype=np.float32)
        # label_patch = []
        # hrmsi_patch = []
        # lrhsi_patch = []
        count = 0
        for i in range(num_star, num_end):
            hrhsi = sio.loadmat(self.data_path + '%d.mat' % i)['HS']
            hrmsi, lrhsi = create_hrms_lrhs(hrhsi, self.B, self.R, self.ratio)

            # Data type conversion
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

            for x in range(0, self.rows - patch_size + 1, self.stride):
                for y in range(0, self.cols - patch_size + 1, self.stride):
                    # rotTimes = random.randint(0, 3)
                    # vFlip = random.randint(0, 1)
                    # hFlip = random.randint(0, 1)
                    # label_patch[count] = self.arguement(hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :], rotTimes, vFlip, hFlip)
                    # hrmsi_patch[count] = self.arguement(hrmsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :], rotTimes, vFlip, hFlip)
                    # lrhsi_patch[count] = self.arguement(lrhsi[x:x + patch_size, y:y + patch_size, :], rotTimes, vFlip, hFlip)
                    label_patch[count] = hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_size, y:y + patch_size, :]
                    count += 1

        return lrhsi_patch, hrmsi_patch, label_patch

    def generateEval(self, patch_size, ratio, num_star, num_end, s,auto_patch=False):
        num_patch = ((self.rows - patch_size) // self.stride + 1) * ((self.cols - patch_size) // self.stride + 1)
        if auto_patch:
            s = int(np.sqrt(num_patch))
        label_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 31), dtype=np.float32)
        hrmsi_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 3), dtype=np.float32)
        lrhsi_patch = np.zeros((s * s * (num_end - num_star), patch_size, patch_size, 31), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            hrhsi = sio.loadmat(self.data_path + '%d.mat' % i)['HS']
            hrmsi, lrhsi = create_hrms_lrhs(hrhsi, self.B, self.R, self.ratio)

            # Data type conversion
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

            for x in range(0, self.rows - patch_size + 1, self.stride):
                for y in range(0, self.cols - patch_size + 1, self.stride):
                    # rotTimes = random.randint(0, 3)
                    # vFlip = random.randint(0, 1)
                    # hFlip = random.randint(0, 1)
                    # label_patch[count] = self.arguement(hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :], rotTimes, vFlip, hFlip)
                    # hrmsi_patch[count] = self.arguement(hrmsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :], rotTimes, vFlip, hFlip)
                    # lrhsi_patch[count] = self.arguement(lrhsi[x:x + patch_size, y:y + patch_size, :], rotTimes, vFlip, hFlip)
                    label_patch[count] = hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_size, y:y + patch_size, :]
                    count += 1

        return lrhsi_patch, hrmsi_patch, label_patch

    def generateTest(self, patch_size, ratio, num_star, num_end, s,auto_patch=False):
        num_patch = ((self.rows - patch_size) // self.stride + 1) * ((self.cols - patch_size) // self.stride + 1)
        if auto_patch:
            s = int(np.sqrt(num_patch))
        print(s)
        label_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 31), dtype=np.float32)
        hrmsi_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 3), dtype=np.float32)
        lrhsi_patch = np.zeros((s * s * (num_end - num_star), patch_size, patch_size, 31), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            hrhsi = sio.loadmat(self.data_path + '%d.mat' % i)['HS']
            hrmsi, lrhsi = create_hrms_lrhs(hrhsi, self.B, self.R, self.ratio)

            # Data type conversion
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

            for x in range(0, self.rows - patch_size + 1, self.stride):
                for y in range(0, self.cols - patch_size + 1, self.stride):
                    # rotTimes = random.randint(0, 3)
                    # vFlip = random.randint(0, 1)
                    # hFlip = random.randint(0, 1)
                    # label_patch[count] = self.arguement(hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :], rotTimes, vFlip, hFlip)
                    # hrmsi_patch[count] = self.arguement(hrmsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :], rotTimes, vFlip, hFlip)
                    # lrhsi_patch[count] = self.arguement(lrhsi[x:x + patch_size, y:y + patch_size, :], rotTimes, vFlip, hFlip)
                    label_patch[count] = hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_size, y:y + patch_size, :]
                    count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(0, 1))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[::-1, :, :].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, index):
        hrhsi = np.transpose(self.label[index], (2,0,1))
        hrmsi = np.transpose(self.msi_data[index], (2,0,1))
        lrhsi = np.transpose(self.hsi_data[index], (2,0,1))

        hrhsi = torch.tensor(hrhsi, dtype=torch.float32)
        hrmsi = torch.tensor(hrmsi, dtype=torch.float32)
        lrhsi = torch.tensor(lrhsi, dtype=torch.float32)

        return lrhsi, hrmsi, hrhsi

    def __len__(self):
        return self.label.shape[0]


# 由于PaviaU数据集是单图,数据集划分较为多样
# 1.我们将其打成patch之后，再进行训练集和测试集的划分
# 2.直接随机切指定尺寸的patch作为测试集，剩余作为训练集
# 补充说明，可以加1个bool变量来控制划分方式
class PaviaUDataset(Dataset):
    def __init__(self, data_path, gen_path, patch_size=8, stride=32, ratio=8, kerSize=8, sigma=2, split_ratio=(0.8, 0.1, 0.1), type='train'):
        super().__init__()
        self.data_path = data_path
        self.stride = stride
        self.rows = 610  # LR-HSI rows = HR-HSI rows / ratio
        self.cols = 340  # LR-HSI cols
        self.ratio = ratio
        self.sigma = sigma
        self.patch_size = patch_size
        self.kerSize = kerSize
        self.type = type
        self.B = gauss_kernel(kerSize, kerSize, sigma)
        self.R = create_spec_resp(3, gen_path)

        # 读取单幅高光谱图像
        hrhsi = sio.loadmat(data_path)['HS'].astype(np.float32)
        hrmsi, lrhsi = create_hrms_lrhs(hrhsi, self.B, self.R, self.ratio)

        # 生成patch
        hr_size = patch_size * ratio
        lr_size = patch_size
        stride = self.stride
        H, W, _ = hrhsi.shape
        hrhs_list, hrms_list, lrhs_list = [], [], []
        n_rows = (H - hr_size) // stride + 1
        n_cols = (W - hr_size) // stride + 1
        for i in range(n_rows):
            for j in range(n_cols):
                hr_row = i * stride
                hr_col = j * stride
                lr_row = i * stride // ratio
                lr_col = j * stride // ratio
                hrhs_patch = hrhsi[hr_row: hr_row + hr_size, hr_col: hr_col + hr_size, :]
                hrms_patch = hrmsi[hr_row: hr_row + hr_size, hr_col: hr_col + hr_size, :]
                lrhs_patch = lrhsi[lr_row: lr_row + lr_size, lr_col: lr_col + lr_size, :]
                hrhs_list.append(hrhs_patch)
                hrms_list.append(hrms_patch)
                lrhs_list.append(lrhs_patch)

        # 划分train/val/test
        (train_hrhs, train_hrms, train_lrhs), (val_hrhs, val_hrms, val_lrhs), (test_hrhs, test_hrms, test_lrhs) = self.split_patches(
            hrhs_list, hrms_list, lrhs_list, split_ratio=split_ratio
        )

        if self.type == 'train':
            self.hrhs_list, self.hrms_list, self.lrhs_list = train_hrhs, train_hrms, train_lrhs
        elif self.type == 'eval' or self.type == 'val':
            self.hrhs_list, self.hrms_list, self.lrhs_list = val_hrhs, val_hrms, val_lrhs
        elif self.type == 'test':
            self.hrhs_list, self.hrms_list, self.lrhs_list = test_hrhs, test_hrms, test_lrhs
        else:
            raise ValueError(f"Unknown type: {self.type}")

    def split_patches(self, hrhs_list, hrms_list, lrhs_list, split_ratio=(0.8, 0.1, 0.1)):
        total = len(hrhs_list)
        indices = np.arange(total)
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        n_train = int(total * split_ratio[0])
        n_val = int(total * split_ratio[1])
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        train = ([hrhs_list[i] for i in train_idx],
                 [hrms_list[i] for i in train_idx],
                 [lrhs_list[i] for i in train_idx])
        val = ([hrhs_list[i] for i in val_idx],
               [hrms_list[i] for i in val_idx],
               [lrhs_list[i] for i in val_idx])
        test = ([hrhs_list[i] for i in test_idx],
                [hrms_list[i] for i in test_idx],
                [lrhs_list[i] for i in test_idx])
        return train, val, test

    def __len__(self):
        return len(self.hrhs_list)

    def __getitem__(self, index):
        hrhs = torch.from_numpy(np.ascontiguousarray(self.hrhs_list[index].astype(np.float32).transpose(2, 0, 1)))
        hrms = torch.from_numpy(np.ascontiguousarray(self.hrms_list[index].astype(np.float32).transpose(2, 0, 1)))
        lrhs = torch.from_numpy(np.ascontiguousarray(self.lrhs_list[index].astype(np.float32).transpose(2, 0, 1)))
        return  lrhs, hrms, hrhs

class HoustonDataset:
    pass

class ChikuseiDataset:
    pass


class BotswanaDataset:
    pass

class XiongAnDataset:
    pass

class WDCMDataset:
    pass

if __name__ == '__main__':
    #! Launch dataset.py independently: 
    #! -------------terminal---------------
    #$ python -m model.datasets               
    #! ------------------------------------

    data_path = '/yehui/GuidedNet/dataset/PaviaU/PaviaU.mat'
    genPath = '/yehui/GuidedNet/dataset/'

    args = Args()
    data_train = PaviaUDataset(data_path, genPath, type='train')
    trainLoader = DataLoader(data_train,batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    data_eval = PaviaUDataset(data_path,genPath, type='eval')
    evalLoader = DataLoader(data_eval, batch_size=1, num_workers=args.num_workers)
    data_test = PaviaUDataset(data_path, genPath, type='test')
    TestLoader = DataLoader(data_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    dataloader = {'train': trainLoader, 'eval': evalLoader,'test':TestLoader}

    print('-------------------------------------------------')
    print("训练集 batch_size：", dataloader['train'].batch_size)
    print("训练集样本数：", len(data_train))
    print("训练集 batch 数量：", len(dataloader['train']))
    print("验证集 batch_size：", dataloader['eval'].batch_size)
    print("验证集样本数：", len(data_eval))
    print("验证集 batch 数量：", len(dataloader['eval']))
    print("测试集 batch_size：", TestLoader.batch_size)
    print("测试集样本数：", len(data_test))
    print("测试集 batch 数量：", len(TestLoader))
    print('-------------------------------------------------')





