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
            self.hsi_data, self.pan_data, self.label = self.generateTrain(self.patch_size, self.ratio, num_star=1, num_end=20, s=10,auto_patch = True)
        if self.type=='eval':
            self.hsi_data, self.pan_data, self.label = self.generateEval(self.patch_size, self.ratio, num_star=20, num_end=20, s=28,auto_patch = True)
        if self.type=='test':
            self.hsi_data, self.pan_data, self.label = self.generateTest(self.patch_size, self.ratio, num_star=21, num_end=33, s=1, auto_patch = True)

    def generateTrain(self, patch_size, ratio, num_star, num_end, s ,auto_patch=True):
        num_patch = ((self.rows - patch_size) // self.stride + 1) * ((self.cols - patch_size) // self.stride + 1)
        print('-------------------------------------------------')
        print('One Image patch number:', num_patch)
        if auto_patch: # 如果采取自动切片，而不是指定切片数量，s代表图像横轴上patch的数量
            s = int(np.sqrt(num_patch))
        label_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 31), dtype=np.float32)
        pan_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 1), dtype=np.float32)
        lrhsi_patch = np.zeros((s * s * (num_end - num_star), patch_size, patch_size, 31), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            hrhsi = sio.loadmat(self.data_path + '%d.mat' % i)['HS']
            # 生成低分辨率高光谱
            lrhsi = create_hrms_lrhs(hrhsi, self.B, self.R, self.ratio)[1]
            # 生成全色图
            pan = np.mean(hrhsi, axis=2)
            pan = np.expand_dims(pan, axis=2)

            # Data type conversion
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if pan.dtype != np.float32: pan = pan.astype(np.float32)

            for x in range(0, self.rows - patch_size + 1, self.stride):
                for y in range(0, self.cols - patch_size + 1, self.stride):
                    label_patch[count] = hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    pan_patch[count] = pan[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_size, y:y + patch_size, :]
                    count += 1

        return lrhsi_patch, pan_patch, label_patch

    def generateEval(self, patch_size, ratio, num_star, num_end, s,auto_patch=False):
        num_patch = ((self.rows - patch_size) // self.stride + 1) * ((self.cols - patch_size) // self.stride + 1)
        if auto_patch:
            s = int(np.sqrt(num_patch))
        label_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 31), dtype=np.float32)
        pan_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 1), dtype=np.float32)
        lrhsi_patch = np.zeros((s * s * (num_end - num_star), patch_size, patch_size, 31), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            hrhsi = sio.loadmat(self.data_path + '%d.mat' % i)['HS']
            lrhsi = create_hrms_lrhs(hrhsi, self.B, self.R, self.ratio)[1]
            pan = np.mean(hrhsi, axis=2)
            pan = np.expand_dims(pan, axis=2)

            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if pan.dtype != np.float32: pan = pan.astype(np.float32)

            for x in range(0, self.rows - patch_size + 1, self.stride):
                for y in range(0, self.cols - patch_size + 1, self.stride):
                    label_patch[count] = hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    pan_patch[count] = pan[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_size, y:y + patch_size, :]
                    count += 1

        return lrhsi_patch, pan_patch, label_patch

    def generateTest(self, patch_size, ratio, num_star, num_end, s,auto_patch=False):
        num_patch = ((self.rows - patch_size) // self.stride + 1) * ((self.cols - patch_size) // self.stride + 1)
        if auto_patch:
            s = int(np.sqrt(num_patch))
        print(s)
        label_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 31), dtype=np.float32)
        pan_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 1), dtype=np.float32)
        lrhsi_patch = np.zeros((s * s * (num_end - num_star), patch_size, patch_size, 31), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            hrhsi = sio.loadmat(self.data_path + '%d.mat' % i)['HS']
            lrhsi = create_hrms_lrhs(hrhsi, self.B, self.R, self.ratio)[1]
            pan = np.mean(hrhsi, axis=2)
            pan = np.expand_dims(pan, axis=2)

            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if pan.dtype != np.float32: pan = pan.astype(np.float32)

            for x in range(0, self.rows - patch_size + 1, self.stride):
                for y in range(0, self.cols - patch_size + 1, self.stride):
                    label_patch[count] = hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    pan_patch[count] = pan[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_size, y:y + patch_size, :]
                    count += 1
        return lrhsi_patch, pan_patch, label_patch

    def __getitem__(self, index):
        hrhsi = np.transpose(self.label[index], (2,0,1))
        pan = np.transpose(self.pan_data[index], (2,0,1))
        lrhsi = np.transpose(self.hsi_data[index], (2,0,1))

        hrhsi = torch.tensor(hrhsi, dtype=torch.float32)
        pan = torch.tensor(pan, dtype=torch.float32)
        lrhsi = torch.tensor(lrhsi, dtype=torch.float32)

        return lrhsi, pan, hrhsi

    def __len__(self):
        return self.label.shape[0]


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

    args = Args()
    data_path = args.data_path
    genPath = args.gen_path
    data_train = CaveDataset(data_path, genPath, type='train')
    trainLoader = DataLoader(data_train,batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    data_eval = CaveDataset(data_path,genPath, type='eval')
    evalLoader = DataLoader(data_eval, batch_size=1, num_workers=args.num_workers)
    data_test = CaveDataset(data_path, genPath, type='test')
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
