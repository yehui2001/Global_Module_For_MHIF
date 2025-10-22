from functools import partial
from numpy import True_
from torch.utils.data import DataLoader
from mlgb.datasets_pan import CaveDataset, QuickBirdDataset
import random
import torch
import numpy as np

def seed_worker(worker_id):
    """
    保证 DataLoader 多进程情况下可复现
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader(args):
    if (args.dataset == 'CAVE'):
        data_train = CaveDataset(args.data_path,args.gen_path,args.patch_size,args.stride,args.ratio,type='train')
        trainLoader = DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(args.seed))
        data_eval = CaveDataset(args.data_path,args.gen_path,args.patch_size,args.stride,args.ratio,type='eval')
        evalLoader = DataLoader(data_eval, batch_size=1, num_workers=args.num_workers,shuffle=False,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(args.seed))
        data_test = CaveDataset(args.data_path,args.gen_path,args.patch_size,args.stride,args.ratio,type='test')
        testLoader = DataLoader(data_test, batch_size=1, num_workers=args.num_workers,shuffle=False,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(args.seed))
        dataloader = {'train': trainLoader, 'eval': evalLoader, 'test': testLoader}
        print_info(dataloader,args) # 打印相关参数
    elif (args.dataset == 'QuickBird'):
        data_train = QuickBirdDataset(args.ms_data_path,args.pan_data_path,args.patch_size,args.stride,args.ratio,type='train')
        trainLoader = DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(args.seed))
        data_eval = QuickBirdDataset(args.ms_data_path,args.pan_data_path,args.patch_size,args.stride,args.ratio,type='train')
        evalLoader = DataLoader(data_eval, batch_size=1, num_workers=args.num_workers, shuffle=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(args.seed))
        data_test = QuickBirdDataset(args.ms_data_path,args.pan_data_path,args.patch_size,args.stride,args.ratio,type='test')
        testLoader = DataLoader(data_test, batch_size=1, num_workers=args.num_workers,shuffle=False,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(args.seed))
        dataloader = {'train': trainLoader, 'test': testLoader}
        print_info(dataloader,args) # 打印相关参数 

    elif (args.dataset == 'WDCM'):
        pass
    else:
        raise SystemExit('Error: no such type of dataset!')
    
    return dataloader

def print_info(dataloader,args):
    data_train = dataloader['train'].dataset
    data_eval = dataloader['eval'].dataset
    data_test = dataloader['test'].dataset
    print('--------------INFORMATION------------------------')
    print(f'Datasets:{args.dataset}')
    print(f'LRHSI patch_size:{args.patch_size}')
    print(f'LRHSI stride:{args.stride}')
    print(f'Ratio:{args.ratio}')
    print(f'Device:{args.device}')
    print(f'Init_lr:{args.init_lr}')
    print("训练集 batch_size：", dataloader['train'].batch_size)
    print("训练集 样本数：", len(data_train))
    print("训练集 batch 数量：", len(dataloader['train']))
    print("验证集 batch_size：", dataloader['eval'].batch_size)
    print("验证集 样本数：", len(data_eval))
    print("验证集 batch 数量：", len(dataloader['eval']))
    print("测试集 batch_size：", dataloader['test'].batch_size)
    print("测试集样本数：", len(data_test))
    print("测试集 batch 数量：", len(dataloader['test']))
    print("总样本数: ", len(data_train) + len(data_eval) + len(data_test))
    print('-------------------------------------------------')