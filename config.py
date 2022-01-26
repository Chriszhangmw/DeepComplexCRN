import os
import warnings

import torch
import torchvision.transforms as transforms



class Config:
    mode = "CL"


    tr_list = '/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/train.lst'
    dev_list = '/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/dev.lst'
    checkpoint_root = '/home/zmw/big_space/zhangmeiwei_space/asr_res_model/dns/dccrn'

    test_0_list = '/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/test_0.lst'
    test_5_list = '/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/test_5.lst'
    test_10_list = '/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/test_10.lst'
    test_15_list = '/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/test_15.lst'
    test_20_list = '/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/test_20.lst'

    batch_size = 32
    lr = 0.001  # learning_rate
    lr_decay = 0.1
    weight_decay = 1e-5
    verbose_inter = 500
    max_epoch = 40
    save_inter = 5
    device_ids = [3, 4]
    device = device_ids[0]
    sr = 16000
    dim = 4 * 16000

    min_sisnr = 99999999

    #test
    best_path = '/home/zmw/big_space/zhangmeiwei_space/asr_res_model/dns/dccrn/DCCRN_CL_23.pth'




