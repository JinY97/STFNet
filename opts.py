'''
Author: JinYin
Date: 2023-05-25 17:07:14
LastEditors: JinYin
LastEditTime: 2023-07-27 19:24:51
FilePath: \07_CTTNet\opts.py
Description: 
'''
import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--EEG_path', type=str, default="/hdd/yj_data/01_data/01_SemiSimulated/Pure_Data.mat")
    parser.add_argument('--NOS_path', type=str, default="/hdd/yj_data/01_data/01_SemiSimulated/Contaminated_Data.mat")
    parser.add_argument('--denoise_network', type=str, default='STFNet')
    parser.add_argument('--save_dir', type=str, default= './result/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)

    opts = parser.parse_args()
    return opts

def get_name(opts):
    name = '{}_{}_{}'.format(opts.denoise_network, opts.epochs, opts.batch_size)
    return name
