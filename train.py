'''
Author: JinYin
Date: 2023-07-07 09:49:38
LastEditors: JinYin
LastEditTime: 2023-12-15 15:37:04
FilePath: \07_CTFNet\train_semisimulated.py
Description: 
'''
import os
import time
from tqdm import trange
import torch.nn.functional as F
import torch, numpy as np
import torch.optim as optim

from opts import get_opts, get_name
from preprocess.SemiMultichannel import *  

from tools import *
from models import *

SingleChannelNetwork = ['FCNN', 'SimpleCNN', 'ResCNN', 'GCTNet', 'DuoCL', 'NovelCNN']
MultiChannelNetwork = ['EEGANet', 'STFNet']

def train(opts, model):
    name = get_name(opts)
    print('train:', name)
    np.random.seed(opts.seed)
    
    EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data = LoadEEGData(opts.EEG_path, opts.NOS_path, opts.fold)
    train_data = GetEEGData_train(EEG_train_data, NOS_train_data, opts.batch_size, device=opts.device)
    val_data = GetEEGData(EEG_val_data, NOS_val_data, opts.batch_size)
    test_data = GetEEGData(EEG_test_data, NOS_test_data, opts.batch_size)
    
    if opts.denoise_network == 'SimpleCNN':
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9), eps=1e-8)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30], 0.1)
    elif opts.denoise_network == 'ResCNN':
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9), eps=1e-8)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30], 0.1)
    elif opts.denoise_network == 'GCTNet':
        learning_rate = 0.001      # 0.0001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-8)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [160], 0.1)
    elif opts.denoise_network == 'DuoCL':
        learning_rate = 0.001   
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-8)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50], 0.1)
    elif opts.denoise_network == 'NovelCNN':
        learning_rate = 0.0001       # 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9), eps=1e-8)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [200], 0.1)
    elif opts.denoise_network == 'STFNet':
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-8)        # transformeer:0.001
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30], 0.1)
    
    best_val_mse = 10.0   
    f = open(opts.save_path + "result.txt", "a+")
    for epoch in range(opts.epochs):
        model.train()
        losses = []
        
        for batch_id in trange(train_data.len()):
            x, y = train_data.get_batch(batch_id)
            if opts.denoise_network in SingleChannelNetwork:
                x, y = x.reshape(-1, 500).unsqueeze(dim=1), y.reshape(-1, 500)
            p = model(x)
            
            if opts.denoise_network in SingleChannelNetwork:
                p, y = p.reshape(-1, 19, 500), y.reshape(-1, 19, 500)  
            loss = ((p - y) ** 2).mean(dim=-1).mean(dim=-1).mean(dim=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
        
        train_data.random_shuffle()
        train_loss = torch.stack(losses).mean().item()
        scheduler.step()
        
        model.eval()
        losses = []
        for batch_id in range(val_data.len()):
            x, y = val_data.get_batch(batch_id)
            if opts.denoise_network in SingleChannelNetwork:
                x, y = x.reshape(-1, 500), y.reshape(-1, 500)
                x, y = torch.Tensor(x).to(opts.device).unsqueeze(dim=1), torch.Tensor(y).to(opts.device)
            else:
                x, y = torch.Tensor(x).to(opts.device), torch.Tensor(y).to(opts.device)
                
            with torch.no_grad():
                p = model(x)
                
                if opts.denoise_network in SingleChannelNetwork:
                    p, y = p.reshape(-1, 19, 500), y.reshape(-1, 19, 500)     
                
                loss = ((p - y) ** 2).mean(dim=-1).mean(dim=-1).sqrt().detach()
                losses.append(loss.detach())
        val_mse = torch.cat(losses, dim=0).mean().item()
        
        model.eval()
        rrmse, acc, snr = [], [], []
        
        for batch_id in range(test_data.len()):
            x, y = test_data.get_batch(batch_id)
            if opts.denoise_network in SingleChannelNetwork:
                x, y = x.reshape(-1, 500), y.reshape(-1, 500)
                x, y = torch.Tensor(x).to(opts.device).unsqueeze(dim=1), torch.Tensor(y).to(opts.device)
            else:     
                x, y = torch.Tensor(x).to(opts.device), torch.Tensor(y).to(opts.device)
            
            with torch.no_grad():
                p = model(x)
                
                if opts.denoise_network in SingleChannelNetwork:
                    p, y = p.reshape(-1, 19, 500), y.reshape(-1, 19, 500) 
                p, y = p.cpu().numpy(), y.cpu().numpy()
                for i in range(p.shape[0]):
                    rrmse.append(rrmse_multichannel(p[i], y[i]))
                    acc.append(acc_multichannel(p[i], y[i]))
                    snr.append(cal_SNR_multichannel(p[i], y[i]))
        
        test_rrmse = np.mean(np.array(rrmse))
        test_acc = np.mean(np.array(acc))
        test_snr = np.mean(np.array(snr))
        
        # Save best model
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_acc = test_acc
            best_snr = test_snr
            best_rrmse = test_rrmse
            print("Save best result")
            f.write("Save best result \n")
            torch.save(model, f"{opts.save_path}/best.pth")
        
        print(f"train_loss:{train_loss}")
        print('epoch: {:3d}, val_mse: {:.4f}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(epoch, val_mse, test_rrmse, test_acc, test_snr))
        f.write('epoch: {:3d}, val_mse: {:.4f}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}\n'.format(epoch, val_mse, test_rrmse, test_acc, test_snr))

    with open(os.path.join('./json_file/{}_{}_{}.log'.format(opts.denoise_network, opts.depth, opts.epochs)), 'a+') as fp:
        fp.write('fold:{}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(opts.fold, best_rrmse, best_acc, best_snr) + "\n")
        
if __name__ == '__main__':
    opts = get_opts()
    
    for opts.fold in range(10):
        print(f"fold:{opts.fold }")
        if opts.denoise_network == 'SimpleCNN':
            model = SimpleCNN(data_num=500).to(opts.device)
        elif opts.denoise_network == 'ResCNN':
            model = ResCNN(data_num=500).to(opts.device)
        elif opts.denoise_network == 'NovelCNN':
            model = NovelCNN(data_num=500).to(opts.device)
        elif opts.denoise_network == 'DuoCL':
            model = DuoCL(data_num=500).to(opts.device)
        elif opts.denoise_network == 'GCTNet':
            model = Generator(data_num=500).to(opts.device)
        elif opts.denoise_network == 'STFNet':
             model = STFNet(data_num=500, emb_size=32, depth=opts.depth, chan=19).to(opts.device)
            
        print(opts.denoise_network)
        
        opts.save_path = r"{}/STFNet/{}_{}/{}_{}_{}_{}/".format(opts.save_dir, opts.denoise_network, opts.depth, opts.denoise_network, opts.depth, opts.epochs, opts.fold)
        if not os.path.exists(opts.save_path):
            os.makedirs(opts.save_path)
        
        train(opts, model)