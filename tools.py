'''
Author: JinYin
Date: 2023-07-06 11:11:13
LastEditors: JinYin
LastEditTime: 2023-12-20 13:53:46
FilePath: \07_CTFNet\tools.py
Description: 
'''
import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.signal import butter, filtfilt

def rrmse_multichannel(predict, truth):
    res = np.sqrt(((predict - truth) ** 2).mean().mean()) / np.sqrt((truth ** 2).mean().mean())
    return res

def acc_multichannel(predict, truth):
    acc = []
    for i in range(predict.shape[0]):
        acc.append(np.corrcoef(predict[i], truth[i])[1, 0])
    return np.mean(np.array(acc))

def cal_SNR_multichannel(predict, truth):
    PS = np.sum(np.sum(np.square(truth), axis=-1), axis=-1)  # power of signal
    PN = np.sum(np.sum(np.square((predict - truth)), axis=-1), axis=-1)  # power of noise
    ratio = PS / PN
    return 10 * np.log10(ratio)

def cal_SNR(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    PS = np.sum(np.square(truth), axis=-1)  # power of signal
    PN = np.sum(np.square((predict - truth)), axis=-1)  # power of noise
    ratio = PS / PN
    return torch.from_numpy(10 * np.log10(ratio))

def Filter_EEG(eeg, f1=0.5, f2=40, fs = 200):
    [b1, a1] = butter(5, [f1/fs * 2, f2/fs * 2], 'bandpass')
    eeg = filtfilt(b1, a1, eeg) 
    
    return eeg

def Standardization(EEG_data, EEG_NOS_data):
    for i in range(EEG_data.shape[0]):
        EEG_data[i] = EEG_data[i] - np.mean(EEG_data[i])
        EEG_NOS_data[i] = EEG_NOS_data[i] - np.mean(EEG_NOS_data[i])
        
        EEG_data[i] = EEG_data[i] / np.std(EEG_NOS_data[i])
        EEG_NOS_data[i] = EEG_NOS_data[i] / np.std(EEG_NOS_data[i])
    return EEG_data, EEG_NOS_data

