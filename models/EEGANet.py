'''
Author: JinYin
Date: 2022-07-31 16:16:10
LastEditors: JinYin
LastEditTime: 2022-10-07 15:02:13
'''
from calendar import c
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, channel=c):
        super(Generator, self).__init__()
        self.layers = 16
        self.input = nn.Sequential(
            nn.Conv1d(channel, 64, (9,), (1,), padding="same"), nn.PReLU()
        )
    
        self.conv = nn.ModuleList()
        for i in range(self.layers):
            self.conv.append(BasicBlock(3,1))
            
        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 64, (3,), (1,), padding="same"), nn.BatchNorm1d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, channel, (9,), (1,), padding="same")
        )

    def forward(self, x):
        x = self.input(x)
        input = x
        for i in range(self.layers):
            x = self.conv[i](x)
        x = torch.add(self.conv1(x), input)
        x = self.conv2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, channel):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channel, 64, (3,), (1,), padding="same"), nn.BatchNorm1d(64), nn.LeakyReLU(0.2)
        )
        self.discriminator = nn.Sequential(
            nn.Conv1d(64, 64, (3,), (2,), padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, (3,), (1,), padding=1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, (3,), (2,), padding=1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, (3,), (1,), padding=1), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, (3,), (2,), padding=1), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, (3,), (1,), padding=1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, (3,), (2,), padding=1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            )
        self.fc1 = nn.Linear(512 * 32, 1024) 
        self.relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)
    
    def forward(self, x):
        x = self.discriminator(self.conv(x)).view(x.shape[0], -1)
        x = self.fc2(self.relu(self.fc1(x)))
        return x
    
class BasicBlock(nn.Module):
    def __init__(self, kernelsize, stride=1):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(64, 64, (kernelsize,), (stride,), padding="same"), nn.BatchNorm1d(64), nn.PReLU(),
            nn.Conv1d(64, 64, (kernelsize,), (stride,), padding="same"), nn.BatchNorm1d(64)
        )    
    
    def forward(self, inputs):
        identity = inputs
        out = self.block(inputs)
        
        output = torch.add(out, identity)
        return output
    
