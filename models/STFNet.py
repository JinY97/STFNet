'''
Author: Yin Jin
Date: 2022-03-06 13:23:20
LastEditTime: 2023-11-22 17:34:34
LastEditors: JinYin
Description: transformer模型作为输入
Reference: Transformer for EEG classification
'''
import numpy as np
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

class EmbeddingLayer(nn.Module):
    def __init__(self, data_num=540, chan=19, emb_size=32):
        super(EmbeddingLayer, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(chan, emb_size, 7, stride=1, padding=3)        # 128*19*540->128*32*540
        )
    def forward(self, x):        
        x = self.projection(x)
        return x
    
class ChannelAttention(nn.Module):
    def __init__(self, emb_size, num_heads, t_size, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(t_size, t_size)
        self.queries = nn.Linear(t_size, t_size)
        self.values = nn.Linear(t_size, t_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(t_size, t_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b c (h d) -> b h c d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b c (h d) -> b h d c", h=self.num_heads)
        values = rearrange(self.values(x), "b c (h d) -> b h c d", h=self.num_heads)
        energy = torch.matmul(queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.matmul(att, values)
        out = rearrange(out, "b h c d -> b c (h d)")
        out = self.projection(out)
        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class TemporalFeatureBlock(nn.Sequential):
    def __init__(self, t_size, drop_p):
        super().__init__(
            nn.Linear(t_size, t_size), nn.GELU(), nn.Dropout(drop_p),
            nn.Linear(t_size, t_size), nn.GELU(), nn.Dropout(drop_p)
        )

class MultidimensionFeatureExtractor(nn.Sequential):
    def __init__(self,
                 emb_size,
                 t_size,
                 num_heads=8,
                 drop_p=0.1,
                 forward_expansion=1,
                 forward_drop_p=0.1):
        super().__init__( 
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(t_size),
                ChannelAttention(emb_size, num_heads=20, t_size=t_size, dropout=drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(t_size),
                TemporalFeatureBlock(t_size, drop_p=forward_drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(t_size),
                TemporalFeatureBlock(t_size, drop_p=forward_drop_p),
            )),
        )
        
class FeatureExtractor(nn.Sequential):
    def __init__(self, depth, emb_size, t_size):
        super().__init__(*[MultidimensionFeatureExtractor(emb_size, t_size) for _ in range(depth)])
        
class DenoisingHeader(nn.Sequential):
    def __init__(self, emb_size, chan):
        super().__init__()
        self.outputhead = nn.Sequential(
            nn.Conv1d(emb_size, emb_size, 7, stride=1, padding=3),
            nn.Conv1d(emb_size, chan, 7, stride=1, padding=3)
        )
    
    def forward(self, x):
        out = self.outputhead(x)
        return out

class STFNet(nn.Sequential):
    def __init__(self, data_num=500, emb_size=32, depth=6, chan=19, **kwargs):
        super(STFNet, self).__init__()
        self.embedding = EmbeddingLayer(data_num, chan, emb_size)
        self.encoder = FeatureExtractor(depth, emb_size, data_num)
        self.out = DenoisingHeader(emb_size, chan)
    
    def forward(self, x):
        res = self.embedding(x)
        x = self.encoder(res)
        out = self.out(x)

        return out
        
if __name__ == '__main__':
    x = torch.rand((8, 19, 500))
    model = STFNet(data_num=500, emb_size=64, depth=3, chan=19)
    y = model(x)
    print(y.shape)
    
