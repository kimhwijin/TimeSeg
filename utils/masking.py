import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import timesynth as ts

def MaskingFunction(mask_type, train_set):
    if mask_type == 'seq':
        return SeqCombMask()
    elif mask_type == 'zero':
        return ZeroMask()
    elif mask_type == 'mean':
        return MeanMask()
    elif mask_type == 'normal':
        mean    = train_set.data['x'].mean(0)
        std     = train_set.data['x'].std(0)
        return GaussianMasking(mean, std)

class ZeroMask(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, mask):
        return x * mask
    
class MeanMask(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, mask):
        return x * mask + x.mean(1, keepdim=True) * torch.logical_not(mask)
    

class SeqCombMask(nn.Module):
    def __init__(self):
        super().__init__()
        noise = ts.noise.GaussianNoise(std=0.01)
        x = ts.signals.NARMA(order=2)
        self.x_ts = ts.TimeSeries(x, noise_generator=noise)
        
    def forward(self, x, mask):
        B, T, D = x.shape
        x_star, _, _ = self.x_ts.sample(torch.arange(T).repeat(B*D))
        x_star = x_star.reshape(B, T, D)
        x_star = torch.tensor(x_star, dtype=x.dtype, device=x.device)

        return x * mask + x_star * torch.logical_not(mask)
    
class GaussianMasking(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.dist = D.Normal(loc=mean, scale=std)
    def forward(self, x, mask):
        B = x.shape[0]
        device = x.device
        x_star = self.dist.sample([B]).to(device)
        
        return x * mask + x_star * torch.logical_not(mask)