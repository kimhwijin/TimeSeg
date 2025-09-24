import sys
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchrl.objectives.value.functional import generalized_advantage_estimate
from torchrl.objectives.value import GAE
from torchrl.modules import ValueOperator

# 
# from backbones import get_default_backbone
from models.layers import get_encoder


class ValueNetwork(nn.Module):
    def __init__(self, d_in, d_model, d_out, seq_len, backbone):
        super().__init__()
        self.backbone = get_encoder(d_in, d_model, seq_len, backbone)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_out),
        )

    def forward(self, x, curr_mask):   
        inputs = torch.concat([x, curr_mask], dim=-1)
        # B x T x D
        z = self.backbone(inputs)
        z = z.reshape(z.shape[0], -1)
        z = F.tanh(z)
        out = self.proj(z)
        return out



def advantage_module(value_module, td):
    gamma = 0.99
    lmbda = 0.95
    
    value_module(td)
    value_module(td['next'])

    value = td['state_value']
    next_value = td['next']['state_value']

    reward = td['next']['reward']
    done = td['next']['done']
    
    adv, value_target = generalized_advantage_estimate(
        gamma,
        lmbda,
        value,
        next_value,
        reward,
        done=done,
        terminated=done
    )
    td.set('advantage', adv)
    td.set('value_target', value_target)
    return td
    

