import sys
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import get_encoder

class PolicyNetwork(nn.Module):
    def __init__(self, d_in, d_model, d_start, d_end, seq_len, backbone):
        super().__init__()
        
        self.start_encoder = get_encoder(d_in, d_model, seq_len, backbone)
        self.end_encoder = get_encoder(d_in, d_model, seq_len, backbone)
        
        self.start_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Mish(),
            nn.Linear(d_model, d_start),
        )
        self.end_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Mish(),
            nn.Linear(d_model, d_end),
        )

    def forward(self, x, curr_mask):        
        inputs = torch.concat([x, curr_mask], dim=-1)
        # B x T x D
        start_z = self.start_encoder(inputs)
        end_z = self.end_encoder(inputs)
        start_z = start_z.reshape(start_z.shape[0], -1)
        end_z = end_z.reshape(end_z.shape[0], -1)

        end_z = start_z + end_z
        start_params = self.start_proj(start_z)
        end_params = self.end_proj(end_z)
        
        return start_params, end_params


