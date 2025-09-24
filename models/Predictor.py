import sys
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)


import torch
import torch.nn as nn
import torch.nn.functional as F
# 
# from backbones import get_default_backbone
from models.layers import get_encoder


class PredictorNetwork(nn.Module):
    def __init__(self, d_in, d_model, d_out, seq_len, backbone):
        super().__init__()
        self.backbone = get_encoder(d_in, d_model, seq_len, backbone)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_out),
        )

    def forward(self, x):   
        # B x T x D
        z = self.backbone(x)
        z = F.gelu(z)
        out = self.proj(z)
        return out


