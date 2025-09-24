import torch
from torch.nn.functional import * 
from torch import nn
import torch.nn.functional as F

def get_encoder(d_in, d_model, seq_len, backbone, **kwargs):
    if backbone == 'cnn':
        return CNNEncoder(
            d_inp=d_in,
            dim=d_model,
            seq_len=seq_len
        )
    elif backbone == 'state':
        return StateEncoder(
            feature_size=d_in,
            hidden_size=d_model,
        )
    elif backbone == 'tcn':
        return TCNEncoder(
            in_channels=d_in,
            **kwargs
        )

class StateEncoder(nn.Module):
    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        rnn: str = "GRU",
        dropout: float = 0.5,
        regres: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn
        self.regres = regres

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                feature_size,
                self.hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                feature_size,
                self.hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            )

        self.regressor = nn.Sequential(
            nn.BatchNorm1d(num_features=self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(self, x, mask=None, timesteps=None, return_all: bool = False):
        if self.rnn_type == "GRU":
            all_encodings, encoding = self.rnn(x)
        else:
            all_encodings, (encoding, state) = self.rnn(x)

        if self.regres:
            if return_all:
                reshaped_encodings = all_encodings.reshape(
                    all_encodings.shape[0] * all_encodings.shape[1], -1
                )
                return self.regressor(reshaped_encodings).reshape(
                    all_encodings.shape[0], all_encodings.shape[1], -1
                )
            return self.regressor(encoding.reshape(encoding.shape[1], -1))
            # return self.regressor(encoding.flatten(1))
        return encoding.reshape(encoding.shape[1], -1)

class CNNEncoder(nn.Module):
    def __init__(self, 
                 d_inp,
                 dim,
                 seq_len):
        super().__init__()

        num_channels = [dim] * 3
        conv_blocks = []
        d_in = d_inp
        for d_out in num_channels:
            conv_blocks.append(nn.Conv1d(d_in, d_out, kernel_size=3, padding=1))
            conv_blocks.append(nn.BatchNorm1d(d_out))
            conv_blocks.append(nn.ReLU(inplace=True))
            d_in = d_out
        self.conv_blocks    = nn.Sequential(*conv_blocks)
        self.flatten        = nn.Flatten()
        self.linear         = nn.Linear(seq_len * d_out, d_out)
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv_blocks(x)
        x = self.flatten(x)
        embed = self.linear(x)
        return embed


class CNNClassifier(nn.Module):
    def __init__(self, d_inp, n_classes, dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(d_inp, out_channels=dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, n_classes),
        )
    def forward(self, x, mask=None, timesteps=None, get_embedding=False, captum_input=False, show_sizes=False, return_all=False):

        x = x.transpose(1, 2) 

        if x.shape[-1] < 8:
       
            x = F.pad(x, (0, 8 - x.shape[-1]), mode="constant", value=0)
        if show_sizes:
            print(f"input {x.shape=}")
        embedding = self.encoder(x)  
        if show_sizes:
            print(f"embedding {embedding.shape=}")
        out = self.mlp(embedding) 
        if show_sizes:
            print(f"{out.shape=}")

        if get_embedding:
            return out, embedding
        else:
            return out


class StateClassifier(nn.Module):
    def __init__(
        self,
        feature_size: int,
        n_state: int,
        hidden_size: int,
        rnn: str = "GRU",
        dropout: float = 0.5,
        regres: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.rnn_type = rnn
        self.regres = regres

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                feature_size,
                self.hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                feature_size,
                self.hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            )

        self.regressor = nn.Sequential(
            nn.BatchNorm1d(num_features=self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.n_state),
        )

    def forward(self, x, mask=None, timesteps=None, return_all: bool = False):
        if self.rnn_type == "GRU":
            all_encodings, encoding = self.rnn(x)
        else:
            all_encodings, (encoding, state) = self.rnn(x)

        if self.regres:
            if return_all:
                reshaped_encodings = all_encodings.reshape(
                    all_encodings.shape[0] * all_encodings.shape[1], -1
                )
                return self.regressor(reshaped_encodings).reshape(
                    all_encodings.shape[0], all_encodings.shape[1], -1
                )
            return self.regressor(encoding.reshape(encoding.shape[1], -1))
        return encoding.reshape(encoding.shape[1], -1)

class StateClassifier_V0(nn.Module):
    def __init__(
        self,
        feature_size: int,
        n_state: int,
        hidden_size: int,
        rnn: str = "GRU",
        dropout: float = 0.5,
        regres: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.rnn_type = rnn
        self.regres = regres

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                feature_size,
                self.hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                feature_size,
                self.hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            )

        self.regressor = nn.Sequential(
            nn.BatchNorm1d(num_features=self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.n_state),
        )

    def forward(self, x, mask=None, timesteps=None, return_all: bool = False):
        if self.rnn_type == "GRU":
            all_encodings, encoding = self.rnn(x)
        else:
            all_encodings, (encoding, state) = self.rnn(x)

        if self.regres:
            if return_all:
                reshaped_encodings = all_encodings.reshape(
                    all_encodings.shape[0] * all_encodings.shape[1], -1
                )
                return self.regressor(reshaped_encodings).reshape(
                    all_encodings.shape[0], all_encodings.shape[1], -1
                )
            return self.regressor(encoding.reshape(encoding.shape[1], -1))
        return encoding.reshape(encoding.shape[1], -1)
    

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=1, padding=0, dilation=dilation, bias=bias, groups=groups)
        self.pad_left = (kernel_size - 1) * dilation

    def forward(self, x):
        if self.pad_left > 0:
            x = F.pad(x, (self.pad_left, 0)) 
        return super().forward(x)

class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1, groups_gn=16):
        super().__init__()
        c = channels
        self.conv1 = CausalConv1d(c, c, kernel_size=kernel_size, dilation=dilation, bias=False)
        self.gn1   = nn.GroupNorm(num_groups=min(groups_gn, c), num_channels=c)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(c, c, kernel_size=kernel_size, dilation=dilation, bias=False)
        self.gn2   = nn.GroupNorm(num_groups=min(groups_gn, c), num_channels=c)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.drop1(F.relu(self.gn1(self.conv1(x))))
        x = self.drop2(F.relu(self.gn2(self.conv2(x))))
        return F.relu(x + residual)

class TCNClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int,
                 n_blocks: int = 6, channels: int = 128,
                 kernel_size: int = 3, dilation_base: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, channels, kernel_size=1, bias=False)
        blocks = []
        for i in range(n_blocks):
            dil = dilation_base ** i
            blocks.append(TCNBlock(channels, kernel_size=kernel_size, dilation=dil, dropout=dropout))
        self.net = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(channels, num_classes)
        self.apply(kaiming_init)

    def forward(self, x, mask=None, timesteps=None, get_embedding=False, captum_input=False, show_sizes=False, return_all=False):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = self.net(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


def kaiming_init(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)

        
class TCNEncoder(nn.Module):
    def __init__(self, in_channels: int,
                 n_blocks: int = 6, channels: int = 128,
                 kernel_size: int = 3, dilation_base: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, channels, kernel_size=1, bias=False)
        blocks = []
        for i in range(n_blocks):
            dil = dilation_base ** i
            blocks.append(TCNBlock(channels, kernel_size=kernel_size, dilation=dil, dropout=dropout))
        self.net = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # self.fc  = nn.Linear(channels, num_classes)
        self.apply(kaiming_init)

    def forward(self, x, mask=None, timesteps=None, get_embedding=False, captum_input=False, show_sizes=False, return_all=False):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = self.net(x)
        x = self.gap(x).squeeze(-1)
        return x