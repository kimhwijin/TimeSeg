import torch
import itertools
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable

import warnings
from typing import Optional, Tuple

from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.functional import * # Import all needed tools for differentiable attn masking
import math

from torch import nn
import torch.nn.functional as F
from typing import List

def get_encoder(d_in, d_model, seq_len, backbone, **kwargs):
    if backbone == 'cnn':
        return CNNEncoder(
            d_inp=d_in,
            dim=d_model,
            seq_len=seq_len
        )
    
    if backbone == 'cnn_v0':
        return CNNEncoder_V0(
            d_inp=d_in,
            dim=d_model,
            seq_len=seq_len
        )
    if backbone == 'cnn_v1':
        return CNNEncoder_V1(
            d_inp=d_in,
            dim=d_model,
            seq_len=seq_len
        )
    if backbone == 'cnn_v2':
        return CNNEncoder_V2(
            d_inp=d_in,
            dim=d_model,
            seq_len=seq_len
        )
    elif backbone == 'state_v0':
        return StateEncoder_V0(
            feature_size=d_in,
            hidden_size=d_model,
        )
    elif backbone == 'state_v1':
        return StateEncoder_V1(
            feature_size=d_in,
            hidden_size=d_model,
        )
    elif backbone == 'state':
        return StateEncoder(
            feature_size=d_in,
            hidden_size=d_model,
        )
    elif backbone == 'transformer':
        trans_config = {
            'd_inp': d_in,
            'd_model': d_model,
            'nhead': 1,
            # 'nhid': 2 * 36,
            'nlayers': 1,
            'enc_dropout': 0.3,
            'max_len': seq_len,
            'd_static': 0,
            'MAX': seq_len,
            'aggreg': 'mean',
            'static': False,
        }
        return TransformerEncoder(**trans_config)

    elif backbone == 'inception':
        return InceptionEncoder(
            seq_len=seq_len,
            in_channels=d_in,
            **kwargs
        )
    elif backbone == 'causal_inception':
        return CausalInceptionEncoder(
            in_channels=d_in,
            **kwargs
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

        # Input to torch LSTM should be of size (batch, seq_len, input_size)
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
class StateEncoder_V1(nn.Module):
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

        # Input to torch LSTM should be of size (batch, seq_len, input_size)
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
            # nn.BatchNorm1d(num_features=self.hidden_size),
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

class StateEncoder_V0(nn.Module):
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

        # Input to torch LSTM should be of size (batch, seq_len, input_size)
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
            # nn.Linear(self.hidden_size, self.hidden_size),
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


class CNNEncoder_V0(nn.Module):
    def __init__(self, 
                 d_inp,
                 dim,
                 seq_len):
        super().__init__()

        num_channels = [dim] * 3
        conv_blocks = []
        d_in = d_inp
        for d_out in num_channels:
            conv_blocks.append(nn.Conv1d(d_in, d_out, kernel_size=7, padding='same'))
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
    
class CNNEncoder_V1(nn.Module):
    def __init__(self, 
                 d_inp,
                 dim,
                 seq_len):
        super().__init__()

        num_channels = [dim] * 3
        conv_blocks = []
        d_in = d_inp
        for d_out in num_channels:
            conv_blocks.append(nn.Conv1d(d_in, d_out, kernel_size=9, padding='same'))
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
    
    
class CNNEncoder_V2(nn.Module):
    def __init__(self, 
                 d_inp,
                 dim,
                 seq_len):
        super().__init__()

        num_channels = [dim] * 3
        conv_blocks = []
        d_in = d_inp
        for d_out in num_channels:
            conv_blocks.append(nn.Conv1d(d_in, d_out, kernel_size=12, padding='same'))
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
    
# class CNNEncoder(nn.Module):
#     def __init__(self, d_inp, dim=128):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(d_inp, out_channels=dim, kernel_size=7, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(dim, dim, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(dim, dim, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.AdaptiveAvgPool1d(1),
#             nn.Flatten(),
#         )
#     def forward(self, x, mask=None, timesteps=None, ):
#         # Input x shape: (B, T, F)
#         # Need to convert to: (B, F, T) for Conv1d
#         x = x.transpose(1, 2)  # (B, F, T)
#         embedding = self.encoder(x)  # (B, dim)
#         return embedding
        

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
        # Input x shape: (B, T, F)
        # Need to convert to: (B, F, T) for Conv1d
        x = x.transpose(1, 2)  # (B, F, T)

        if x.shape[-1] < 8:
            # pad sequence to at least 8 so two max pools don't fail
            # necessary for when WinIT uses a small window
            x = F.pad(x, (0, 8 - x.shape[-1]), mode="constant", value=0)
        if show_sizes:
            print(f"input {x.shape=}")
        embedding = self.encoder(x)  # (B, dim)
        if show_sizes:
            print(f"embedding {embedding.shape=}")
        out = self.mlp(embedding)  # (B, n_classes)
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

        # Input to torch LSTM should be of size (batch, seq_len, input_size)
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

        # Input to torch LSTM should be of size (batch, seq_len, input_size)
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
    
class TransformerEncoder(nn.Module):

    def __init__(self, 
            d_inp,  # Dimension of input from samples (must be constant)
            max_len, # Max length of any sample to be fed into model
            d_model,
            enc_dropout = None, # Encoder dropout 
            nhead = 1, # Number of attention heads
            trans_dim_feedforward = 72, # Number of hidden layers 
            trans_dropout = 0.25, # Dropout rate in Transformer encoder
            nlayers = 1, # Number of Transformer layers
            aggreg = 'mean', # Aggregation of transformer embeddings
            MAX = 10000, # Arbitrary large number
            static=False, # Whether to use some static vector in additional to time-varying
            d_static = 0, # Dimensions of static input  
            d_pe = 16, # Dimension of positional encoder
            norm_embedding = False,
            time_rand_mask_size = None,
            attn_rand_mask_size = None,
            no_return_attn = True,
            pre_seq_mlp = False,
            stronger_clf_head = False,
            pre_agg_transform = False,
            ):

        super(TransformerEncoder, self).__init__()
        self.model_type = 'Transformer'
        self.d_inp = d_inp
        self.max_len = max_len
        self.d_model = d_model
        self.enc_dropout = enc_dropout
        self.nhead = nhead
        self.trans_dim_feedforward = trans_dim_feedforward
        self.trans_dropout = trans_dropout
        self.nlayers = nlayers
        self.aggreg = aggreg
        self.static = static
        self.d_static = d_static
        self.d_pe = d_pe
        self.norm_embedding = norm_embedding
        self.pre_seq_mlp = pre_seq_mlp
        self.stronger_clf_head = stronger_clf_head
        self.pre_agg_transform = pre_agg_transform

        self.time_rand_mask_size = time_rand_mask_size
        self.attn_rand_mask_size = attn_rand_mask_size
        self.no_return_attn = no_return_attn

        self.pos_encoder = PositionalEncodingTF(self.d_pe, max_len, MAX)

        #Set up Transformer encoder:
        encoder_layers = TransformerEncoderLayerInterpret(
            d_model = self.d_pe + d_inp, #self.d_pe + d_inp
            nhead = self.nhead, 
            dim_feedforward = self.trans_dim_feedforward, 
            dropout = self.trans_dropout,
            batch_first = False)
        #if self.norm_embedding:
            #lnorm = nn.LayerNorm(self.d_pe + d_inp) # self.d_pe + d_inp
            #self.transformer_encoder = TransformerEncoderInterpret(encoder_layers, self.nlayers, norm = lnorm)
        #else:
        self.transformer_encoder = TransformerEncoderInterpret(encoder_layers, self.nlayers)

        # Encode input
        self.MLP_encoder = nn.Linear(self.d_pe + d_inp, d_model)

        if self.pre_seq_mlp:
            self.pre_MLP_encoder = nn.Sequential(
                nn.Linear(d_inp, d_inp),
                nn.PReLU(),
                nn.Linear(d_inp, d_inp),
                nn.PReLU(),
            )

        if self.static:
            self.emb = nn.Linear(self.d_static, d_inp)

        if static == False:
            d_fi = d_inp + self.d_pe
        else:
            d_fi = d_inp + self.d_pe + d_inp

        if self.pre_agg_transform:
            self.pre_agg_net = nn.Sequential(
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
            )

        self.relu = nn.ReLU()

        if self.enc_dropout is not None:
            self.enc_dropout_layer = nn.Dropout(self.enc_dropout)
        else:
            self.enc_dropout_layer = lambda x: x # Identity arbitrary function

        # Initialize weights of module
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.MLP_encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)

    def set_config(self):
        self.config = {
            'd_inp': self.d_inp,
            'max_len': self.max_len,
            'n_classes': self.n_classes,
            'enc_dropout': self.enc_dropout,
            'nhead': self.nhead,
            'trans_dim_feedforward': self.trans_dim_feedforward,
            'trans_dropout': self.trans_dropout,
            'nlayers': self.nlayers,
            'aggreg': self.aggreg,
            'static': self.static,
            'd_static': self.d_static,
            'd_pe': self.d_pe,
            'norm_embedding': self.norm_embedding,
        }

    def embed(self, src, times, static = None, captum_input = False,
            show_sizes = False,
            src_mask = None,
            attn_mask = None,
            aggregate = True,
            get_both_agg_full = False,
        ):
        # print('src at entry', src.isnan().sum())

        if captum_input:
            # Flip from (B, T, d) -> (T, B, d)
            times = times.transpose(0, 1)
            src = src.transpose(0,1) # Flip from (B,T) -> (T,B) 

        if len(src.shape) < 3:
            src = src.unsqueeze(dim=1)

        if (src_mask is None) and torch.any(times < -1e5) and (attn_mask is None):
            src_mask = (times < -1e5).transpose(0,1)
            # if attn_mask is not None:
            #     attn_mask *= src_mask.unsqueeze(-1).repeat(1, 1, attn_mask.shape[-1])
            #     src_mask = None

        if show_sizes:
            print('captum input = {}'.format(captum_input), src.shape, 'time:', times.shape)

        lengths = torch.sum(times > 0, dim=0) # Lengths should be size (B,)
        maxlen, batch_size = src.shape[0], src.shape[1]

        if show_sizes:
            print('torch.sum(times > 0, dim=0)', lengths.shape)

        # Encode input vectors
        #src = self.MLP_encoder(src)

        if self.pre_seq_mlp:
            src = self.pre_MLP_encoder(src)

        if show_sizes:
            print('self.MLP_encoder(src)', src.shape)

        # Must flip times to (T, B) for positional encoder
        # if src.detach().clone().isnan().sum() > 0:
        #     print('src before pe', src.isnan().sum())
        pe = self.pos_encoder(times) # Positional encoder
        pe = pe.to(src.device)
        x = torch.cat([pe, src], axis=2) # Concat position and src

        if pe.isnan().sum() > 0:
            print('pe', pe.isnan().sum())
        if src.detach().clone().isnan().sum() > 0:
            print('src after pe', src.isnan().sum())

        if show_sizes:
            print('torch.cat([pe, src], axis=2)', x.shape)

        if self.enc_dropout is not None:
            x = self.enc_dropout_layer(x)

        if show_sizes:
            print('self.enc_dropout(x)', x.shape)

        if static is not None:
            emb = self.emb(static)

        # Transformer must have (T, B, d)
        # src_key_padding_mask is (B, T)
        # mask is (B*n_heads,T,T) - if None has no effect
        if x.isnan().sum() > 0:
            print('before enc', x.isnan().sum())
        # print(x.shape)
        # raise RuntimeError
        output_preagg, attn = self.transformer_encoder(x, src_key_padding_mask = src_mask, mask = attn_mask)

        if show_sizes:
            print('transformer_encoder', output.shape)

        if self.pre_agg_transform:
            output_preagg = self.pre_agg_net(output_preagg)

        # Aggregation scheme:
        if aggregate:
            # Transformer embeddings through MLP --------------------------------------
            #mask2 = mask.permute(1, 0).unsqueeze(2).long()
            if show_sizes:
                print('mask.permute(1, 0).unsqueeze(2).long()', mask2.shape)

            if self.aggreg == 'mean':
                lengths2 = lengths.unsqueeze(1)
                if src_mask is not None:
                    #import ipdb; ipdb.set_trace()
                    output = torch.sum(output_preagg * (1 - src_mask.transpose(0,1).unsqueeze(-1).repeat(1, 1, output_preagg.shape[-1]).float()), dim=0) / (lengths2 + 1)
                else:
                    output = torch.sum(output_preagg, dim=0) / (lengths2 + 1)
            elif self.aggreg == 'max':
                output, _ = torch.max(output_preagg, dim=0)

            if show_sizes:
                print('self.aggreg: {}'.format(self.aggreg), output.shape)

            if static is not None: # Use embedding of static vector:
                output = torch.cat([output, emb], dim=1)

        if self.norm_embedding and aggregate:
            output = F.normalize(output, dim = -1)
        

        if get_both_agg_full:
            return output, output_preagg

        if aggregate:
            return output
            
        else:
            
            return output_preagg

    def forward(self, 
            src,
            mask = None,
            timesteps = None,
            return_all = False,
            static = None, 
            captum_input = False, # Using captum-style input scheme (src.shape = (B, d, T), times.shape = (B, T))
            show_sizes = False, # Used for debugging
            attn_mask = None,
            src_mask = None,
            get_embedding = False,
            get_agg_embed = False,
            ):
        '''
        * Ensure all inputs are cuda before calling forward method

        Dimensions of inputs:
            (B = batch, T = time, d = dimensions of each time point)
            src = (T, B, d)
            times = (T, B)

        Times must be length of longest sample in dataset, with 0's padded at end

        Params:
            given_time_mask (torch.Tensor): Mask on which to apply before feeding input into transformer encoder
                - Can provide random mask for baseline purposes
            given_attn_mask (torch.Tensor): Mask on which to apply to the attention mechanism
                - Can provide random mask for baseline comparison
        '''
        if timesteps==None:
            timesteps=(
                torch.linspace(0, 1, src.shape[1], device=src.device)
                .unsqueeze(0)
                .repeat(src.shape[0], 1)
            )
        # print(timesteps.shape)
        
        src = src.transpose(0, 1)
        times=timesteps.transpose(0, 1)
        # src_mask=mask

        out, out_full = self.embed(src, times,
            static = static,
            captum_input = captum_input,
            show_sizes = show_sizes,
            attn_mask = attn_mask,
            src_mask = src_mask,
            get_both_agg_full = True)
        out = self.MLP_encoder(out)
        
        if get_embedding:
            return out, out_full
        elif get_agg_embed:
            return out, out, out_full
        else:
            return out
    
class TransformerClassifier(nn.Module):
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length
        MAX  = positional encoder MAX parameter
        n_classes = number of classes
    """

    def __init__(self, 
            d_inp,  # Dimension of input from samples (must be constant)
            max_len, # Max length of any sample to be fed into model
            n_classes, # Number of classes for classification head
            enc_dropout = None, # Encoder dropout 
            nhead = 1, # Number of attention heads
            trans_dim_feedforward = 72, # Number of hidden layers 
            trans_dropout = 0.25, # Dropout rate in Transformer encoder
            nlayers = 1, # Number of Transformer layers
            aggreg = 'mean', # Aggregation of transformer embeddings
            MAX = 10000, # Arbitrary large number
            static=False, # Whether to use some static vector in additional to time-varying
            d_static = 0, # Dimensions of static input  
            d_pe = 16, # Dimension of positional encoder
            norm_embedding = False,
            time_rand_mask_size = None,
            attn_rand_mask_size = None,
            no_return_attn = True,
            pre_seq_mlp = False,
            stronger_clf_head = False,
            pre_agg_transform = False,
            ):

        super(TransformerClassifier, self).__init__()
        self.model_type = 'Transformer'
        self.d_inp = d_inp
        self.max_len = max_len
        self.n_classes = n_classes
        self.enc_dropout = enc_dropout
        self.nhead = nhead
        self.trans_dim_feedforward = trans_dim_feedforward
        self.trans_dropout = trans_dropout
        self.nlayers = nlayers
        self.aggreg = aggreg
        self.static = static
        self.d_static = d_static
        self.d_pe = d_pe
        self.norm_embedding = norm_embedding
        self.pre_seq_mlp = pre_seq_mlp
        self.stronger_clf_head = stronger_clf_head
        self.pre_agg_transform = pre_agg_transform

        self.time_rand_mask_size = time_rand_mask_size
        self.attn_rand_mask_size = attn_rand_mask_size
        self.no_return_attn = no_return_attn

        self.pos_encoder = PositionalEncodingTF(self.d_pe, max_len, MAX)

        #Set up Transformer encoder:
        encoder_layers = TransformerEncoderLayerInterpret(
            d_model = self.d_pe + d_inp, #self.d_pe + d_inp
            nhead = self.nhead, 
            dim_feedforward = self.trans_dim_feedforward, 
            dropout = self.trans_dropout,
            batch_first = False)
        #if self.norm_embedding:
            #lnorm = nn.LayerNorm(self.d_pe + d_inp) # self.d_pe + d_inp
            #self.transformer_encoder = TransformerEncoderInterpret(encoder_layers, self.nlayers, norm = lnorm)
        #else:
        self.transformer_encoder = TransformerEncoderInterpret(encoder_layers, self.nlayers)

        # Encode input
        self.MLP_encoder = nn.Linear(d_inp, d_inp)

        if self.pre_seq_mlp:
            self.pre_MLP_encoder = nn.Sequential(
                nn.Linear(d_inp, d_inp),
                nn.PReLU(),
                nn.Linear(d_inp, d_inp),
                nn.PReLU(),
            )

        if self.static:
            self.emb = nn.Linear(self.d_static, d_inp)

        if static == False:
            d_fi = d_inp + self.d_pe
        else:
            d_fi = d_inp + self.d_pe + d_inp

        # Classification head
        if stronger_clf_head:
            self.mlp = nn.Sequential(
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
                nn.Linear(d_fi, n_classes),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_fi, d_fi),
                nn.ReLU(),
                nn.Linear(d_fi, n_classes),
            )

        if self.pre_agg_transform:
            self.pre_agg_net = nn.Sequential(
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
                nn.Linear(d_fi, d_fi),
                nn.PReLU(),
            )

        self.relu = nn.ReLU()

        if self.enc_dropout is not None:
            self.enc_dropout_layer = nn.Dropout(self.enc_dropout)
        else:
            self.enc_dropout_layer = lambda x: x # Identity arbitrary function

        # Initialize weights of module
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.MLP_encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)

    def set_config(self):
        self.config = {
            'd_inp': self.d_inp,
            'max_len': self.max_len,
            'n_classes': self.n_classes,
            'enc_dropout': self.enc_dropout,
            'nhead': self.nhead,
            'trans_dim_feedforward': self.trans_dim_feedforward,
            'trans_dropout': self.trans_dropout,
            'nlayers': self.nlayers,
            'aggreg': self.aggreg,
            'static': self.static,
            'd_static': self.d_static,
            'd_pe': self.d_pe,
            'norm_embedding': self.norm_embedding,
        }

    def embed(self, src, times, static = None, captum_input = False,
            show_sizes = False,
            src_mask = None,
            attn_mask = None,
            aggregate = True,
            get_both_agg_full = False,
        ):
        # print('src at entry', src.isnan().sum())

        if captum_input:
            # Flip from (B, T, d) -> (T, B, d)
            times = times.transpose(0, 1)
            src = src.transpose(0,1) # Flip from (B,T) -> (T,B) 

        if len(src.shape) < 3:
            src = src.unsqueeze(dim=1)

        if (src_mask is None) and torch.any(times < -1e5) and (attn_mask is None):
            src_mask = (times < -1e5).transpose(0,1)
            # if attn_mask is not None:
            #     attn_mask *= src_mask.unsqueeze(-1).repeat(1, 1, attn_mask.shape[-1])
            #     src_mask = None

        if show_sizes:
            print('captum input = {}'.format(captum_input), src.shape, 'time:', times.shape)

        lengths = torch.sum(times > 0, dim=0) # Lengths should be size (B,)
        maxlen, batch_size = src.shape[0], src.shape[1]

        if show_sizes:
            print('torch.sum(times > 0, dim=0)', lengths.shape)

        # Encode input vectors
        #src = self.MLP_encoder(src)

        if self.pre_seq_mlp:
            src = self.pre_MLP_encoder(src)

        if show_sizes:
            print('self.MLP_encoder(src)', src.shape)

        # Must flip times to (T, B) for positional encoder
        # if src.detach().clone().isnan().sum() > 0:
        #     print('src before pe', src.isnan().sum())
        pe = self.pos_encoder(times) # Positional encoder
        pe = pe.to(src.device)
        x = torch.cat([pe, src], axis=2) # Concat position and src

        if pe.isnan().sum() > 0:
            print('pe', pe.isnan().sum())
        if src.detach().clone().isnan().sum() > 0:
            print('src after pe', src.isnan().sum())

        if show_sizes:
            print('torch.cat([pe, src], axis=2)', x.shape)

        if self.enc_dropout is not None:
            x = self.enc_dropout_layer(x)

        if show_sizes:
            print('self.enc_dropout(x)', x.shape)

        if static is not None:
            emb = self.emb(static)

        # Transformer must have (T, B, d)
        # src_key_padding_mask is (B, T)
        # mask is (B*n_heads,T,T) - if None has no effect
        if x.isnan().sum() > 0:
            print('before enc', x.isnan().sum())
        # print(x.shape)
        # raise RuntimeError
        output_preagg, attn = self.transformer_encoder(x, src_key_padding_mask = src_mask, mask = attn_mask)

        if show_sizes:
            print('transformer_encoder', output.shape)

        if self.pre_agg_transform:
            output_preagg = self.pre_agg_net(output_preagg)

        # Aggregation scheme:
        if aggregate:
            # Transformer embeddings through MLP --------------------------------------
            #mask2 = mask.permute(1, 0).unsqueeze(2).long()
            if show_sizes:
                print('mask.permute(1, 0).unsqueeze(2).long()', mask2.shape)

            if self.aggreg == 'mean':
                lengths2 = lengths.unsqueeze(1)
                if src_mask is not None:
                    #import ipdb; ipdb.set_trace()
                    output = torch.sum(output_preagg * (1 - src_mask.transpose(0,1).unsqueeze(-1).repeat(1, 1, output_preagg.shape[-1]).float()), dim=0) / (lengths2 + 1)
                else:
                    output = torch.sum(output_preagg, dim=0) / (lengths2 + 1)
            elif self.aggreg == 'max':
                output, _ = torch.max(output_preagg, dim=0)

            if show_sizes:
                print('self.aggreg: {}'.format(self.aggreg), output.shape)

            if static is not None: # Use embedding of static vector:
                output = torch.cat([output, emb], dim=1)

        if self.norm_embedding and aggregate:
            output = F.normalize(output, dim = -1)
        

        if get_both_agg_full:
            return output, output_preagg

        if aggregate:
            return output
            
        else:
            
            return output_preagg

    def forward(self, 
            src,
            mask = None,
            timesteps = None,
            return_all = False,
            static = None, 
            captum_input = False, # Using captum-style input scheme (src.shape = (B, d, T), times.shape = (B, T))
            show_sizes = False, # Used for debugging
            attn_mask = None,
            src_mask = None,
            get_embedding = False,
            get_agg_embed = False,
            ):
        '''
        * Ensure all inputs are cuda before calling forward method

        Dimensions of inputs:
            (B = batch, T = time, d = dimensions of each time point)
            src = (T, B, d)
            times = (T, B)

        Times must be length of longest sample in dataset, with 0's padded at end

        Params:
            given_time_mask (torch.Tensor): Mask on which to apply before feeding input into transformer encoder
                - Can provide random mask for baseline purposes
            given_attn_mask (torch.Tensor): Mask on which to apply to the attention mechanism
                - Can provide random mask for baseline comparison
        '''
        if timesteps==None:
            timesteps=(
                torch.linspace(0, 1, src.shape[1], device=src.device)
                .unsqueeze(0)
                .repeat(src.shape[0], 1)
            )
        # print(timesteps.shape)
        
        src = src.transpose(0, 1)
        times=timesteps.transpose(0, 1)
        # src_mask=mask

        out, out_full = self.embed(src, times,
            static = static,
            captum_input = captum_input,
            show_sizes = show_sizes,
            attn_mask = attn_mask,
            src_mask = src_mask,
            get_both_agg_full = True)

        output = self.mlp(out)

        if show_sizes:
            print('self.mlp(output)', output.shape)

        # if self.no_return_attn:
        #     return output
        if get_embedding:
            return output, out_full
        elif get_agg_embed:
            return output, out, out_full
        else:
            return output 
        

class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000,):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1] # Number of batches

        P_time = P_time.float()

        # timescales = self.max_len ** torch.linspace(0, 1, self._num_timescales).to(device) this was numpy
        timescales = self.max_len ** torch.linspace(0, 1, self._num_timescales).to(P_time.device)

        #times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        times = P_time.unsqueeze(2)

        scaled_time = times / torch.Tensor(timescales[None, None, :])
        # Use a 32-D embedding to represent a single time point
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        #pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        #pe = pe.to(device)
        return pe
    

class TransformerEncoderLayerInterpret(nn.TransformerEncoderLayer):
    '''
    Overloaded version of the encoder layer s.t. we can extract self-attention
        - Also implements differentiable attention masking
    '''
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayerInterpret, self).__init__(*args, **kwargs)

        d_model = kwargs['d_model']
        nhead = kwargs['nhead']
        if 'dropout' in kwargs.keys():
            dropout = kwargs['dropout']
        else:
            dropout = None
        batch_first = False
        if 'batch_first' in kwargs.keys():
            batch_first = kwargs['batch_first']
        device = None
        if 'device' in kwargs.keys():
            device = kwargs['device']
        dtype = None
        if 'dtype' in kwargs.keys():
            dtype = kwargs['dtype']

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = MultiHeadAttnMask(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        if (src.dim() == 3 and not self.norm_first and not self.training and
            self.self_attn.batch_first and
            self.self_attn._qkv_same_embed_dim and self.activation_relu_or_gelu and
            self.norm1.eps == self.norm2.eps and
            src_mask is None and
                not (src.is_nested and src_key_padding_mask is not None)):
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )
            if (not torch.overrides.has_torch_function(tensor_args) and
                    # We have to use a list comprehension here because TorchScript
                    # doesn't support generator expressions.
                    all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]) and
                    (not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]))):
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    False,  # norm_first, currently not supported
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    src_mask if src_mask is not None else src_key_padding_mask,  # TODO: split into two args
                )
        x = src
        attn_list = []
        if self.norm_first:
            sa_add, attn = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            attn_list.append(attn)
            x = sa_add #x + sa_add KEY EDIT: Removes residual connection that leaked information when masking input
            x = x + self._ff_block(self.norm2(x))
        else:
            sa_add, attn = self._sa_block(x, src_mask, src_key_padding_mask)
            attn_list.append(attn)
            x = self.norm1(sa_add) #self.norm1(x + sa_add) KEY EDIT: Removes residual connection that leaked information when masking inputs
            x = self.norm2(x + self._ff_block(x))

        return x, attn_list

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        # Modified to output attention weights
        x, attn_weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        return self.dropout1(x), attn_weights

class TransformerEncoderInterpret(nn.TransformerEncoder):
    r'''
    Modified version of Transformer Encoder s.t. we can extract self-attention
    '''

    def __init__(self, *args, **kwargs):
        super(TransformerEncoderInterpret, self).__init__(*args, **kwargs)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        # Add check for custom Transformer Encoder layer:
        if isinstance(first_layer, torch.nn.TransformerEncoderLayer) or isinstance(first_layer, TransformerEncoderLayerInterpret):
            if (not first_layer.norm_first and not first_layer.training and
                    first_layer.self_attn.batch_first and
                    first_layer.self_attn._qkv_same_embed_dim and first_layer.activation_relu_or_gelu and
                    first_layer.norm1.eps == first_layer.norm2.eps and
                    src.dim() == 3 and self.enable_nested_tensor) :
                if src_key_padding_mask is not None and not output.is_nested and mask is None:
                    tensor_args = (
                        src,
                        first_layer.self_attn.in_proj_weight,
                        first_layer.self_attn.in_proj_bias,
                        first_layer.self_attn.out_proj.weight,
                        first_layer.self_attn.out_proj.bias,
                        first_layer.norm1.weight,
                        first_layer.norm1.bias,
                        first_layer.norm2.weight,
                        first_layer.norm2.bias,
                        first_layer.linear1.weight,
                        first_layer.linear1.bias,
                        first_layer.linear2.weight,
                        first_layer.linear2.bias,
                    )
                    if not torch.overrides.has_torch_function(tensor_args):
                        if not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]):
                            if output.is_cuda or 'cpu' in str(output.device):
                                convert_to_nested = True
                                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not())

        attn_per_layer = []
        for mod in self.layers:
            if convert_to_nested:
                output, attn = mod(output, src_mask=mask)
            else:
                output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

            attn_per_layer.append(attn)
        

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_per_layer

class MultiHeadAttnMask(torch.nn.MultiheadAttention):
    def __init__(self, **kwargs):
        super(MultiHeadAttnMask, self).__init__(**kwargs)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
                value will be ignored.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
            :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
            where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
            embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
            returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
            :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
            :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
            head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights)
        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward_differentiable(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward_differentiable(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

def multi_head_attention_forward_differentiable(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True
    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            # Fills parts of attn-mask with key-padding mask
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    bool_attn_mask = False
    if attn_mask is not None and attn_mask.dtype == torch.bool: # Only if bool
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask
        bool_attn_mask = True

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    # NOTE: The below code is outdated PyTorch code that has since been replaced with a 
    #   CPP-style call to attention computation. 
    #   See commit: https://github.com/pytorch/pytorch/commit/4d7ec302202caaf35bb8c997d035c54f0c24e192
    #       in the PyTorch GitHub for when it was changed.

    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)
    if attn_mask is not None and bool_attn_mask:
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
    else:
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_output_weights = softmax(attn_output_weights, dim=-1)

    # Need to zero-out attention values post-softmax:
    if attn_mask is not None and (not (bool_attn_mask)):
        #print('masking')
        attn_output_weights = torch.mul(attn_mask, attn_output_weights) # Differentiable masking of attention
        #print('attn weights', attn_output_weights.sum())
        # Rescale attn values:


    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, p=dropout_p)

    attn_output = torch.bmm(attn_output_weights, v)

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None
    

class InceptionEncoder(nn.Module):
    """
    Keras InceptionTime 기본 설정을 재현하는 Predictor.
    (Classifier head: GAP + Linear)
    """
    def __init__(self,
                 seq_len: int,
                #  num_classes: int,
                 in_channels: int,
                 nb_filters: int = 32,
                 depth: int = 6,
                 kernel_size: int = 41,
                 num_kernels: int = 3,
                 use_residual: bool = True,
                 use_bottleneck: bool = True,
                 bottleneck_size: int = 32,
                 emulate_keras: bool = True):
        super().__init__()
        print(f"[InceptionPredictor-KerasStyle] in={in_channels}, nb_filters={nb_filters}, depth={depth}")

        self.use_residual = use_residual
        layers = []
        res_root_channels = in_channels
        x_ch = in_channels
        last_out_ch = None

        for d in range(depth):
            inc = InceptionLayer(
                in_channels=x_ch,
                nb_filters=nb_filters,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                use_bottleneck=use_bottleneck,
                bottleneck_size=bottleneck_size,
                emulate_keras=emulate_keras,
            )
            layers.append(inc)
            out_ch = (num_kernels + 1) * nb_filters
            last_out_ch = out_ch

            if use_residual and d % 3 == 2:
                layers.append(ResidualBlock(res_root_channels, out_ch))
                res_root_channels = out_ch

            x_ch = out_ch

        self.feature_extractor = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        return: logits (B, num_classes)
        """
        x = x.transpose(1, 2)
        res = x
        for layer in self.feature_extractor:
            if isinstance(layer, ResidualBlock):
                x = layer(res, x)
                res = x
            else:
                x = layer(x)

        x = self.gap(x).squeeze(-1)   # (B, C_last)
        return x

class InceptionClassifier(nn.Module):
    """
    Keras InceptionTime 기본 설정을 재현하는 Predictor.
    (Classifier head: GAP + Linear)
    """
    def __init__(self,
                 seq_len: int,
                 num_classes: int,
                 in_channels: int,
                 nb_filters: int = 32,
                 depth: int = 6,
                 kernel_size: int = 41,
                 num_kernels: int = 3,
                 use_residual: bool = True,
                 use_bottleneck: bool = True,
                 bottleneck_size: int = 32,
                 emulate_keras: bool = True,
                 **kwargs,
                 ):
        super().__init__()
        print(f"[InceptionPredictor-KerasStyle] in={in_channels}, nb_filters={nb_filters}, depth={depth}")

        self.use_residual = use_residual
        layers = []
        res_root_channels = in_channels
        x_ch = in_channels
        last_out_ch = None

        for d in range(depth):
            inc = InceptionLayer(
                in_channels=x_ch,
                nb_filters=nb_filters,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                use_bottleneck=use_bottleneck,
                bottleneck_size=bottleneck_size,
                emulate_keras=emulate_keras,
            )
            layers.append(inc)
            out_ch = (num_kernels + 1) * nb_filters
            last_out_ch = out_ch

            if use_residual and d % 3 == 2:
                layers.append(ResidualBlock(res_root_channels, out_ch))
                res_root_channels = out_ch

            x_ch = out_ch

        self.feature_extractor = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(last_out_ch, num_classes)
        self.num_classes = num_classes 
        self.apply(kaiming_init)

    def forward(self, x, mask=None, timesteps=None, return_all: bool = False):
        """
        x: (B, C, T)
        return: logits (B, num_classes)
        """
        x = x.transpose(1, 2)
        res = x

        for layer in self.feature_extractor:
            if isinstance(layer, ResidualBlock):
                x = layer(res, x)
                res = x
            else:
                x = layer(x)

        x = self.gap(x).squeeze(-1)   # (B, C_last)
        return self.classifier(x)
    

def kaiming_init(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# class InceptionLayer(nn.Module):
#     """
#     Keras InceptionTime 기본 구현을 가깝게 재현:
#       - (선택) Bottleneck 1x1 (고정 32채널)
#       - kernel_size_s = [ (kernel_size-1)//(2**i) ]  (emulate_keras=True)
#       - 3개 Conv 분기 + (MaxPool→1x1) 분기
#       - 분기 concat 후 BN + ReLU
#       - SAME padding (가능하면) / 아니면 수동 길이 보정
#     """
#     def __init__(self,
#                  in_channels: int,
#                  nb_filters: int = 32,
#                  kernel_size: int = 41,
#                  num_kernels: int = 3,
#                  use_bottleneck: bool = True,
#                  bottleneck_size: int = 32,
#                  emulate_keras: bool = True,
#                  force_same_padding: bool = True):
#         super().__init__()
#         self.use_bottleneck = use_bottleneck and in_channels > 1
#         self.nb_filters = nb_filters
#         self.num_kernels = num_kernels
#         self.emulate_keras = emulate_keras
#         self.force_same_padding = force_same_padding

#         # Keras: 내부에서 kernel_size - 1 사용
#         effective_base = kernel_size - 1 if emulate_keras else kernel_size
#         # 커널 목록 (내림)
#         kernel_sizes: List[int] = [max(1, effective_base // (2 ** i)) for i in range(num_kernels)]
#         # (Keras 예: 41 → 내부 40 → [40, 20, 10])

#         # Bottleneck
#         if self.use_bottleneck:
#             self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)
#             conv_in = bottleneck_size
#         else:
#             self.bottleneck = None
#             conv_in = in_channels

#         self.kernel_sizes = kernel_sizes
#         self.conv_layers = nn.ModuleList()
#         self.need_postpad = False  # 수동 same 패딩 필요 여부

#         for ks in kernel_sizes:
#             if force_same_padding:
#                 # PyTorch 1.10+ same padding 지원
#                 try:
#                     conv = nn.Conv1d(conv_in, nb_filters, kernel_size=ks,
#                                      padding='same', bias=False)
#                 except TypeError:
#                     # fallback: manual (even kernel 시 한 칸 부족 → post pad)
#                     pad = ks // 2 if ks % 2 == 1 else ks // 2 - 1
#                     conv = nn.Conv1d(conv_in, nb_filters, kernel_size=ks,
#                                      padding=pad, bias=False)
#                     if ks % 2 == 0:
#                         self.need_postpad = True
#             else:
#                 # 개선 옵션을 끄고 완전 동일 재현이 목적이므로 even kernel 그대로
#                 pad = ks // 2 if ks % 2 == 1 else ks // 2 - 1
#                 conv = nn.Conv1d(conv_in, nb_filters, kernel_size=ks,
#                                  padding=pad, bias=False)
#                 if ks % 2 == 0:
#                     self.need_postpad = True
#             self.conv_layers.append(conv)

#         # MaxPool branch
#         self.pool_branch = nn.Sequential(
#             nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
#             nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)
#         )

#         total_out = (num_kernels + 1) * nb_filters
#         self.bn = nn.BatchNorm1d(total_out)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, C, T)
#         return: (B, (num_kernels+1)*nb_filters, T)
#         """
#         if self.bottleneck is not None:
#             x_in = self.bottleneck(x)
#         else:
#             x_in = x

#         outs = []
#         for conv, ks in zip(self.conv_layers, self.kernel_sizes):
#             o = conv(x_in)
#             # manual even kernel padding 보정
#             if self.need_postpad and ks % 2 == 0 and o.shape[-1] == x.shape[-1] - 1:
#                 o = F.pad(o, (0, 1))
#             outs.append(o)

#         pool_out = self.pool_branch(x)
#         outs.append(pool_out)

#         # 혹시 fallback 조건에서 길이 불일치 발생 시 정렬
#         lengths = [t.shape[-1] for t in outs]
#         if len(set(lengths)) > 1:
#             max_len = max(lengths)
#             outs = [F.pad(t, (0, max_len - t.shape[-1])) for t in outs]

#         x_cat = torch.cat(outs, dim=1)
#         x_cat = self.bn(x_cat)
#         return F.relu(x_cat, inplace=True)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
    def forward(self, x_prev, x_cur):
        return F.relu(self.proj(x_prev) + x_cur, inplace=True)

class InceptionLayer(nn.Module):
    """
    Causal Inception Layer:
      - bottleneck(1x1) (옵션)
      - 여러 kernel sizes 병렬 branch
      - MaxPool branch
      - concat -> BN -> ReLU
      - 모든 conv/pool은 "왼쪽 패딩만" 적용해 causal 보장
    """
    def __init__(self, in_channels, nb_filters=32, kernel_size=41, num_kernels=3,
                 use_bottleneck=True, bottleneck_size=32, emulate_keras=True, causal=False):
        super().__init__()
        self.use_bottleneck = use_bottleneck and in_channels > 1
        self.emulate_keras = emulate_keras
        self.causal = causal
        eff = kernel_size - 1 if emulate_keras else kernel_size
        self.kernel_sizes: List[int] = [max(1, eff // (2 ** i)) for i in range(num_kernels)]

        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)
            conv_in = bottleneck_size
        else:
            self.bottleneck = None
            conv_in = in_channels

        # conv branches (padding=0, forward에서 왼쪽 패딩)
        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, nb_filters, kernel_size=ks, padding=0, bias=False)
            for ks in self.kernel_sizes
        ])

        # pool branch (왼쪽 패딩 후 pooling)
        self.pool_k = 3
        self.pool_1x1 = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        total_out = (num_kernels + 1) * nb_filters
        self.bn = nn.BatchNorm1d(total_out)

    @staticmethod
    def _pad_left(x, pad):  # (left,right)
        return F.pad(x, (pad, 0)) if pad > 0 else x

    def forward(self, x):  # x: (B,C,T)
        x_in = self.bottleneck(x) if self.use_bottleneck else x

        outs = []
        for conv, ks in zip(self.convs, self.kernel_sizes):
            if self.causal:
                x_pad = self._pad_left(x_in, ks - 1)
                o = conv(x_pad)
            else:
                # non-causal same padding (참고용)
                pad = ks // 2 if ks % 2 == 1 else ks // 2 - 1
                o = F.conv1d(x_in, conv.weight, conv.bias, padding=pad)
                if ks % 2 == 0 and o.shape[-1] == x.shape[-1] - 1:
                    o = F.pad(o, (0, 1))
            outs.append(o)

        # pool branch
        if self.causal:
            x_pad = self._pad_left(x, self.pool_k - 1)
            pool = F.max_pool1d(x_pad, kernel_size=self.pool_k, stride=1, padding=0)
        else:
            pool = F.max_pool1d(x, kernel_size=self.pool_k, stride=1, padding=1)
        pool = self.pool_1x1(pool)
        outs.append(pool)

        # 길이 정렬 (이론상 동일 T)
        Ls = [t.shape[-1] for t in outs]
        if len(set(Ls)) > 1:
            maxL = max(Ls)
            outs = [F.pad(t, (0, maxL - t.shape[-1])) for t in outs]

        x_cat = torch.cat(outs, dim=1)
        x_cat = self.bn(x_cat)
        return F.relu(x_cat, inplace=True)

class CausalInceptionClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, depth=6,
                 nb_filters=32, kernel_size=41, num_kernels=3,
                 use_residual=True, use_bottleneck=True, dropout=0.0):
        super().__init__()
        self.use_residual = use_residual
        layers = []
        res_root = in_channels
        ch = in_channels
        last_out = None

        for d in range(depth):
            inc = InceptionLayer(
                in_channels=ch, nb_filters=nb_filters,
                kernel_size=kernel_size, 
                num_kernels=num_kernels,
                use_bottleneck=use_bottleneck, 
                emulate_keras=True, 
                causal=True  # ★ causal
            )
            layers.append(inc)
            out_ch = (num_kernels + 1) * nb_filters
            last_out = out_ch
            if use_residual and d % 3 == 2:
                layers.append(ResidualBlock(res_root, out_ch))
                res_root = out_ch
            ch = out_ch

        self.feature = nn.Sequential(*layers)
        self.norm = nn.GroupNorm(num_groups=16, num_channels=last_out)  # BN 대신 GN
        self.drop = nn.Dropout(dropout)
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(last_out, num_classes)
        self.apply(kaiming_init)

    def forward(self, x):  # x: (B,T,C) -> (B,C,T)
        x = x.transpose(1, 2)
        res = x
        for layer in self.feature:
            if isinstance(layer, ResidualBlock):
                x = layer(res, x); res = x
            else:
                x = layer(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)
    

class CausalInceptionEncoder(nn.Module):
    def __init__(self, in_channels: int, depth=6,
                 nb_filters=32, kernel_size=41, num_kernels=3,
                 use_residual=True, use_bottleneck=True, dropout=0.0):
        super().__init__()
        self.use_residual = use_residual
        layers = []
        res_root = in_channels
        ch = in_channels
        last_out = None

        for d in range(depth):
            inc = InceptionLayer(
                in_channels=ch, nb_filters=nb_filters,
                kernel_size=kernel_size, num_kernels=num_kernels,
                use_bottleneck=use_bottleneck, emulate_keras=True, causal=True  # ★ causal
            )
            layers.append(inc)
            out_ch = (num_kernels + 1) * nb_filters
            last_out = out_ch
            if use_residual and d % 3 == 2:
                layers.append(ResidualBlock(res_root, out_ch))
                res_root = out_ch
            ch = out_ch

        self.feature = nn.Sequential(*layers)
        self.norm = nn.GroupNorm(num_groups=16, num_channels=last_out)  # BN 대신 GN
        self.drop = nn.Dropout(dropout)
        self.gap  = nn.AdaptiveAvgPool1d(1)
        # self.fc   = nn.Linear(last_out, num_classes)
        self.apply(kaiming_init)

    def forward(self, x):  # x: (B,T,C) -> (B,C,T)
        x = x.transpose(1, 2)
        res = x
        for layer in self.feature:
            if isinstance(layer, ResidualBlock):
                x = layer(res, x); res = x
            else:
                x = layer(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)
        return x
    

    

class CausalConv1d(nn.Conv1d):
    """왼쪽 패딩만 적용하는 causal conv (padding=0으로 둔 뒤 forward에서 수동 패딩)."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=False, groups=1):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=1, padding=0, dilation=dilation, bias=bias, groups=groups)
        self.pad_left = (kernel_size - 1) * dilation

    def forward(self, x):
        if self.pad_left > 0:
            x = F.pad(x, (self.pad_left, 0))  # (left, right)
        return super().forward(x)

class TCNBlock(nn.Module):
    """Causal Dilated Conv -> GN -> ReLU -> Dropout, 2회 + Residual"""
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
    """
    TCN: Causal + Dilated + Residual
    - in_channels: 특성 채널 수
    - n_blocks: 블록 수
    - dilation_base: 1,2,4,... 지수 확장
    """
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
    
class TCNEncoder(nn.Module):
    """
    TCN: Causal + Dilated + Residual
    - in_channels: 특성 채널 수
    - n_blocks: 블록 수
    - dilation_base: 1,2,4,... 지수 확장
    """
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