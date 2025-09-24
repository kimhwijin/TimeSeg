from torch import nn
import torch
import torch.nn.functional as F

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
    
class TCN(nn.Module):
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

    def forward(self, x, _times, get_embedding=False, captum_input=False, show_sizes=False):
        if not captum_input: 
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            # time, batch, channels -> batch, time, channels
            x = x.permute(1, 0, 2)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = self.net(x)
        embedding = self.gap(x).squeeze(-1)
        out = self.fc(embedding)
        if get_embedding:
            return out, embedding
        else:
            return out
    

class CNN(nn.Module):
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

    def forward(self, x, _times, get_embedding=False, captum_input=False, show_sizes=False):
        if captum_input:
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            # batch, time, channels -> batch, channels, time
            x = x.permute(0, 2, 1)
        else: 
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            # time, batch, channels -> batch, channels, time
            x = x.permute(1, 2, 0)

        if x.shape[-1] < 8:
            # pad sequence to at least 8 so two max pools don't fail
            # necessary for when WinIT uses a small window
            x = F.pad(x, (0, 8 - x.shape[-1]), mode="constant", value=0)

        embedding = self.encoder(x)
        out = self.mlp(embedding)

        if get_embedding:
            return out, embedding
        else:
            return out


class LSTM(nn.Module):
    def __init__(self, d_inp, n_classes, dim=128):
        super().__init__()
        self.encoder = nn.LSTM(
            d_inp,
            dim // 2, # half for bidirectional
            num_layers=3,
            batch_first=True,
            bidirectional=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, n_classes),
        )

    def forward(self, x, _times, get_embedding=False, captum_input=False, show_sizes=False):
        if not captum_input: 
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            # time, batch, channels -> batch, time, channels
            x = x.permute(1, 0, 2)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0)

        embedding, _ = self.encoder(x)
        embedding = embedding.mean(dim=1) # mean over time
        out = self.mlp(embedding)

        if get_embedding:
            return out, embedding
        else:
            return out


class GRU(nn.Module):
    def __init__(self, d_inp, n_classes, dim=128, dropout=0.5):
        super().__init__()
        self.encoder = nn.GRU(
            d_inp,
            dim,
            batch_first=True,
            bidirectional=False,
        )

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(num_features=dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, n_classes),
        )

    def forward(self, x, _times, get_embedding=False, captum_input=False, show_sizes=False):
        if not captum_input: 
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            # time, batch, channels -> batch, time, channels
            x = x.permute(1, 0, 2)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0)

        # embedding, _ = self.encoder(x)
        # embedding = embedding.mean(dim=1) # mean over time
        # out = self.mlp(embedding)
        
        _, encoding = self.encoder(x)
        out = self.mlp(encoding.reshape(encoding.shape[1], -1))
        embedding = encoding.squeeze(0)

        if get_embedding:
            return out, embedding
        else:
            return out
        


class Inception(nn.Module):
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
                 emulate_keras: bool = True):
        super().__init__()
        # print(f"[InceptionPredictor-KerasStyle] in={in_channels}, nb_filters={nb_filters}, depth={depth}")

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
                force_same_padding=True
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

    def forward(self, x, _times, get_embedding=False, captum_input=False, show_sizes=False):
        """
        x: (B, C, T)
        return: logits (B, num_classes)
        """
        if not captum_input: 
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            # time, batch, channels -> batch, time, channels
            x = x.permute(1, 0, 2)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.transpose(1, 2)
        res = x

        for layer in self.feature_extractor:
            if isinstance(layer, ResidualBlock):
                x = layer(res, x)
                res = x
            else:
                x = layer(x)

        embedding = self.gap(x).squeeze(-1)   # (B, C_last)
        out = self.classifier(embedding)

        if get_embedding:
            return out, embedding
        else:
            return out
    

def kaiming_init(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class InceptionLayer(nn.Module):
    """
    Keras InceptionTime 기본 구현을 가깝게 재현:
      - (선택) Bottleneck 1x1 (고정 32채널)
      - kernel_size_s = [ (kernel_size-1)//(2**i) ]  (emulate_keras=True)
      - 3개 Conv 분기 + (MaxPool→1x1) 분기
      - 분기 concat 후 BN + ReLU
      - SAME padding (가능하면) / 아니면 수동 길이 보정
    """
    def __init__(self,
                 in_channels: int,
                 nb_filters: int = 32,
                 kernel_size: int = 41,
                 num_kernels: int = 3,
                 use_bottleneck: bool = True,
                 bottleneck_size: int = 32,
                 emulate_keras: bool = True,
                 force_same_padding: bool = True):
        super().__init__()
        self.use_bottleneck = use_bottleneck and in_channels > 1
        self.nb_filters = nb_filters
        self.num_kernels = num_kernels
        self.emulate_keras = emulate_keras
        self.force_same_padding = force_same_padding

        # Keras: 내부에서 kernel_size - 1 사용
        effective_base = kernel_size - 1 if emulate_keras else kernel_size
        # 커널 목록 (내림)
        kernel_sizes: List[int] = [max(1, effective_base // (2 ** i)) for i in range(num_kernels)]
        # (Keras 예: 41 → 내부 40 → [40, 20, 10])

        # Bottleneck
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)
            conv_in = bottleneck_size
        else:
            self.bottleneck = None
            conv_in = in_channels

        self.kernel_sizes = kernel_sizes
        self.conv_layers = nn.ModuleList()
        self.need_postpad = False  # 수동 same 패딩 필요 여부

        for ks in kernel_sizes:
            if force_same_padding:
                # PyTorch 1.10+ same padding 지원
                try:
                    conv = nn.Conv1d(conv_in, nb_filters, kernel_size=ks,
                                     padding='same', bias=False)
                except TypeError:
                    # fallback: manual (even kernel 시 한 칸 부족 → post pad)
                    pad = ks // 2 if ks % 2 == 1 else ks // 2 - 1
                    conv = nn.Conv1d(conv_in, nb_filters, kernel_size=ks,
                                     padding=pad, bias=False)
                    if ks % 2 == 0:
                        self.need_postpad = True
            else:
                # 개선 옵션을 끄고 완전 동일 재현이 목적이므로 even kernel 그대로
                pad = ks // 2 if ks % 2 == 1 else ks // 2 - 1
                conv = nn.Conv1d(conv_in, nb_filters, kernel_size=ks,
                                 padding=pad, bias=False)
                if ks % 2 == 0:
                    self.need_postpad = True
            self.conv_layers.append(conv)

        # MaxPool branch
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)
        )

        total_out = (num_kernels + 1) * nb_filters
        self.bn = nn.BatchNorm1d(total_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        return: (B, (num_kernels+1)*nb_filters, T)
        """
        if self.bottleneck is not None:
            x_in = self.bottleneck(x)
        else:
            x_in = x

        outs = []
        for conv, ks in zip(self.conv_layers, self.kernel_sizes):
            o = conv(x_in)
            # manual even kernel padding 보정
            if self.need_postpad and ks % 2 == 0 and o.shape[-1] == x.shape[-1] - 1:
                o = F.pad(o, (0, 1))
            outs.append(o)

        pool_out = self.pool_branch(x)
        outs.append(pool_out)

        # 혹시 fallback 조건에서 길이 불일치 발생 시 정렬
        lengths = [t.shape[-1] for t in outs]
        if len(set(lengths)) > 1:
            max_len = max(lengths)
            outs = [F.pad(t, (0, max_len - t.shape[-1])) for t in outs]

        x_cat = torch.cat(outs, dim=1)
        x_cat = self.bn(x_cat)
        return F.relu(x_cat, inplace=True)


class ResidualBlock(nn.Module):
    """
    Keras 구현처럼 '항상' 1x1 Conv + BN 로 projection 후 Add + ReLU.
    (채널 같아도 동일하게 적용)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x_prev: torch.Tensor, x_cur: torch.Tensor) -> torch.Tensor:
        shortcut = self.proj(x_prev)
        return F.relu(shortcut + x_cur, inplace=True)