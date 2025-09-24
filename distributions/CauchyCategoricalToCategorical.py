import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import torchrl

def cauchy_kernel_1d(size, gamma = 3, device=None, dtype=torch.float32):
    half = size // 2
    d = torch.arange(-half, half+1, device=device, dtype=dtype)
    k = 1.0 / (1.0 + (d / gamma) ** 2)
    k = k / (k.sum() + 1e-8)           # normalize
    return k.view(1, 1, -1)            # (1,1,K)

@torch.no_grad()
def cauchy_smooth_with_mask(probs: torch.Tensor,
                            valid: torch.Tensor,
                            size: int = 21,
                            gamma: float = 1.5) -> torch.Tensor:
    B, T = probs.shape
    device = probs.device
    k = cauchy_kernel_1d(size, gamma, device=device, dtype=probs.dtype)  # (1,1,K)

    x_num = (probs * valid.float()).unsqueeze(1)      # (B,1,T)
    y_num = F.conv1d(x_num, k, padding=size//2)       # (B,1,T)

    x_den = valid.float().unsqueeze(1)                # (B,1,T)
    y_den = F.conv1d(x_den, k, padding=size//2)       # (B,1,T)

    y = (y_num / (y_den + 1e-8)).squeeze(1)          # (B,T)
    y = y * valid.float()
    y = y / (y.sum(dim=1, keepdim=True) + 1e-8)
    return y

class CauchyCategoricalToCategorical(D.Distribution):
    def __init__(self, start_logits, end_logits, seq_len=None):

        # self.cauchy = cauchy_kernel(seq_len).to(start_logits.device)
        # start_logits = smooth_logits_with_cauchy(start_logits, self.cauchy, eta=0.3)

        self.cat_start = D.Categorical(logits=start_logits)
        # self.end_logits = smooth_logits_with_cauchy(end_logits, self.cauchy, eta=0.3)
        self.end_logits = end_logits
        self.seq_len = seq_len

    # def get_end_dist(self, start):
    #     arange = torch.arange(self.seq_len, dtype=start.dtype, device=start.device).unsqueeze(0)

    #     mask = start <= arange
    #     end_probs = self.end_logits.masked_fill(~mask, float('-inf')).softmax(-1)
        
    #     end_logits = mask * self.end_logits + torch.logical_not(mask) * -1e10
    #     return D.Categorical(logits=end_logits)

    def get_end_dist(self, start):
        arange = torch.arange(self.seq_len, dtype=start.dtype, device=start.device).unsqueeze(0)
        mask = start <= arange

        logits_masked = self.end_logits.masked_fill(~mask, float('-inf'))  # (B,T)
        probs = torch.softmax(logits_masked, dim=-1)                         # (B,T)
        probs_smooth = cauchy_smooth_with_mask(
            probs, mask, size=self.seq_len//10, gamma=3.
        )
        end_dist = D.Categorical(probs=probs_smooth.clamp_min(1e-12))
        return end_dist

    def rsample(self, sample_shape):
        start = self.cat_start.sample().unsqueeze(-1)
        self.cat_end = self.get_end_dist(start)

        end = self.cat_end.sample().unsqueeze(-1)
        return torch.concat([start, end], dim=1).int()

    def log_prob(self, value):
        start = value[:, 0]
        end = value[:, 1]

        self.cat_end = self.get_end_dist(start.unsqueeze(-1))

        start_log_probs = self.cat_start.log_prob(start)
        end_log_probs = self.cat_end.log_prob(end)

        return start_log_probs + end_log_probs
    
    @property
    def deterministic_sample(self):
        start = self.cat_start.mode.unsqueeze(-1)
        self.cat_end = self.get_end_dist(start)

        end = self.cat_end.mode.unsqueeze(-1)
        return torch.concat([start, end], dim=1).int()
    
    def entropy(self):
        return self.cat_start.entropy() + D.Categorical(logits=self.end_logits).entropy()

    @torch.no_grad()
    def calculate_marginal_mask(self):
        start_probs = self.cat_start.probs
        exp_scores = self.end_logits.exp()
        cum_rev = torch.cumsum(exp_scores.flip(-1), dim=-1).flip(-1)  # (B, T)
        denom   = cum_rev + 1e-7

        ratio = cum_rev[:, :, None] / denom[:, None, :]
        tri_mask = torch.tril(torch.ones(
            self.seq_len, self.seq_len, device=ratio.device, dtype=ratio.dtype)
        )                                                            # (T, T)
        ratio *= tri_mask                                            # broadcast

        # ----- 4) Σ_s  P(S=s) · P(E ≥ t | S=s)  -----------------------
        mask_probs = torch.sum(ratio * start_probs[:, None, :], dim=-1)  # (B, T)
        return mask_probs.clamp_(0.0, 1.0)   # 수치 오차 방지


    #
    @torch.no_grad()
    def start_param(self):
        return f"m: {self.cat_start.probs.mean().item():.2f}, s:{self.cat_start.probs.std().item():.2f}"
    @torch.no_grad()
    def end_param(self):
        return f"m: {self.end_logits.softmax(-1).mean().item():.2f}, s:{self.end_logits.softmax(-1).std().item():.2f}"
    
    def start_marginal(self):
        return self.cat_start.probs.cpu().squeeze().numpy()

    def end_marginal(self):
        return self.end_logits.softmax(-1).cpu().squeeze().numpy()