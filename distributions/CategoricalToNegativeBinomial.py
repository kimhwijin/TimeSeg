import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import torchrl
import scipy.special as S
from utils import nb_transform_fn

class CategoricalToNegativeBinomial(D.Distribution):
    def __init__(self, start_logits, end_logits, seq_len=None, nb_transform='r_p'):
        self.cat = D.Categorical(logits=start_logits)

        self.nb = nb_transform_fn(end_logits, nb_transform, seq_len)
        # self.nb = D.NegativeBinomial(
        #     total_count=tc,
        #     probs=probs
        # )
        self.seq_len = seq_len

    def rsample(self, sample_shape):
        
        start = self.cat.sample(sample_shape).unsqueeze(-1)
        delta = self.nb.sample(sample_shape).unsqueeze(-1)

        end = torch.clamp(start + delta, 0., self.seq_len-1)

        return torch.concat([start, end, delta], dim=1).int()

    def log_prob(self, value):
        start = value[:, 0]
        delta = value[:, 2]
        
        start_log_probs = self.cat.log_prob(start)
        end_log_probs = self.nb.log_prob(delta)
        end_log_probs = torch.where(end_log_probs <= -30, torch.ones_like(end_log_probs), end_log_probs)
        
        return start_log_probs + end_log_probs
    
    @property
    def deterministic_sample(self):
        start = self.cat.mode.unsqueeze(-1)
        delta = self.nb.mode.unsqueeze(-1)
        end = torch.clamp(start + delta, 0., self.seq_len-1)
        
        return torch.concat([start, end, delta], dim=1).int()
    
    def entropy(self):
        def nb_entropy(r, p):
            """Entropy of NB(r, p)  with r>0, p∈(0,1)."""
            t1 = -r * torch.log(p)
            t2 = -(1 - p) / p * torch.log1p(-p)     # log(1-p)
            t3 = (1 - p) / p * (torch.digamma(r + 1) - torch.digamma(torch.tensor(1.)))
            return t1 + t2 + t3
        return self.cat.entropy() + nb_entropy(self.nb.total_count, self.nb.probs)

    @torch.no_grad()
    def calculate_marginal_mask(self):
        
        start_p = self.cat.probs.detach().cpu()                      # (B, T)
        B, T    = start_p.shape
        device  = start_p.device
        dtype   = start_p.dtype
        
        r = self.nb.total_count.to(dtype).detach().cpu()             # (B,) or scalar
        p = self.nb.probs.to(dtype).detach().cpu()                   # (B,)

        # --- NB CDF: F(k; r, p) = I_p(r, k+1) ------------------------
        k_vals  = torch.arange(T, dtype=dtype)   # 0…T-1
        # shape → (B, T) by unsqueeze & broadcast
        F_k     = S.betainc(r.unsqueeze(1), k_vals.unsqueeze(0) + 1, p.unsqueeze(1))
        # survival for offset d = P(Δ ≥ d) = 1 - F(d-1),  with F(-1)=0
        surv = torch.ones_like(F_k)                   # (B, T)  d=0 → 1
        surv[:, 1:] = 1.0 - F_k[:, :-1]               # d≥1

        # --- convolution‐style accumulation -------------------------
        # mask_probs[b, t] = Σ_{s≤t} start_p[b, s] * surv[b, t-s]
        mask_probs = torch.zeros_like(start_p)

        for s in range(T):                            # loop ≤100 → 빠름
            mask_probs[:, s:] += start_p[:, s:s+1] * surv[:, :T-s]
        return mask_probs.clamp_(0.0, 1.0)

    @torch.no_grad()
    def start_param(self):
        return f"m: {self.cat.probs.mean().item():.2f}, s:{self.cat.probs.std().item():.2f}"
    @torch.no_grad()
    def end_param(self):
        return f"r: {self.nb.total_count.item():.2f}, p:{self.nb.probs.item():.2f}"

    def start_marginal(self):
        return self.cat.probs.cpu().squeeze().numpy()

    def end_marginal(self):
        return self.nb.log_prob(torch.arange(self.seq_len, device=self.nb.total_count.device)).exp().cpu().squeeze().numpy()
    