import sys
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)


import torch
import torch.nn.functional as F
import torchrl
import torch.distributions as D

def nb_transform_fn(logits, nb_transform, seq_len):
    if nb_transform == 'r_p':
        tc          = torchrl.modules.utils.biased_softplus(1.)(logits[:, 0]).clamp(min=1e-2)
        probs       = torch.ones_like(logits[:, 0]) * 0.5
        return D.NegativeBinomial(total_count=tc, probs=probs)
    
    if nb_transform == 'r_p_full':
        tc      = torchrl.modules.utils.biased_softplus(1.)(logits[:, 0]).clamp(min=1e-2)
        p       = torch.sigmoid(logits[:, 1])
        logits_probs = torch.log(p) - torch.log1p(-p)
        logits_probs = logits_probs.clamp(-15, 15)
        # logits_probs    = -1 * torchrl.modules.utils.biased_softplus(1.)(logits[:, 1])

        return D.NegativeBinomial(total_count=tc, logits=logits_probs)
    
    elif nb_transform == 'mu_alpha':

        mu          = F.softplus(1+logits[:, 0]) + 1e-4
        alpha       = F.softplus(1+logits[:, 1]) + 1e-4
        probs       = torch.ones_like(logits[:, 0]) * 0.5
        return D.NegativeBinomial(total_count=alpha, probs=probs)
    
    elif nb_transform == 'mu_alpha_full':
        mu              = F.softplus(1+logits[:, 0]) + 1e-4
        alpha           = F.softplus(1+logits[:, 1]) + 1e-4
        logits_probs    = (alpha.log() - mu.log()).clamp(-15, 15)
        return D.NegativeBinomial(total_count=alpha, logits=logits_probs)


    elif nb_transform == 'r_temp':
        tc          = torchrl.modules.utils.biased_softplus(1.)(logits[:, 0]).clamp_min(1e-4)
        probs       = F.sigmoid(logits[:, 1] / 3.0)
    return tc, probs



def safe_nb(logit_tc_pair):
    tc_raw, logit_raw = logit_tc_pair
    # total_count: remove NaN/Inf, ensure >0
    tc = torch.nan_to_num(tc_raw, nan=0.0, posinf=20.0, neginf=0.0)
    tc = F.softplus(tc) + 1e-2
    # tc = torch.nan_to_num(tc, nan=1e-4, posinf=10., neginf=1e-4)

    # logit = torch.nan_to_num(raw_logit, nan=0., posinf=10., neginf=-10.)
    # logits: clamp & NaN fix
    logit = torch.nan_to_num(logit_raw, nan=0.0, posinf=10.0, neginf=-10.0)
    return tc, logit


class EarlyStopping:

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        assert mode in ("min", "max"), "`mode` must be 'min' or 'max'"
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_value = float("inf") if mode == "min" else -float("inf")
        self.num_bad    = 0
        self.stop       = False

    def __call__(self, metric_value: float) -> bool:
        improved = (
            metric_value < self.best_value - self.min_delta  # "min" 모드
            if self.mode == "min"
            else metric_value > self.best_value + self.min_delta  # "max" 모드
        )
        if improved:
            self.best_value = metric_value
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience and self.patience > 0:
                self.stop = True
        return improved

    # 편의용 속성 -------------------------------------------------------
    @property
    def best(self):
        return self.best_value

    @property
    def bad_streak(self):
        return self.num_bad