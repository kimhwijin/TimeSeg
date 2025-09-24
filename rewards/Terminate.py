import sys
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)

from collections import OrderedDict
import torch
import torch.nn.functional as F
from functools import partial
from utils import masking

def get_terminate_fn(terminate_type, max_segment, mask_fn, predictor, threshold=0.05):
    if terminate_type == 'max_segment':
        terminate_fn = partial(max_length_terminate, 
            max_step    = max_segment
        )
    if terminate_type == 'ce_diff':
        terminate_fn = partial(cross_entropy_difference_terminate, 
            mask_fn     = mask_fn,
            predictor   = predictor,
            max_step    = max_segment,
            threshold   = threshold
        )
    if terminate_type == 'rel_ce_diff':
        terminate_fn = partial(rel_cross_entropy_difference_terminate, 
            mask_fn     = mask_fn,
            predictor   = predictor,
            max_step    = max_segment,
            threshold   = threshold
        )
    if terminate_type == 'mix_ce_diff':
        terminate_fn = partial(mix_cross_entropy_difference_terminate, 
            mask_fn         = mask_fn,
            predictor       = predictor,
            max_step        = max_segment,
            abs_threshold   = 0.05, 
            rel_threshold   = 0.5
        )
        
    if terminate_type == 'kld_diff':
        terminate_fn = partial(kl_div_terminate, 
            mask_fn     = mask_fn,
            predictor   = predictor,
            max_step    = max_segment
        )
    return terminate_fn
    

@torch.no_grad()
def max_length_terminate(x, y, curr_mask, new_mask, step, max_step):
    B = x.size(0)
    device = x.device
    if step < max_step:
        return torch.zeros(B, 1,  dtype=bool, device=device)
    else:
        return torch.ones(B, 1, dtype=bool, device=device)

@torch.no_grad()
def cross_entropy_difference_terminate(x, y, curr_mask, new_mask, predictor, mask_fn, step, max_step, threshold=0.05):
    B = x.size(0)
    device = x.device

    curr_x = mask_fn(x, curr_mask)
    next_x = mask_fn(x, torch.logical_or(curr_mask, new_mask))

    curr_logits = predictor(curr_x)
    next_logits = predictor(next_x)
    
    ce_next = F.cross_entropy(next_logits, y, reduction='none')
    ce_curr = F.cross_entropy(curr_logits, y, reduction='none')
    # rel_ac = (ce_curr - ce_next)/ (ce_curr + 1e-6)
    aq_ce = ce_curr - ce_next
    # C = next_logits.size(-1)
    # norm = torch.log(torch.tensor(float(C), device=device))
    # aq_ce /= norm
    
    # Terminate
    if step == max_step:
        is_done = torch.ones(B, 1, dtype=bool, device=device)
    else:
        is_done = torch.where(aq_ce < threshold, torch.tensor(True), torch.tensor(False)).to(bool).unsqueeze(-1)
    return is_done

@torch.no_grad()
def mix_cross_entropy_difference_terminate(x, y, curr_mask, new_mask, predictor, mask_fn, step, max_step, abs_threshold=0.05, rel_threshold=0.5):
    B = x.size(0)
    device = x.device

    curr_x = mask_fn(x, curr_mask)
    next_x = mask_fn(x, torch.logical_or(curr_mask, new_mask))

    curr_logits = predictor(curr_x)
    next_logits = predictor(next_x)
    
    ce_next = F.cross_entropy(next_logits, y, reduction='none')
    ce_curr = F.cross_entropy(curr_logits, y, reduction='none')
    rel_ac = (ce_curr - ce_next)/ (ce_curr + 1e-6)

    aq_ce = ce_curr - ce_next

    if step == max_step:
        is_done = torch.ones(B, 1, dtype=bool, device=device)
    else:
        is_done = torch.where((rel_ac < rel_threshold) or (aq_ce < abs_threshold), torch.tensor(True), torch.tensor(False)).to(bool).unsqueeze(-1)
    return is_done

@torch.no_grad()
def rel_cross_entropy_difference_terminate(x, y, curr_mask, new_mask, predictor, mask_fn, step, max_step, threshold=0.5):
    B = x.size(0)
    device = x.device

    curr_x = mask_fn(x, curr_mask)
    next_x = mask_fn(x, torch.logical_or(curr_mask, new_mask))

    curr_logits = predictor(curr_x)
    next_logits = predictor(next_x)
    
    ce_next = F.cross_entropy(next_logits, y, reduction='none')
    ce_curr = F.cross_entropy(curr_logits, y, reduction='none')
    if (curr_mask.sum([1, 2]) == 0).all():
        rel_ac = torch.ones_like(ce_curr) * (threshold + 1e-6)
    else:
        rel_ac = (ce_curr - ce_next)/ (ce_curr + 1e-6)
    

    if step == max_step:
        is_done = torch.ones(B, 1, dtype=bool, device=device)
    else:
        is_done = torch.where(rel_ac < threshold, torch.tensor(True), torch.tensor(False)).to(bool).unsqueeze(-1)
    return is_done


@torch.no_grad()
def cross_entropy_difference_with_original_terminate(x, y, curr_mask, new_mask, predictor, mask_fn, step, max_step):
    B = x.size(0)
    device = x.device

    curr_x = mask_fn(x, curr_mask)
    next_x = mask_fn(x, torch.logical_or(curr_mask, new_mask))

    # curr_logits = predictor(curr_x)
    next_logits    = predictor(next_x)
    orig_logits    = predictor(x)

    next_ce = F.cross_entropy(next_logits, y, reduction='none')
    orig_ce = F.cross_entropy(orig_logits, y, reduction='none')

    ce_diff = next_ce - orig_ce

    # Terminate
    if step == max_step:
        is_done = torch.ones(B, 1, dtype=bool, device=device)
    else:
        is_done = torch.where(ce_diff < 0.01, torch.tensor(True), torch.tensor(False)).to(bool).unsqueeze(-1)
    return is_done


@torch.no_grad()
def kl_div_terminate(x, y, curr_mask, new_mask, predictor, mask_fn, step, max_step):
    B = x.size(0)
    device = x.device

    curr_x = mask_fn(x, curr_mask)
    next_x = mask_fn(x, torch.logical_or(curr_mask, new_mask))

    curr_log_probs = predictor(curr_x).log_softmax(-1)
    next_log_probs = predictor(next_x).log_softmax(-1)
    
    kl_div = (curr_log_probs.exp() * (curr_log_probs - next_log_probs)).sum(-1)

    # Terminate
    if step == max_step:
        is_done = torch.ones(B, 1, dtype=bool, device=device)
    else:
        is_done = torch.where(kl_div < 0.005, torch.tensor(True), torch.tensor(False)).to(bool).unsqueeze(-1)
    return is_done