import sys
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)

import torch
import torch.nn.functional as F
from functools import partial

def get_terminate_fn(terminate_type, max_segment, mask_fn, predictor, threshold=0.3):
    if terminate_type == 'rel_ce_diff':
        terminate_fn = partial(rel_cross_entropy_difference_terminate, 
            mask_fn     = mask_fn,
            predictor   = predictor,
            max_step    = max_segment,
            threshold   = threshold
        )
    return terminate_fn
    
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
