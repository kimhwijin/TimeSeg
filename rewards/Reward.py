import sys
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)


from functools import partial
from collections import OrderedDict
import torch
import torch.nn.functional as F

from utils import masking


def get_reward_fn(reward_types, weights, mask_fn, predictor):
    reward_fn_dict = {}
    print(reward_types)
    for reward_type in reward_types:
        if reward_type == 'exp_ce':
            reward_fn_dict["CrossEntropy"] = partial(
                exp_minus_cross_entropy_reward, 
                mask_fn      = mask_fn, 
                predictor    = predictor
            )
        if reward_type == 'ce':
            reward_fn_dict["CrossEntropy"] = partial(
                minus_cross_entropy_reward, 
                mask_fn      = mask_fn, 
                predictor    = predictor
            )
        if reward_type == 'length':
            reward_fn_dict["Length"] = length_reward
        if reward_type == 'overlap':
            reward_fn_dict["Overlap"] = overlap_reward

    reward_fn = partial(compose_reward, 
        reward_fns  = reward_fn_dict, 
        weights     = weights
    )
    return reward_fn


@torch.no_grad()
def compose_reward(reward_fns, weights, **kwargs):
    reward_dict = {
        name: reward_fn(**kwargs) for name, reward_fn in reward_fns.items()
    }
    rewards = torch.stack(list(reward_dict.values()), dim=1)
    if weights is not None:
        weights = torch.tensor([weights], dtype=rewards.dtype, device=rewards.device)
        # B,
        reward = (rewards * weights).sum(-1, keepdim=True)
    elif weights is None:
        reward = rewards
    return reward, reward_dict

@torch.no_grad()
def overlap_reward(x, y, curr_mask, new_mask):
    seq_len = x.shape[1]
    and_mask = torch.logical_and(curr_mask, new_mask)
    or_mask = torch.logical_or(curr_mask, new_mask)
    
    overlap = and_mask.sum([1, 2], dtype=x.dtype) / or_mask.sum([1, 2], dtype=x.dtype)

    reward = (1 - overlap) #/ or_mask.sum([1, 2], dtype=x.dtype)
    return reward.detach()


@torch.no_grad()
def length_reward(x, y, curr_mask, new_mask):
    seq_len = x.shape[1]
    length = new_mask.sum([1, 2], dtype=x.dtype) / seq_len
    reward = -length

    return reward.detach()

@torch.no_grad()
def next_length_reward(x, y, curr_mask, new_mask):
    seq_len = x.shape[1]
    next_mask = torch.logical_or(curr_mask, new_mask)
    length = next_mask.sum([1, 2], dtype=x.dtype) / seq_len
    reward = -length

    return reward.detach()


@torch.no_grad()
def minus_cross_entropy_reward(x, y, curr_mask, new_mask, predictor, mask_fn):
    curr_x = mask_fn(x, curr_mask)
    next_x = mask_fn(x, torch.logical_or(curr_mask, new_mask))

    curr_logits = predictor(curr_x)
    next_logits = predictor(next_x)

    CE = F.cross_entropy(next_logits, y, reduction='none') - F.cross_entropy(curr_logits, y, reduction='none')

    reward = -CE
    return reward.detach()

@torch.no_grad()
def exp_minus_cross_entropy_reward(x, y, curr_mask, new_mask, predictor, mask_fn):
    curr_x = mask_fn(x, curr_mask)
    next_x = mask_fn(x, torch.logical_or(curr_mask, new_mask))
    
    curr_logits = predictor(curr_x)
    next_logits = predictor(next_x)
    
    if (curr_mask.sum([1, 2]) == 0).all():
        dif_ce = (-F.cross_entropy(next_logits, y, reduction='none')).exp()
    else:        
        dif_ce = (-F.cross_entropy(next_logits, y, reduction='none')).exp() - \
            (-F.cross_entropy(curr_logits, y, reduction='none')).exp()
    
    reward = dif_ce.detach()
    return reward.detach()

@torch.no_grad()
def inverse_cross_entropy_reward(x, y, curr_mask, new_mask, predictor, mask_fn):
    curr_x = mask_fn(x, curr_mask)
    next_x = mask_fn(x, torch.logical_or(curr_mask, new_mask))
    
    curr_logits = predictor(curr_x)
    next_logits = predictor(next_x)
    
    if curr_mask.sum() > 0:
        CE = F.cross_entropy(next_logits, y, reduction='none')  - F.cross_entropy(curr_logits, y, reduction='none')
    else:
        CE = F.cross_entropy(next_logits, y, reduction='none')
    reward = 1 / (CE + 1e-10)
    return reward.detach()


    