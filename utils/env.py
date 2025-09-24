import torch
from tensordict import TensorDict

def step(
    tensordict,
    policy_module,
    reward_fn,
    terminate_fn,
    mode=False,
):  

    if not mode:
        policy_module(tensordict)
    else:
        dist = policy_module.get_dist(tensordict)
        action = dist.deterministic_sample
        tensordict['action'] = action
        
    x = tensordict["x"]
    y = tensordict["y"]
    action = tensordict["action"]
    curr_mask = tensordict["curr_mask"]

    arange = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    start, end = action[:, 0].unsqueeze(-1), action[:, 1].unsqueeze(-1)

    new_mask = torch.logical_and(start <= arange, arange <= end).unsqueeze(-1)
    next_mask = torch.logical_or(curr_mask, new_mask)
    
        
    reward, reward_dict     = reward_fn(x=x, y=y, curr_mask=curr_mask, new_mask=new_mask) # no grad reward func
    is_done                 = terminate_fn(x, y, curr_mask, new_mask)
    
    
    next_tensordict = TensorDict(
        {
            'x': x,
            'y': y,
            'new_mask': new_mask,
            'curr_mask': next_mask,
            'reward': reward,
            'reward_dict': reward_dict,
            'done': is_done,
        },
        batch_size=tensordict.shape,
        device=tensordict.device
    )
    tensordict['next'] = next_tensordict
    return tensordict
    


def bern_step(
    tensordict,
    policy_module,
    reward_fn,
    terminate_fn,
    mode=False,
):  
        # Action
    if not mode:
        policy_module(tensordict)
    else:
        dist = policy_module.get_dist(tensordict)
        action = dist.deterministic_sample
        tensordict['action'] = action
        
    x = tensordict["x"]
    y = tensordict["y"]
    action = tensordict["action"]
    curr_mask = tensordict["curr_mask"]

    new_mask = action
    next_mask = torch.logical_or(curr_mask, new_mask)

    reward, reward_dict     = reward_fn(x=x, y=y, curr_mask=curr_mask, new_mask=new_mask) # no grad reward func
    is_done                 = terminate_fn(x, y, curr_mask, new_mask)
    
    next_tensordict = TensorDict(
        {
            'x': x,
            'y': y,
            'new_mask': new_mask,
            'curr_mask': next_mask,
            'reward': reward,
            'reward_dict': reward_dict,
            'done': is_done,
        },
        batch_size=tensordict.shape,
        device=tensordict.device
    )
    tensordict['next'] = next_tensordict
    return tensordict
    