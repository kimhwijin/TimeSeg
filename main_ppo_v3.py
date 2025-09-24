import warnings
warnings.filterwarnings('ignore')

from functools import partial
from itertools import cycle
from collections import defaultdict, OrderedDict

import os
import psutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, jaccard_score
from sklearn.preprocessing import label_binarize
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
# 
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl.data.replay_buffers import (
    TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
)

# Custom
from datasets import get_datamodule

from distributions import get_dist
from models import Policy, Value, Predictor
from utils import env, EarlyStopping
from utils.masking import MaskingFunction
from rewards import Reward, Terminate
from models.classifier import ClassifierNet, get_classifier
from tint.metrics.white_box import (
    aup,
    aur,
    information,
    entropy,
    roc_auc,
    auprc,
)

from torchmetrics import Accuracy, Precision, Recall, AUROC, F1Score, AveragePrecision

import gc
torch.set_num_threads(8)


def main(
    seg_dist,
    nb_transform,
    reward_types,
    terminate_type,
    weights,
    backbone,
    dataset,
    split,
    entropy_coef,
    mask_type,
    epochs,
    batch_size,
    rollout_len,
    ppo_epochs,
    max_segment,
    early_stop,
    seed,
    threshold,
    device,
):

    datamodule, num_features, seq_len, num_class = get_datamodule(data=dataset, fold=split, seed=seed, batch_size=batch_size)
    datamodule.setup()
    train_set = datamodule.train
    valid_set = datamodule.val
    test_set = datamodule.test

    rollout_len = rollout_len
    total_len   = rollout_len*4
    batch_size  = batch_size
    d_in        = 1
    d_model     = 128
    d_out       = num_class
    history     = defaultdict(list)

    train_loader        = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader    = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader         = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    cycle_train_loader  = cycle(train_loader)

    distribution_class, distribution_kwargs, (d_start, d_end) = get_dist(seg_dist, seq_len, nb_transform)

    policy_net = Policy.PolicyNetwork(
        d_in        = d_in+1,
        d_model     = d_model,
        d_start     = d_start,
        d_end       = d_end,
        seq_len     = seq_len,
        backbone    = backbone[1]
    )
    policy_module = TensorDictModule(
        policy_net, 
        in_keys     = ['x', 'curr_mask'], 
        out_keys    = ['start_logits', 'end_logits']
    )
    policy_module = ProbabilisticActor(
        module                      = policy_module,
        in_keys                     = ["start_logits", "end_logits"],
        distribution_class          = distribution_class,
        distribution_kwargs         = distribution_kwargs,
        return_log_prob             = True,
        default_interaction_type    = InteractionType.RANDOM,
    )

    value_net = Value.ValueNetwork(
        d_in      =  d_in+1,
        d_model   = d_model,
        d_out     = 1, 
        seq_len   = seq_len,
        backbone  = backbone[1]
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["x", 'curr_mask'],
    )

    advantage_module = GAE(
        value_network=None,
        gamma = 0.99, lmbda=1., average_gae=True, time_dim=1
    )

    classifier = get_classifier(num_features=d_in, num_classes=num_class, max_len=seq_len, model_type=backbone[0], data=dataset)
    classifier.load_state_dict(torch.load(f"./model_ckpt/{dataset}/{backbone[0]}_classifier_{split}_{seed}.ckpt", weights_only=False)['state_dict'])

    policy_module = policy_module.to(device)
    policy_module.eval()
    value_module = value_module.to(device)
    value_module.eval()
    classifier = classifier.to(device)
    classifier.eval()

    loss_module = ClipPPOLoss(
        actor            = policy_module,
        critic           = value_module,
        clip_epsilon     = 0.2,
        entropy_bonus    = True,
        entropy_coef     = entropy_coef,
        critic_coef      = 1.0,
        loss_critic_type = "smooth_l1",
    )

    mask_fn = MaskingFunction(mask_type, train_set)

    policy_optim = torch.optim.AdamW(policy_module.parameters(), lr=1e-4, weight_decay = 0.0001)
    value_optim = torch.optim.AdamW(value_module.parameters(), lr=1e-4)

    warmup_epochs = 100
    policy_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(policy_optim, start_factor=1/3, end_factor=1., total_iters=warmup_epochs)
    policy_cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(policy_optim, epochs, eta_min=5e-5)
    policy_scheduler = torch.optim.lr_scheduler.SequentialLR(policy_optim, schedulers=[policy_warmup_scheduler, policy_cosine_scheduler], milestones=[warmup_epochs])

    value_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(value_optim, start_factor=1/3, end_factor=1., total_iters=warmup_epochs)
    value_cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optim, epochs, eta_min=2e-5)
    value_scheduler = torch.optim.lr_scheduler.SequentialLR(value_optim, schedulers=[value_warmup_scheduler, value_cosine_scheduler], milestones=[warmup_epochs])


    storage = LazyTensorStorage(total_len, device='cpu')
    sampler = SamplerWithoutReplacement()
    replay_buffer = TensorDictReplayBuffer(
        storage     = storage, 
        sampler     = sampler, 
        batch_size  = batch_size,
    )
    
    reward_fn       = Reward.get_reward_fn(reward_types, weights, mask_fn, classifier)
    terminate_fn    = Terminate.get_terminate_fn(terminate_type, max_segment, mask_fn, classifier, threshold)
    
    early_stop = EarlyStopping(patience = early_stop)
    for epoch in range(epochs):

        gc.collect()
        torch.cuda.empty_cache()
        
        msg = f"Epoch : {epoch}"; print(msg)
        collect_samples(
            epoch               = epoch,
            policy_module       = policy_module,
            value_module        = value_module,
            advantage_module    = advantage_module,
            classifier          = classifier,
            replay_buffer       = replay_buffer,
            rollout_len         = rollout_len,
            reward_fn           = reward_fn,
            terminate_fn        = terminate_fn,
            max_segment         = max_segment,
            cycle_loader        = cycle_train_loader,
            device              = device
        )

        ppo_update(
            epoch               = epoch,
            policy_module       = policy_module,
            value_module        = value_module,
            loss_module         = loss_module,
            ppo_epochs          = ppo_epochs,
            replay_buffer       = replay_buffer,
            rollout_len         = rollout_len,
            policy_optim        = policy_optim,
            policy_scheduler    = policy_scheduler,
            value_optim         = value_optim,
            value_scheduler     = value_scheduler,
            history             = history,
            device              = device,
        )
        valid_step(
            epoch               = epoch,
            loader              = valid_loader,
            plot_set            = valid_set,
            num_classes         = num_class,
            policy_module       = policy_module,
            predictor           = classifier,
            reward_fn           = reward_fn,
            terminate_fn        = terminate_fn,
            max_segment         = max_segment,
            early_stop          = early_stop,
            mask_fn             = mask_fn,
            history             = history,
            seg_dist            = seg_dist,
            seq_len             = seq_len,
            classifier          = classifier,
            device              = device,
        )
        if early_stop.stop: break
        
    test_sample_plot(
        epoch               = epoch,
        policy_module       = policy_module,
        predictor           = classifier,
        plot_set            = test_set,
        num_classes         = num_class,
        reward_fn           = reward_fn,
        terminate_fn        = terminate_fn,
        mask_fn             = mask_fn,
        seg_dist            = seg_dist,
        seq_len             = seq_len,
        max_segment         = max_segment,
        device              = device,
    )

    test_step(
        dataset         = dataset,
        split           = split,
        seed            = seed,
        seg_dist        = seg_dist,
        nb_transform    = nb_transform,
        backbone        = backbone,
        entropy_coef    = entropy_coef,
        mask_type       = mask_type,
        batch_size      = batch_size,
        reward_types    = reward_types,
        weights         = weights,
        max_segment     = max_segment,
        terminate_type  = terminate_type,
        device          = device,
    )

@torch.no_grad()
def test_step(
    dataset,
    split,
    seed,
    seg_dist,
    nb_transform,
    backbone,
    entropy_coef,
    mask_type,
    batch_size,
    reward_types,
    weights,
    max_segment,
    terminate_type,
    device,
):

    datamodule, num_features, seq_len, num_class = get_datamodule(data=dataset, fold=split, seed=seed, batch_size=batch_size)
    datamodule.setup()
    train_set       = datamodule.train
    valid_set       = datamodule.val
    test_set        = datamodule.test
    # true_saliency   = datamodule.true_saliency()

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    d_in        = 1
    d_model     = 128
    d_out       = num_class

    distribution_class, distribution_kwargs, (d_start, d_end) = get_dist(seg_dist, seq_len, nb_transform)

    policy_net = Policy.PolicyNetwork(
        d_in        = d_in+1,
        d_model     = d_model,
        d_start     = d_start,
        d_end       = d_end,
        seq_len     = seq_len,
        backbone    = backbone[1]
    )
    policy_module = TensorDictModule(
        policy_net, 
        in_keys     = ['x', 'curr_mask'], 
        out_keys    = ['start_logits', 'end_logits']
    )
    policy_module = ProbabilisticActor(
        module                      = policy_module,
        in_keys                     = ["start_logits", "end_logits"],
        distribution_class          = distribution_class,
        distribution_kwargs         = distribution_kwargs,
        return_log_prob             = True,
        default_interaction_type    = InteractionType.RANDOM,
    )

    policy_module.load_state_dict(torch.load("./model_ckpt/policy.pth"))
    classifier = get_classifier(num_features=d_in, num_classes=num_class, max_len=seq_len, model_type=backbone[0], data=dataset)
    classifier.load_state_dict(torch.load(f"./model_ckpt/{dataset}/{backbone[0]}_classifier_{split}_{seed}.ckpt", weights_only=False)['state_dict'])

    policy_module = policy_module.to(device);policy_module.eval()
    classifier = classifier.to(device);classifier.eval()

    mask_fn         = MaskingFunction(mask_type, train_set)
    reward_fn       = Reward.get_reward_fn(reward_types, weights, mask_fn, classifier)
    terminate_fn    = Terminate.get_terminate_fn(terminate_type, max_segment, mask_fn, classifier)

    step_mask = []
    step_reward = []
    mask = []
    next_mask = []
    for batch in tqdm(test_loader):
        x = batch['x'].to(device)
        y = classifier(x).softmax(-1)
        B = x.size(0)

        
        _step_mask = torch.zeros((B, max_segment, x.shape[1], 1), device='cpu').bool()
        _step_reward = torch.zeros((B, max_segment, 2), device='cpu').float()
        _mask = torch.zeros((B, x.shape[1], 1), device='cpu').bool()
        _next_mask = torch.zeros((B, x.shape[1], 1), device='cpu').bool()

        _alive_idx = torch.arange(B, device=device)
        _tensor_dict = None

        _batch_x = x.clone()
        _batch_y = y.clone()
        _batch_B = B
        for _step in range(1, max_segment+1):

            if _tensor_dict is not None:
                _batch_x     = _batch_x[~_is_done]
                _batch_y     = _batch_y[~_is_done]
                _curr_mask   = _tensor_dict['next', 'curr_mask'][~_is_done]
                _alive_idx   = _alive_idx[~_is_done]
                _batch_B     = _batch_x.shape[0]
                if _is_done.all():
                    break
            
            _tensor_dict = TensorDict(
                {
                    "x": _batch_x,
                    "y": _batch_y,
                    "curr_mask": torch.zeros_like(x, dtype=bool) if _tensor_dict is None else _curr_mask
                }, 
                batch_size=(_batch_B,), device=device)
            # ------------------------ New Segment ------------------
            # _dist = policy_module.get_dist(_tensor_dict)
            env.step(
                _tensor_dict, 
                policy_module, 
                reward_fn, 
                partial(terminate_fn, step=_step), 
                mode=True
            )
            # ------------------------------------------
            _is_done = _tensor_dict['next', 'done'].view(-1).cpu()
            _step_mask[_alive_idx.cpu(), _step-1, :] = _tensor_dict["next", 'new_mask'].clone().cpu()
            _mask[_alive_idx.cpu(), :] = _tensor_dict['curr_mask'].clone().cpu()
            _next_mask[_alive_idx.cpu(), :] = _tensor_dict['next', 'curr_mask'].clone().cpu()
            _step_reward[_alive_idx.cpu(), _step-1, 0] = _tensor_dict['next', 'reward_dict']["CrossEntropy"].clone().cpu()
            _step_reward[_alive_idx.cpu(), _step-1, 1] = _tensor_dict['next', 'reward_dict']["Length"].clone().cpu()

        step_mask.append(_step_mask)
        step_reward.append(_step_reward)
        mask.append(_mask)
        next_mask.append(_next_mask)

    step_mask = torch.concat(step_mask, dim=0)
    step_reward = torch.concat(step_reward, dim=0)
    mask = torch.concat(mask, dim=0)
    next_mask = torch.concat(next_mask, dim=0)


    sparsity = mask.float().mean().item()
    # baseline_mask = {}
    # baseline_attr = torch.load(f"./model_ckpt/{dataset}/{backbone[0]}_attr_{split}_{seed}.pth")
    # for m, attr in baseline_attr.items():
    #     _threshold = torch.quantile(attr, 1-sparsity, keepdim=True, dim=1)
    #     baseline_mask[m] = torch.where(attr >= _threshold, True, False).bool().cpu()
    
    baseline_mask = {}
    baseline_mask['Ours'] = mask
    attr = torch.rand_like(attr)
    _threshold = torch.quantile(attr, 1-sparsity, keepdim=True, dim=1)
    baseline_mask['random'] = torch.where(attr >= _threshold, True, False).bool().cpu()
    baseline_mask['original'] = torch.ones_like(attr)
    
    y = test_set.data['y']
    gt_mask = test_set.data['true']
    saliency_index = (y != 0)
    for m, b_mask in baseline_mask.items():
        print(m, round(f1_score(gt_mask[saliency_index], b_mask[saliency_index], average='samples'), 2))


@torch.no_grad()
def test_sample_plot(
    epoch,
    policy_module,
    predictor,
    plot_set,
    num_classes,
    reward_fn,
    terminate_fn,
    mask_fn,
    seg_dist,
    seq_len,
    max_segment,
    device,
 ):
    
    policy_module.load_state_dict(torch.load(f'./model_ckpt/policy.pth'))
    policy_module.eval()

    class_samples = defaultdict(list)
    for batch in plot_set:
        x = batch['x']; y = batch['y']
        if len(class_samples[y.item()]) < 20:
            d = {"x": x.squeeze(0), "y": y}
            if "gt_mask" in batch:
                d.update({"gt_mask": batch["gt_mask"].squeeze(0)})
            class_samples[y.item()].append(d)

        tot = 0
        for k, v in class_samples.items():
            tot += len(v)
        if tot == 20 * num_classes:
            break

    n_cols, rows_per_class = 5, 4    
    n_rows = rows_per_class * num_classes

    fig = plt.figure(figsize=(20, 4 * n_rows))
    outer_gs = fig.add_gridspec(n_rows, n_cols)

    for cls in range(num_classes):
        for idx, (d_dict) in enumerate(class_samples[cls]):
            x_sample, y_sample = d_dict['x'], d_dict['y']
            gt_mask = d_dict.get("gt_mask", None)

            row = cls * rows_per_class + idx // n_cols
            col = idx % n_cols

            inner = outer_gs[row, col].subgridspec(
                4, 1, height_ratios=[2, 1, 1, 1], hspace=0.05)

            ax_top   = fig.add_subplot(inner[0])
            ax_mask  = fig.add_subplot(inner[1], sharex=ax_top)
            ax_start = fig.add_subplot(inner[2], sharex=ax_top)
            ax_end   = fig.add_subplot(inner[3], sharex=ax_top)


            x_tensor = x_sample.unsqueeze(0).to(device)  # [1, T, C]
            y_target = y_sample.unsqueeze(0).to(device)


            td = None
            mask_rows, start_rows, end_rows = [], [], []
            start_params, end_params        = [], []
            step_masks = []
            for _step in range(1, max_segment+1):
                if td is not None:
                    is_done = td['next', 'done'].view(-1)
                    if is_done.all():
                        _step -= 1
                        break
                    x_tensor = x_tensor[~is_done]
                    y_target = y_target[~is_done]
                    curr_mask = td['next', 'curr_mask'][~is_done]

                td = TensorDict(
                    {"x": x_tensor, 
                     'y':y_target ,
                     "curr_mask": torch.zeros_like(x_tensor, dtype=torch.bool) if td is None else curr_mask
                    },
                    batch_size=(1,), device=device
                )

                dist = policy_module.get_dist(td)
                start_rows.append(dist.start_marginal())
                start_params.append(dist.start_param())

                end_rows.append(dist.end_marginal())
                end_params.append(dist.end_param())

                mask_rows.append(dist.calculate_marginal_mask().cpu().squeeze().numpy())
                env.step(td, policy_module, reward_fn, partial(terminate_fn, step=_step), mode=True)

                new_mask = td['next', 'new_mask'].cpu().squeeze().numpy()
                step_masks.append(new_mask.astype(int))


            x_np        = x_sample.squeeze().numpy()
            x_masked    = mask_fn(x_tensor, td["next", "curr_mask"])
            x_probs     = predictor(x_masked).softmax(dim=-1)[:, cls].squeeze().item()
            x_masked    = x_masked.cpu().squeeze().numpy()
            masked      = np.where(td["next", 'curr_mask'].cpu().squeeze().numpy(), x_masked, np.nan)
            y_lo, y_hi  = ax_top.get_ylim()
            band_h      = (y_hi - y_lo) / _step

            ax_top.plot(x_np, label="original")
            ax_top.plot(masked, label="masked")
            if gt_mask is not None:
                ax_top.plot(gt_mask, label="gt_mask")
            ax_top.legend()

            ax_top.set_title(f"Class {cls} Sample {idx+1} Probs {x_probs:.3f}, Seg {_step}", fontsize=9)


            for s_idx, m in enumerate(step_masks):
                y0 = y_hi - band_h * s_idx
                y1 = y0   - band_h
                color = mpl.cm.tab20(s_idx % 20)
                
                d  = np.diff(m.astype(int))
                st = np.where(d == 1)[0] + 1
                ed = np.where(d == -1)[0] + 1
                if m[0]:  st = np.r_[0, st]
                if m[-1]: ed = np.r_[ed, m.size]
                
                for a, b in zip(st, ed):
                    ax_top.axvspan(a, b, y0, y1, facecolor=color, alpha=0.7,
                                edgecolor=None)


            ax_mask.imshow(np.vstack(mask_rows),
                           aspect="auto", cmap="Greens",
                           interpolation="nearest",
                           extent=[0, seq_len-1, 0, len(mask_rows)])
            ax_mask.set_yticks([]); ax_mask.set_ylabel("mask", fontsize=6)

            # start / end heat-map
            ax_start.imshow(np.vstack(start_rows),
                            aspect="auto", cmap="Blues",
                            interpolation="nearest",
                            extent=[0, seq_len-1, 0, len(start_rows)])
            ax_start.set_yticks(np.arange(len(start_rows)) + 0.5)
            ax_start.set_yticklabels(start_params, fontsize=6)
            ax_start.set_ylabel("start p", fontsize=6)

            ax_end.imshow(np.vstack(end_rows),
                          aspect="auto", cmap="Oranges",
                          interpolation="nearest",
                          extent=[0, seq_len-1, 0, len(end_rows)])
            ax_end.set_yticks(np.arange(len(end_rows)) + 0.5)
            ax_end.set_yticklabels(end_params, fontsize=6)
            ax_end.set_ylabel("end p", fontsize=6)
            ax_end.set_xlabel("t", fontsize=6)

    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.suptitle(f"Epoch {epoch} – multi-segment masking & start/end probs",
                 fontsize=14)
    plt.close(fig)

@torch.no_grad()
def valid_step(
    epoch,
    loader,
    plot_set,
    num_classes,
    policy_module,
    predictor,
    reward_fn,
    terminate_fn,
    max_segment,
    early_stop,
    mask_fn,
    history,
    seg_dist,
    seq_len,
    classifier,
    device,
):
    val_epoch_total  = 0
    val_sample_total = 0
    val_avg_length   = 0.0
    val_avg_reward   = -999
    val_avg_each_reward   = defaultdict(lambda:0.)
    val_targets, val_trues, val_masked_preds, val_masked_probs = [], [], [], []
    val_og_targets = []

    for batch in loader:
        x = batch['x'].to(device)
        B = x.size(0)
        val_sample_total += B
        y_target = classifier(x).softmax(-1)


        x_tensor = x.clone()
        y_tensor = y_target.clone()
        y_true = batch['y'].to(device)

        td = None
        td_x, td_y, td_mask = [], [], []
        for _step in range(1, max_segment+1):
            if td is not None:
                if is_done.all():
                    break
                x_tensor = x_tensor[~is_done]
                y_tensor = y_tensor[~is_done]
                curr_mask = td['next', 'curr_mask'][~is_done]
                B = x_tensor.shape[0]

            td = TensorDict(
                {
                    "x": x_tensor,
                    "y": y_tensor,
                    "curr_mask": torch.zeros_like(x, dtype=bool) if td is None else curr_mask
                }, 
                batch_size=(B,), device=device)
            
            env.step(
                td, 
                policy_module, 
                reward_fn, 
                partial(terminate_fn, step=_step), 
                mode=True
            )
            is_done = td['next', 'done'].view(-1)
            if is_done.any():
                td_x.append(x_tensor[is_done])
                td_y.append(y_tensor[is_done])
                td_mask.append(td['curr_mask'][is_done])
                val_avg_length   += td["curr_mask"][is_done].sum([1, 2], dtype=float).sum().item()
            val_epoch_total  += B
            val_avg_reward   += td["next", "reward"][~is_done].sum().item()


        td_x = torch.concat(td_x, dim=0)
        td_y = torch.concat(td_y, dim=0)
        td_mask = torch.concat(td_mask, dim=0)

        x_masked  = mask_fn(td_x, td_mask)
        y_masked  = predictor(x_masked).softmax(-1)
        y_pred = y_masked.argmax(-1)

        val_trues.append(y_true.cpu())

        val_og_targets.append(y_target.argmax(-1).cpu())
        val_targets.append(td_y.argmax(-1).cpu())

        val_masked_preds.append(y_pred.cpu())
        val_masked_probs.append(y_masked.detach().cpu())

    val_avg_length   /= val_sample_total
    val_avg_reward   /= val_sample_total
    for reward_name in val_avg_each_reward.keys():
        val_avg_each_reward[reward_name] /= val_epoch_total

    val_trues        = torch.cat(val_trues).numpy()
    val_og_targets   = torch.cat(val_og_targets).numpy()

    val_targets      = torch.cat(val_targets).numpy()
    val_masked_preds = torch.cat(val_masked_preds).numpy()
    val_masked_probs = torch.cat(val_masked_probs).numpy()

    masked_acc = accuracy_score(val_trues, val_masked_preds)
    masked_f1  = f1_score(val_trues, val_masked_preds, average='macro' if num_classes == 2 else 'macro')

    masked_auroc = roc_auc_score(
        val_trues, 
        val_masked_probs if num_classes != 2 else val_masked_probs[:, 1], 
        average=None if num_classes == 2 else 'macro', 
        multi_class='raise' if num_classes==2 else 'ovr'
    )
    masked_auprc = average_precision_score(
        val_trues, 
        val_masked_probs if num_classes != 2 else val_masked_probs[:, 1], 
        average=None if num_classes == 2 else 'macro'
    )

    msg = f"\t| Avg Val Length: {val_avg_length:.4f}"\
        + f" | Avg Val Reward: {val_avg_reward:.4f}"\
        + f"\n\t| Masked Acc: {masked_acc:.2f}"\
        + f" | Masked F1: {masked_f1:.2f}"\
        + f" | AUROC : {masked_auroc:.2f}"\
        + f" | AUPRC : {masked_auprc:.2f}"
    print(msg)

    history['valid_length'].append(val_avg_length)
    history['valid_reward'].append(val_avg_reward)
    history['masked_acc'].append(masked_acc)
    history['masked_f1'].append(masked_f1)

    if early_stop(val_avg_reward): # is imporved
        best_reward = early_stop.best
        os.makedirs('./model_ckpt/', exist_ok=True)
        torch.save(policy_module.state_dict(), './model_ckpt/policy.pth')


def ppo_update(
    epoch,
    policy_module,
    value_module,
    loss_module,
    ppo_epochs,
    replay_buffer,
    rollout_len,
    policy_optim,
    policy_scheduler,
    value_optim,
    value_scheduler,
    history,
    device,
):
    policy_module.train(); value_module.train()

    epoch_total = 0.
    avg_length = 0.
    avg_reward = 0.
    avg_each_rewards = defaultdict(lambda:0.)
    avg_actor_loss = 0.
    avg_critic_loss = 0.

    for _ in range(ppo_epochs):
        per_epochs = 0.
        for mb in replay_buffer:
            B = mb.shape[0]
            mb = mb.to(device)
            loss_td = loss_module(mb)
            loss = loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_entropy"]

            policy_optim.zero_grad()
            value_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_module.parameters(), 1.)
            torch.nn.utils.clip_grad_norm_(value_module.parameters(), 1.)
            policy_optim.step()
            value_optim.step()

            epoch_total += B
            per_epochs += B
            avg_length += mb['curr_mask'].sum([1, 2], dtype=float).sum().item()
            avg_reward += mb['next', 'reward'].sum().item()
            for reward_name, each_reward in mb['next', 'reward_dict'].items():
                avg_each_rewards[f"reward/train_{reward_name}"] += each_reward.sum().item()

            avg_actor_loss += loss_td['loss_objective'] * B
            avg_critic_loss += loss_td['loss_critic'] * B

            if per_epochs >= rollout_len:
                break

    avg_length /= epoch_total
    avg_reward /= epoch_total
    for reward_name in avg_each_rewards.keys():
        avg_each_rewards[reward_name] /= epoch_total
    avg_actor_loss /= epoch_total
    avg_critic_loss /= epoch_total

    msg = f"\t| Avg Actor Loss: {avg_actor_loss:.4f} " \
        + f"| Avg Critic Loss: {avg_critic_loss:.4f} " \
        + f"\n\t| Avg Length: {avg_length:.4f} " \
        + f"| Avg Reward: {avg_reward:.4f}"
    print(msg)
    history['actor_loss'].append(avg_actor_loss)
    history['critic_loss'].append(avg_critic_loss)
    history['train_length'].append(avg_length)
    history['train_reward'].append(avg_reward)

    policy_module.eval(); value_module.eval()
    # policy_scheduler.step(); value_scheduler.step()
    

def predictor_train(
    epoch,
    predictor,
    classifier,
    predictor_epochs,
    replay_buffer,
    loader,
    collect_normal_data,
    rollout_len,
    pred_optim,
    predictor_scheduler,
    mask_fn,  
    history,
    target,
    device,
):
    predictor.train()
    pred_epoch_total = 0.
    avg_pred_loss = 0.
    for _ in range(predictor_epochs):
        if not collect_normal_data:
            per_epochs = 0
            for mb in replay_buffer:
                x = mb['x'].to(device)
                y = mb['y'].to(device)
                B = x.size(0)
                with torch.no_grad():
                    mask = torch.logical_or(mb['curr_mask'], mb['next', 'curr_mask']).to(device)
                    masked_x = mask_fn(x, mask)
                logits = predictor(masked_x)
                loss = F.cross_entropy(logits, y)

                pred_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.)
                pred_optim.step()

                pred_epoch_total += B
                per_epochs += B
                avg_pred_loss += loss.item() * B
                if per_epochs >= rollout_len:
                    break

        elif collect_normal_data:
            per_epochs = 0
            for mb, batch in zip(replay_buffer, cycle(loader)):
                x_1 = mb['x'].to(device)
                y_1 = mb['y'].to(device)

                B_1 = x_1.size(0)
                with torch.no_grad():
                    mask = torch.logical_or(mb['curr_mask'], mb['next', 'curr_mask']).to(device)
                    masked_x = mask_fn(x_1, mask)
                
                x_2 = batch['x'].to(device)
                if target == 'y':
                    y_2 = batch['y'].to(device)
                elif target == 'model':
                    with torch.no_grad():
                        y_2 = classifier(x_2).softmax(-1)
                B_2 = x_2.size(0)

                x = torch.concat([masked_x, x_2], dim=0)
                y = torch.concat([y_1, y_2], dim=0)

                B = B_1 + B_2

                logits = predictor(x)
                loss = F.cross_entropy(logits, y)

                pred_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.)
                pred_optim.step()

                pred_epoch_total += B
                per_epochs += B
                avg_pred_loss += loss.item() * B
                if per_epochs >= rollout_len:
                    break
        
    avg_pred_loss /= pred_epoch_total

    msg = f"\t| Avg Predictor Loss: {avg_pred_loss:.4f}"; print(msg)

    history['pred_loss'].append(avg_pred_loss)
    wandb.log({
        "pred/loss": avg_pred_loss
    }, step=epoch)
    predictor.eval(); predictor_scheduler.step()
    

@torch.no_grad()  
def collect_samples(
    epoch,
    policy_module,
    value_module,
    advantage_module,
    classifier,
    replay_buffer,
    rollout_len,
    reward_fn,
    terminate_fn,
    max_segment,
    cycle_loader,
    device
):
    collected = 0
    n_samples = 0
    for batch in cycle_loader:
        x = batch['x'].to(device)
        B = x.size(0)
        y_target = classifier(x).softmax(-1)
            
        n_samples += B

        td = None
        for _step in range(1, max_segment+1):

            if td is not None:
                is_done = td['next', 'done'].view(-1)
                if is_done.all():
                    break
                x = x[~is_done]
                y_target = y_target[~is_done]
                curr_mask = td['next', 'curr_mask'][~is_done]
                B = x.shape[0]

            td = TensorDict(
                {
                    "x": x,
                    "y": y_target,
                    "curr_mask": torch.zeros_like(x, dtype=bool) if td is None else curr_mask,
                }, 
                batch_size=(B,), device=device)
            
            env.step(td, policy_module, reward_fn, partial(terminate_fn, step=_step))
            value_module(td)
            value_module(td['next'])
            td['advantage']     = td['next', 'reward'] + 0.99 * (~td['next', 'done'] * td['next', 'state_value']) - td['state_value']
            td['value_target']  = td['advantage'] + td['state_value']
            
            replay_buffer.extend(td.view(-1).detach().cpu().clone())
            collected += B
            
            if collected >= rollout_len:
                break
        if collected >= rollout_len:
                break

        
def pretraining_predictor(
    predictor_pretrain,
    predictor,
    pred_optim,
    loader,
    seq_len,
    mask_fn,
    device,
):  
    if predictor_pretrain == 0 or predictor_pretrain is None:
        return
    
    predictor.train()
    pbar = tqdm(range(predictor_pretrain))
    for _ in pbar:
        total_loss = 0.
        y_true = []
        y_pred = []

        total = 0.
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            B = x.size(0)


            indices, _ = torch.multinomial(
                torch.tensor([1/seq_len]).repeat(B, seq_len), 2, replacement=True
            ).sort()
            start, end = indices[:, :1], indices[:, 1:]
            arange = torch.arange(seq_len).unsqueeze(0)
            mask = torch.logical_and(start <= arange, arange <= end).unsqueeze(-1).to(device)
            segments = mask_fn(x, mask)
            logits = predictor(torch.concat([segments, x], dim=0))
            loss = F.cross_entropy(logits, torch.concat([y, y], dim=0))

            pred_optim.zero_grad()
            loss.backward()
            pred_optim.step()

            total_loss += B * loss.item()
            y_pred.append(y.cpu().detach().numpy())
            y_pred.append(logits.softmax(-1).argmax(-1).cpu().detach().numpy())
            total += B
        pbar.set_postfix(loss = round(total_loss / total, 4))
    predictor.eval()
    
