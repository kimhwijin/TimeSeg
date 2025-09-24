import argparse
from datasets.TimeX import SpikeTrainDataset
import os
import torch as th
import numpy as np
import random
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_dist', type=str, choices=['cat_cat', 'cat_nb', 'cauchy_cat_cat'])
    parser.add_argument('--weights', type=str)
    parser.add_argument('--reward_types', type=str)
    parser.add_argument('--terminate_type', type=str, default='ce_diff')
    parser.add_argument('--backbone', type=str, default='tcn,cnn')
    parser.add_argument('--train_type', type=str, default='ppo')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--mask_type', type=str, default='mean')
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--rollout_len", type=int, default=1024)
    parser.add_argument("--ppo_epochs", type=int)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_segment", type=int, default=5)
    parser.add_argument("--early_stop", type=int, default=0)
    parser.add_argument("--note", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.3)

    args = parser.parse_args()


    
    seg_dist            = args.seg_dist
    weights             = list(map(float, args.weights.split(',')))
    if len(weights) == 2 and args.reward_types is None:
        args.reward_types = 'exp_ce,length'
    terminate_type      = args.terminate_type
    reward_types        = list(args.reward_types.split(','))
    backbone            = args.backbone
    backbone            = list(backbone.split(","))
    
    train_type          = args.train_type
    dataset             = args.dataset
    split               = args.split
    mask_type           = args.mask_type
    epochs              = args.epochs
    ppo_epochs          = args.ppo_epochs
    device              = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_segment         = args.max_segment
    early_stop          = args.early_stop
    batch_size          = args.batch_size
    rollout_len         = args.rollout_len
    entropy_coef        = args.entropy_coef
    seed                = args.seed
    threshold           = args.threshold

    set_seed(seed)

    if train_type == 'ppo':
        import main_ppo
        main_ppo.main(
            seg_dist            = seg_dist,
            reward_types        = reward_types,
            terminate_type      = terminate_type,
            weights             = weights,
            backbone            = backbone,
            dataset             = dataset,
            split               = split,
            entropy_coef        = entropy_coef,
            mask_type           = mask_type,
            epochs              = epochs,
            batch_size          = batch_size,
            rollout_len         = rollout_len,
            ppo_epochs          = ppo_epochs,
            max_segment         = max_segment,
            early_stop          = early_stop,
            seed                = seed,
            threshold           = threshold,
            device              = device,
        )

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    print(f"set seed as {seed}")



if __name__ == "__main__":
    main()