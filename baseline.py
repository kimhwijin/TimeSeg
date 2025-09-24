import sys
from pathlib import Path
p = str(Path(__file__).absolute().parent)
sys.path.append(p)


import multiprocessing as mp
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader, TensorDataset

# Package
from datasets import get_datamodule
from models.classifier import ClassifierNet, get_classifier
from explainers import DynaMask, FIT, WinIT, TimeX, TimeXplusplus, IntegratedGradientsAbs, LimeSegment, ExtremalMask, ContraLSP, LIME, GradientShapAbs, DeepLiftAbs, AugmentedOcclusion
import os

import os
import torch

# CPU 코어 제약 (4개만 사용)
N_THREADS = 4
torch.set_num_threads(N_THREADS)

os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_THREADS)

# 나머지 import (scipy, sklearn, LimeSegment 등)
from scipy import signal
import numpy as np
import stumpy
from sklearn.linear_model import Ridge

def main(
    data,
    model_type,
    fold,
    seed,
    inject=False
):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    accelerator = 'cuda'
    device_id = 1

    print("Baseline Start")
    datamodule, num_features, max_len, num_classes = get_datamodule(data, fold, seed, model_type)
    classifier = get_classifier(num_features, num_classes, max_len, model_type, data)
    classifier.load_state_dict(torch.load(f"./model_ckpt/{data}/{model_type}_classifier_{fold}_{seed}.ckpt", weights_only=False)['state_dict'])

    lock = mp.Lock()
    with lock:
        x_train = datamodule.preprocess(split="train")["x"].to(device)
        x_test = datamodule.preprocess(split="test")["x"].to(device)
        y_train = datamodule.preprocess(split="train")["y"].to(device)
        y_test = datamodule.preprocess(split="test")["y"].to(device)
        # mask_train = datamodule.preprocess(split="train")["mask"].to(device)
        
        mask_test = torch.ones_like(x_test)


    classifier.eval()
    classifier.to(device)
    if accelerator == "cuda":
        torch.backends.cudnn.enabled = False

    test_dataset = TensorDataset(x_test, mask_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    data_mask = mask_test
    data_len, t_len, _ = x_test.shape
    timesteps=(
        torch.linspace(0, 1, t_len, device=x_test.device)
        .unsqueeze(0)
        .repeat(data_len, 1)
    )

    if not os.path.exists(f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth'):
        attr = dict()
    else:
        attr = torch.load(f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth', weights_only=False)

    # print("AFO")
    # attr['afo'] = AugmentedOcclusion.explain(
    #     classifier, x_train, data_mask, x_test, y_test, temporal_additional_forward_args=(False, False, False)
    # )
    # torch.save(attr, f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth')

    if ('ig_abs' not in attr.keys()) or inject:
        print("IGAbs")
        attr['ig_abs'] = IntegratedGradientsAbs.explain(
            classifier, test_loader, timesteps, num_classes, device
        )
        torch.save(attr, f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth')

    if ('dyna_mask' not in attr.keys()) or inject:
        print("DynaMask")
        attr["dyna_mask"] = DynaMask.explain(
            classifier, accelerator, device_id, x_test, data_mask, timesteps, device
        )
        torch.save(attr, f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth')

    # if ('timex' not in attr.keys()) or inject:
    #     print("TimeX")
    #     attr["timex"] = TimeX.explain(
    #         classifier, timesteps, x_train, y_train, x_test, y_test, num_features, num_classes, max_len, data, fold, model_type, seed, test_loader, device
    #     )
    #     torch.save(attr, f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth')
    if ('timex++' not in attr.keys()) or inject:
        print("TimeX++")
        attr["timex++"] = TimeXplusplus.explain(
            classifier, timesteps, x_train, y_train, x_test, y_test, num_features, num_classes, max_len, data, fold, model_type, seed, test_loader, device
        )
        torch.save(attr, f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth')
        
    if ('winit' not in attr.keys()) or inject:
        print("WinIT")
        attr["winit"] = WinIT.explain(
            classifier, datamodule, data, timesteps, num_features, False, fold, model_type, test_loader, device
        )
        torch.save(attr, f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth')

    if 'lime_segment' not in attr.keys():
        attr["lime_segment"] = LimeSegment.explain(
            classifier, test_dataset
        )
        torch.save(attr, f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth')

    # print("LIME")
    # attr['lime'] = LIME.explain(
    #     classifier, x_test, y_test
    # )
    # torch.save(attr, f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth')

    # print("DeepLiftAbs")
    # attr['deeplift_abs'] = DeepLiftAbs.explain(
    #     classifier, test_loader, timesteps, device
    # )
    # torch.save(attr, f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth')

    # print("GradShapAbs")
    # attr['gradshap_abs'] = GradientShapAbs.explain(
    #     classifier, test_loader, timesteps, device
    # )
    # torch.save(attr, f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth')
    
    # if 'extr_mask' not in attr.keys():
    #     print("ExtrMask")
    #     attr["extr_mask"] = ExtremalMask.explain(
    #         classifier, accelerator, device_id, x_test, data_mask, timesteps, device, lambda_1=0.01, lambda_2=10, mask_lr=0.01
    #     )
    #     torch.save(attr, f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth')

    # print("ContraLSP")
    # attr["ContraLSP"] = ContraLSP.explain(
    #     classifier, accelerator, device_id, x_test, data_mask, timesteps, device, lambda_1=0.005, lambda_2=0.01, mask_lr=0.1
    # )
    # torch.save(attr, f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth')
        
    # print("FIT")
    # attr["fit"] = FIT.explain(
    #     classifier, datamodule, data, timesteps, num_features, True, fold, model_type, test_loader, device
    # )
    # torch.save(attr, f'./model_ckpt/{data}/{model_type}_attr_{fold}_{seed}.pth')


import random
import numpy as np
import torch as th

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    # th.backends.cudnn.deterministic = True
    # th.backends.cudnn.benchmark = False
    print(f"set seed as {seed}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--inject", action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    main(args.data, args.model_type, args.fold, args.seed, args.inject)
    