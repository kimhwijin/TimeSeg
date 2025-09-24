import numpy as np
import os
import pickle as pkl
import torch as th

from datasets.dataset import DataModule, Dataset
from txai.utils.data.preprocess import process_MITECG

file_dir = os.path.dirname(__file__)

class MITECG(DataModule):

    def __init__(
        self,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "MITECG",
        ),
        batch_size: int = 32,
        prop_val: float = 0.2,
        n_folds: int = 5,
        fold: int = None,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            prop_val=prop_val,
            n_folds=n_folds,
            fold=fold,
            num_workers=num_workers,
            seed=seed,
        )
        self._mean = None
        self._std = None
        self.data = None

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir, "all_data", "X.pt")):
            raise RuntimeError("No data exists or wrong path")

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train = Dataset(self.preprocess("train"))
            self.val = Dataset(self.preprocess("val"))
            
        if stage == "test" or stage is None:
            self.test = Dataset(self.preprocess("test"))

        if stage == "predict" or stage is None:
            self.predict = Dataset(self.preprocess("test"))

    def preprocess(self, split: str = "train") -> dict:
        if self.data == None:
            self.data = process_MITECG(
                split_no = self.fold + 1, 
                base_path = self.data_dir,
                device = 'cpu', hard_split = True,
                normalize = False, balance_classes = False, div_time = False, need_binarize = True, exclude_pac_pvc = True
            )
        features, y = self.data[split]
        features = features.transpose(0, 1)
        
        if split == "train":
            self._mean = features.mean(dim=(0, 1), keepdim=True)
            self._std = features.std(dim=(0, 1), keepdim=True)
            
        EPS = 1e-5
        features = (features - self._mean) / (self._std + EPS)
        if split == 'test':
            return {
                "x": features.float(),
                "y": y.long(),
                'true': self.true_saliency()
            }
        
        return {
            "x": features.float(),
            "y": y.long()
        }
    
    def true_saliency(self):
        return self.data['gt_exps'].bool().transpose(0, 1)
    


import torch
def process_MITECG(split_no = 1, device = None, hard_split = False, normalize = False, exclude_pac_pvc = False, balance_classes = False, div_time = False, 
        need_binarize = False, base_path = None):

    split_path = 'split={}.pt'.format(split_no)
    idx_train, idx_val, idx_test = torch.load(os.path.join(base_path, split_path))
    if hard_split:
        X = torch.load(os.path.join(base_path, 'all_data/X.pt'))
        y = torch.load(os.path.join(base_path, 'all_data/y.pt')).squeeze()

        times = torch.zeros(X.shape[0],X.shape[1])
        for i in range(X.shape[1]):
            times[:,i] = torch.arange(360)

        saliency = torch.load(os.path.join(base_path, 'all_data/saliency.pt'))
        
    else:
        X, times, y = torch.load(os.path.join(base_path, 'all_data.pt'))
    idx_train = idx_train.long()
    idx_val = idx_val.long()
    idx_test = idx_test.long()

    Ptrain, time_train, ytrain = X[:,idx_train,:].float(), times[:,idx_train], y[idx_train].long()
    Pval, time_val, yval = X[:,idx_val,:].float(), times[:,idx_val], y[idx_val].long()
    Ptest, time_test, ytest = X[:,idx_test,:].float(), times[:,idx_test], y[idx_test].long()

    if normalize:

        mu = Ptrain.mean()
        std = Ptrain.std()
        Ptrain = (Ptrain - mu) / std
        Pval = (Pval - mu) / std
        Ptest = (Ptest - mu) / std

    if div_time:
        time_train = time_train / 60.0
        time_val = time_val / 60.0
        time_test = time_test / 60.0

    if exclude_pac_pvc:
        train_mask_in = (ytrain < 3)
        Ptrain = Ptrain[:,train_mask_in,:]
        time_train = time_train[:,train_mask_in]
        ytrain = ytrain[train_mask_in]

        val_mask_in = (yval < 3)
        Pval = Pval[:,val_mask_in,:]
        time_val = time_val[:,val_mask_in]
        yval = yval[val_mask_in]

        test_mask_in = (ytest < 3)
        Ptest = Ptest[:,test_mask_in,:]
        time_test = time_test[:,test_mask_in]
        ytest = ytest[test_mask_in]
    
    if need_binarize:
        ytrain = (ytrain > 0).long()
        ytest = (ytest > 0).long()
        yval = (yval > 0).long()

    if balance_classes:
        diff_to_mask = (ytrain == 0).sum() - (ytrain == 1).sum()
        all_zeros = (ytrain == 0).nonzero(as_tuple=True)[0]
        mask_out = all_zeros[:diff_to_mask]
        to_mask_in = torch.tensor([not (i in mask_out) for i in torch.arange(Ptrain.shape[1])])
        print('Num before', (ytrain == 0).sum())
        Ptrain = Ptrain[:,to_mask_in,:]
        time_train = time_train[:,to_mask_in]
        ytrain = ytrain[to_mask_in]
        print('Num after 0', (ytrain == 0).sum())
        print('Num after 1', (ytrain == 1).sum())

        diff_to_mask = (yval == 0).sum() - (yval == 1).sum()
        all_zeros = (yval == 0).nonzero(as_tuple=True)[0]
        mask_out = all_zeros[:diff_to_mask]
        to_mask_in = torch.tensor([not (i in mask_out) for i in torch.arange(Pval.shape[1])])
        print('Num before', (yval == 0).sum())
        Pval = Pval[:,to_mask_in,:]
        time_val = time_val[:,to_mask_in]
        yval = yval[to_mask_in]
        print('Num after 0', (yval == 0).sum())
        print('Num after 1', (yval == 1).sum())

        diff_to_mask = (ytest == 0).sum() - (ytest == 1).sum()
        all_zeros = (ytest == 0).nonzero(as_tuple=True)[0]
        mask_out = all_zeros[:diff_to_mask]
        to_mask_in = torch.tensor([not (i in mask_out) for i in torch.arange(Ptest.shape[1])])
        print('Num before', (ytest == 0).sum())
        Ptest = Ptest[:,to_mask_in,:]
        time_test = time_test[:,to_mask_in]
        ytest = ytest[to_mask_in]
        print('Num after 0', (ytest == 0).sum())
        print('Num after 1', (ytest == 1).sum())

    train_chunk = ECGchunk(Ptrain, None, time_train, ytrain, device = device)
    val_chunk = ECGchunk(Pval, None, time_val, yval, device = device)
    test_chunk = ECGchunk(Ptest, None, time_test, ytest, device = device)

    print('Num after 0', (yval == 0).sum())
    print('Num after 1', (yval == 1).sum())
    print('Num after 0', (ytest == 0).sum())
    print('Num after 1', (ytest == 1).sum())

    if hard_split:
        gt_exps = saliency.transpose(0,1).unsqueeze(-1)[:,idx_test,:]
        if exclude_pac_pvc:
            gt_exps = gt_exps[:,test_mask_in,:]
        # return train_chusnk, val_chunk, test_chunk, gt_exps
        return {
            "train": (train_chunk.X, train_chunk.y),
            "val": (val_chunk.X, val_chunk.y),
            "test": (test_chunk.X, test_chunk.y),
            "gt_exps": gt_exps
        }
    else:
        return train_chunk, val_chunk, test_chunk
    

import random
class ECGchunk:
    def __init__(self, train_tensor, static, time, y, device = None):
        self.X = train_tensor.to(device)
        self.static = None if static is None else static.to(device)
        self.time = time.to(device)
        self.y = y.to(device)

    def choose_random(self):
        n_samp = self.X.shape[1]           
        idx = random.choice(np.arange(n_samp))

        static_idx = None if self.static is None else self.static[idx]
        return self.X[idx,:,:].unsqueeze(dim=1), \
            self.time[:,idx].unsqueeze(dim=-1), \
            self.y[idx].unsqueeze(dim=0), \
            static_idx

    def get_all(self):
        static_idx = None
        return self.X, self.time, self.y, static_idx

    def __getitem__(self, idx): 
        static_idx = None if self.static is None else self.static[idx]
        return self.X[:,idx,:], \
            self.time[:,idx], \
            self.y[idx].unsqueeze(dim=0)
