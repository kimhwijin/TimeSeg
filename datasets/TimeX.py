import torch
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
import os
import numpy as np
import random
f_dir = Path(__file__).parent.parent

def process_Synth(split_no, device, base_path, regression = False, label_noise = None):

    split_path = os.path.join(base_path, 'split={}.pt'.format(split_no))

    D = torch.load(split_path, weights_only=False)

    D['train_loader'].X = D['train_loader'].X.float().to(device).transpose(0, 1)
    D['train_loader'].times = D['train_loader'].times.float().to(device).transpose(0, 1)
    if regression:
        D['train_loader'].y = D['train_loader'].y.float().to(device)
    else:
        D['train_loader'].y = D['train_loader'].y.long().to(device)

    val = []
    val.append(D['val'][0].float().to(device).transpose(0, 1))
    val.append(D['val'][1].float().to(device).transpose(0, 1))
    val.append(D['val'][2].long().to(device))
    if regression:
        val[-1] = val[-1].float()
    D['val'] = tuple(val)

    test = []
    test.append(D['test'][0].float().to(device).transpose(0, 1))
    test.append(D['test'][1].float().to(device).transpose(0, 1))
    test.append(D['test'][2].long().to(device))
    if regression:
        test[-1] = test[-1].float()
    D['test'] = tuple(test)

    if label_noise is not None:
        # Find some samples in training to switch labels:

        to_flip = int(label_noise * D['train_loader'].y.shape[0])
        to_flip = to_flip + 1 if (to_flip % 2 == 1) else to_flip # Add one if it isn't even

        flips = torch.randperm(D['train_loader'].y.shape[0])[:to_flip]

        max_label = D['train_loader'].y.max()

        for i in flips:
            D['train_loader'].y[i] = (D['train_loader'].y[i] + 1) % max_label

    D['gt_exps'] = D['gt_exps'].transpose(0, 1)
    return D


def process_Boiler_OLD(split_no, device, base_path):

    data = pd.read_csv(os.path.join(base_path, 'full.csv')).values
    data = data[:, 2:]  #remove time step

    window_size = 6
    segments_length = [1, 2, 3, 4, 5, 6]

    # Load path

    print('positive sample size:',sum(data[:,-1]))
    feature, label = [], []
    for i in range(window_size - 1, len(data)):
        label.append(data[i, -1])

        sample = []
        for length in segments_length:
            a = data[(i- length + 1):(i + 1), :-1]
            a = np.pad(a,pad_width=((0,window_size -length),(0,0)),mode='constant')# padding to [window_size, x_dim]
            sample.append(a)

        sample = np.array(sample)
        sample = np.transpose(sample,axes=((2,0,1)))[:,:,:]

        feature.append(sample)

    feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.int32)

    x_full = torch.tensor(feature.reshape(*feature.shape[:-2], -1)).permute(2,0,1)
    y_full = torch.from_numpy(label)

    # Make times:
    T_full = torch.zeros(36, x_full.shape[1])
    for i in range(T_full.shape[1]):
        T_full[:,i] = torch.arange(36)

    # Now split:
    idx_train, idx_val, idx_test = torch.load(os.path.join(base_path, 'split={}.pt'.format(split_no)))

    x_full, T_full, y_full = x_full.to(device), T_full.to(device), y_full.to(device).long()

    train_d = (x_full[:,idx_train,:].transpose(1, 0), 
               T_full[:,idx_train].transpose(1, 0), 
               y_full[idx_train]
               )
    val_d = (x_full[:,idx_val,:].transpose(1, 0), 
             T_full[:,idx_val].transpose(1, 0), 
             y_full[idx_val]
            )
    test_d = (x_full[:,idx_test,:].transpose(1, 0), 
              T_full[:,idx_test].transpose(1, 0) ,
              y_full[idx_test]
            )

    return train_d, val_d, test_d

    
def process_MITECG(split_no, base_path, device = None, hard_split = False, normalize = False, exclude_pac_pvc = False, balance_classes = False, div_time = False, need_binarize = False):
    split_path = 'split={}.pt'.format(split_no)
    idx_train, idx_val, idx_test = torch.load(os.path.join(base_path, split_path))
    if hard_split:
        X = torch.load(os.path.join(base_path, 'all_data/X.pt'))
        y = torch.load(os.path.join(base_path, 'all_data/y.pt')).squeeze()

        # Make times on the fly:
        times = torch.zeros(X.shape[0],X.shape[1])
        for i in range(X.shape[1]):
            times[:,i] = torch.arange(360)

        saliency = torch.load(os.path.join(base_path, 'all_data/saliency.pt'))
        
    else:
        X, times, y = torch.load(os.path.join(base_path, 'all_data.pt'))

    Ptrain, time_train, ytrain = X[:,idx_train,:].float(), times[:,idx_train], y[idx_train].long()
    Pval, time_val, yval = X[:,idx_val,:].float(), times[:,idx_val], y[idx_val].long()
    Ptest, time_test, ytest = X[:,idx_test,:].float(), times[:,idx_test], y[idx_test].long()

    if normalize:
        # Get mean, std of the whole sample from training data, apply to val, test:
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

    train_chunk = ECGchunk(Ptrain.transpose(1, 0), None, time_train.transpose(1, 0), ytrain, device = device)
    val_chunk = ECGchunk(Pval.transpose(1, 0), None, time_val.transpose(1, 0), yval, device = device)
    test_chunk = ECGchunk(Ptest.transpose(1, 0), None, time_test.transpose(1, 0), ytest, device = device)
    

    print('Num after 0', (yval == 0).sum())
    print('Num after 1', (yval == 1).sum())
    print('Num after 0', (ytest == 0).sum())
    print('Num after 1', (ytest == 1).sum())

    # N x T x D
    if hard_split:
        gt_exps = saliency.unsqueeze(-1)[idx_test,:,:]
        if exclude_pac_pvc:
            gt_exps = gt_exps[test_mask_in,:,:]
        return train_chunk, val_chunk, test_chunk, gt_exps
    else:
        return train_chunk, val_chunk, test_chunk
    


def process_Epilepsy(split_no, device, base_path):

    # train = torch.load(os.path.join(loc, 'train.pt'))
    # val = torch.load(os.path.join(loc, 'val.pt'))
    # test = torch.load(os.path.join(loc, 'test.pt'))

    split_path = 'split_{}.npy'.format(split_no)
    idx_train, idx_val, idx_test = np.load(os.path.join(base_path, split_path), allow_pickle = True)

    # Ptrain, Pval, Ptest = train['samples'].transpose(1, 2), val['samples'].transpose(1, 2), test['samples'].transpose(1, 2)
    # ytrain, yval, ytest = train['labels'], val['labels'], test['labels']

    X, y = torch.load(os.path.join(base_path, 'all_epilepsy.pt'))

    Ptrain, ytrain = X[idx_train], y[idx_train]
    Pval, yval = X[idx_val], y[idx_val]
    Ptest, ytest = X[idx_test], y[idx_test]

    T, F = Ptrain[0].shape
    D = 1

    Ptrain_static_tensor = np.zeros((len(Ptrain), D))

    mf, stdf = getStats(Ptrain)
    #print('Before tensor_normalize_other', Ptrain.shape)
    Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_ECG(Ptrain, ytrain, mf, stdf)
    Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_ECG(Pval, yval, mf, stdf)
    Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_ECG(Ptest, ytest, mf, stdf)
    #print('After tensor_normalize (X)', Ptrain_tensor.shape)

    Ptrain_tensor = Ptrain_tensor.permute(2, 0, 1)
    Pval_tensor = Pval_tensor.permute(2, 0, 1)
    Ptest_tensor = Ptest_tensor.permute(2, 0, 1)

    #print('Before s-permute', Ptrain_time_tensor.shape)
    Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
    Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
    Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

    # print('X', Ptrain_tensor)
    # print('time', Ptrain_time_tensor)
    print('X', Ptrain_tensor.shape)
    print('time', Ptrain_time_tensor.shape)
    # print('time of 0', Ptrain_time_tensor.sum())
    # print('train under 0', (Ptrain_tensor > 1e-10).sum() / Ptrain_tensor.shape[1])
    #print('After s-permute', Ptrain_time_tensor.shape)
    #exit()
    train_chunk = ECGchunk(Ptrain_tensor.transpose(1, 0), None, Ptrain_time_tensor.transpose(1, 0), ytrain_tensor, device = device)
    val_chunk = ECGchunk(Pval_tensor.transpose(1, 0), None, Pval_time_tensor.transpose(1, 0), yval_tensor, device = device)
    test_chunk = ECGchunk(Ptest_tensor.transpose(1, 0), None, Ptest_time_tensor.transpose(1, 0), ytest_tensor, device = device)

    return train_chunk, val_chunk, test_chunk

def process_PAM(split_no, device, base_path, gethalf):

    split_path = 'splits/PAMAP2_split_{}.npy'.format(split_no)
    idx_train, idx_val, idx_test = np.load(os.path.join(base_path, split_path), allow_pickle=True)

    Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
    arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)

    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]

    y = arr_outcomes[:, -1].reshape((-1, 1))

    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]

    #return Ptrain, Pval, Ptest, ytrain, yval, ytest

    T, F = Ptrain[0].shape
    D = 1

    Ptrain_tensor = Ptrain
    Ptrain_static_tensor = np.zeros((len(Ptrain), D))

    mf, stdf = getStats(Ptrain)
    Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_other(Ptrain, ytrain, mf, stdf)
    Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_other(Pval, yval, mf, stdf)
    Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_other(Ptest, ytest, mf, stdf)

    Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
    Pval_tensor = Pval_tensor.permute(1, 0, 2)
    Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

    if gethalf:
        Ptrain_tensor = Ptrain_tensor[:,:,:(Ptrain_tensor.shape[-1] // 2)]
        Pval_tensor = Pval_tensor[:,:,:(Pval_tensor.shape[-1] // 2)]
        Ptest_tensor = Ptest_tensor[:,:,:(Ptest_tensor.shape[-1] // 2)]

    Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
    Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
    Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

    train_chunk = PAMchunk(Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor, device = device)
    val_chunk = PAMchunk(Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor, device = device)
    test_chunk = PAMchunk(Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor, device = device)

    return train_chunk, val_chunk, test_chunk

class PAMchunk:
    '''
    Class to hold chunks of PAM data
    '''
    def __init__(self, train_tensor, static, time, y, device = None):
        self.X = train_tensor.to(device)
        self.static = None if static is None else static.to(device)
        self.time = time.to(device)
        self.y = y.to(device)

    def choose_random(self):
        n_samp = len(self.X)           
        idx = random.choice(np.arange(n_samp))
        
        static_idx = None if self.static is None else self.static[idx]
        print('In chunk', self.time.shape)
        return self.X[:,idx,:].unsqueeze(dim=1), \
            self.time[:,idx].unsqueeze(dim=-1), \
            self.y[idx].unsqueeze(dim=0), \
            static_idx

    def __getitem__(self, idx): 
        static_idx = None if self.static is None else self.static[idx]
        return self.X[:,idx,:].unsqueeze(dim=1), \
            self.time[:,idx].unsqueeze(dim=-1), \
            self.y[idx].unsqueeze(dim=0), \
            static_idx
    
class ECGchunk:
    '''
    Class to hold chunks of ECG data
    '''
    def __init__(self, train_tensor, static, time, y, device = None):
        self.X = train_tensor.to(device)
        self.static = None if static is None else static.to(device)
        self.time = time.to(device)
        self.y = y.to(device)

    def choose_random(self):
        n_samp = self.X.shape[1]           
        idx = random.choice(np.arange(n_samp))

        static_idx = None if self.static is None else self.static[idx]
        #print('In chunk', self.time.shape)
        return self.X[idx,:,:].unsqueeze(dim=1), \
            self.time[idx:].unsqueeze(dim=-1), \
            self.y[idx].unsqueeze(dim=0), \
            static_idx

    def get_all(self):
        static_idx = None # Doesn't support non-None 
        return self.X, self.time, self.y, static_idx

    def __getitem__(self, idx): 
        static_idx = None if self.static is None else self.static[idx]
        return self.X[:,idx,:], \
            self.time[:,idx], \
            self.y[idx].unsqueeze(dim=0)
            #static_idx





class DatasetwInds(torch.utils.data.Dataset):
    def __init__(self, X, times, y, gt_mask = None):
        self.X = X
        self.times = times
        self.y = y
        self.gt_mask = None if gt_mask is None else gt_mask.squeeze(-1)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx]
        T = self.times[idx]
        y = self.y[idx]
        if self.gt_mask is None:
            return {"x": x, "y": y}
        else:
            return {"x": x, "y": y, "gt_mask": self.gt_mask[idx]}
    

class MITECGDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, phase):
        path = f_dir / 'dataset' / 'realworld' / 'TimeX' / dataset

        split_path = 'split={}.pt'.format(1)

        idx_train, idx_val, idx_test = torch.load(os.path.join(path, split_path))
        if phase.lower() == "train":
            idx = idx_train
        elif phase.lower() == "valid":
            idx = idx_val
        elif phase.lower() == "test":
            idx = idx_test
        
        X = torch.load(os.path.join(path, 'all_data/X.pt'), weights_only=False)
        y = torch.load(os.path.join(path, 'all_data/y.pt'), weights_only=False).squeeze()
        times = torch.zeros(X.shape[0],X.shape[1])
        for i in range(X.shape[1]):
            times[:,i] = torch.arange(360)
    
        saliency = torch.load(os.path.join(path, 'all_data/saliency.pt'))
        
        self.X = X[:, idx, :].permute(1, 0, 2).float().clone() # N x T x D
        self.y = torch.LongTensor(LabelEncoder().fit_transform(y[idx].unsqueeze(-1).numpy())).clone()
        self.gt_mask = saliency[idx].float().clone()

        del X, y, times, saliency

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        gt_mask = self.gt_mask[idx]

        return {"x": x, "y": y, "gt_mask": gt_mask}


def getStats(P_tensor):
    N, T, F = P_tensor.shape
    if isinstance(P_tensor, np.ndarray):
        Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    else:
        Pf = P_tensor.permute(2, 0, 1).reshape(F, -1).detach().clone().cpu().numpy()
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        mf[f] = np.mean(vals_f)
        stdf[f] = np.std(vals_f)
        stdf[f] = np.where(stdf[f]>eps, stdf[f], eps)
    return mf, stdf

def tensorize_normalize_ECG(P, y, mf, stdf):
    F, T = P[0].shape

    P_time = np.zeros((len(P), T, 1))
    for i in range(len(P)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        P_time[i] = tim
    P_tensor = mask_normalize_ECG(P, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0

    y_tensor = y
    y_tensor = y_tensor.type(torch.LongTensor)
    return P_tensor, None, P_time, y_tensor


def mask_normalize_ECG(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    P_tensor = P_tensor.numpy()
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2,0,1)).reshape(F,-1)
    M = 1*(P_tensor>0) + 0*(P_tensor<=0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F,N,T)).transpose((1,2,0))
    #Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
    return Pnorm_tensor


def tensorize_normalize_other(P, y, mf, stdf):
    T, F = P[0].shape

    P_time = np.zeros((len(P), T, 1))
    for i in range(len(P)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        P_time[i] = tim
    P_tensor = mask_normalize(P, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0

    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, None, P_time, y_tensor

def mask_normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2,0,1)).reshape(F,-1)
    M = 1*(P_tensor>0) + 0*(P_tensor<=0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F,N,T)).transpose((1,2,0))
    Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
    return Pfinal_tensor



import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.signal import butter, lfilter, freqz
import pickle as pkl
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import matplotlib.pyplot as plt

import torch
from tqdm import trange, tqdm
import timesynth as ts

# def generate_spikes(T = 500, D = 15, n_spikes = 4, important_sensors = 2):

#     samp = np.zeros((T, D))
#     # Choose important sensors:
#     imp_sensors = np.random.choice(np.arange(D), size = (important_sensors,), replace = False)
#     #assert (n_spikes > important_sensors), 'n_spikes must be larger than important_sensors'

#     important_sensors = min(n_spikes, important_sensors)
    
#     spikes_left = n_spikes
#     count_imp_sensors = 0

#     spike_locations = []

#     for di in range(D):
#         noise = ts.noise.GaussianNoise(std=0.001)
#         x = ts.signals.NARMA(order=2,seed=random.seed())
#         x_ts = ts.TimeSeries(x, noise_generator=noise)
#         x_sample, signals, errors = x_ts.sample(np.array(range(T)))

#         if di in imp_sensors:
#             max_samp = max(x_sample)

#             if di == max(imp_sensors):
#                 spikes_to_take = spikes_left
#             else:
#                 spikes_to_take = max(np.random.randint(spikes_left),1)
#                 spikes_left -= spikes_to_take
        
#             # Choose random indices, not overlapping:
#             spike_inds = np.random.choice(np.arange(T), 
#                 size = (spikes_to_take,), replace = False)

#             for s in spike_inds:
#                 x_sample[s] = max_samp * 3
#                 spike_locations.append((s, di))
#                 #x_sample[s-2:s+3] = np.array([3, 4, 5, 4, 3]) * max_samp

#             count_imp_sensors += 1

#         samp[:,di] = x_sample

#     return samp, spike_locations

def generate_spikes(T = 500, D = 15, n_spikes = 4, important_sensors = 2):

    samp = np.zeros((T, D))
    # Choose important sensors:
    imp_sensors = np.random.choice(np.arange(D), size = (important_sensors,), replace = False)
    #assert (n_spikes > important_sensors), 'n_spikes must be larger than important_sensors'

    #important_sensors = min(n_spikes, important_sensors)
    

    spike_locations = []
    time_pts = np.random.choice(np.arange(T), size=(n_spikes,), replace = False)

    for di in range(D):
        noise = ts.noise.GaussianNoise(std=0.001)
        x = ts.signals.NARMA(order=2,seed=random.seed())
        x_ts = ts.TimeSeries(x, noise_generator=noise)
        x_sample, signals, errors = x_ts.sample(np.array(range(T)))

        if di in imp_sensors:
            max_samp = max(x_sample)

            x_sample[time_pts] = max_samp * 3

            for t in time_pts:
                spike_locations.append((t, di))

        samp[:,di] = x_sample

    return samp, spike_locations

def generate_spike_dataset(N = 1000, T = 500, D = 15, n_classes = 4, important_sensors = 2):

    # Get even number of samples for each class:
    class_count = [(N // n_classes)] * (n_classes - 1)
    class_count.append(N - sum(class_count))

    #print('Class count', class_count)

    gt_exps = []
    X = np.zeros((N, T, D))
    times = np.zeros((N,T))
    y = np.zeros(N)
    total_count = 0

    for i, n in enumerate(class_count):
        for _ in range(n):
            # n_spikes increases with count (add 1 because zero index)
            Xi, locs = generate_spikes(T, D, n_spikes = (i + 1), important_sensors = important_sensors)
            X[total_count,:,:] = Xi
            times[total_count,:] = np.arange(1,T+1) # Steadily increasing times
            y[total_count] = i # Needs to be zero-indexed
            gt_exps.append(locs)
            total_count += 1

    return X, times, y, gt_exps
    

class SpikeTrainDataset(torch.utils.data.Dataset):
    def __init__(self, N = 1000, T = 500, D = 15, n_classes = 4, important_sensors = 2):
        
        self.X, self.times, self.y, _ = generate_spike_dataset(
            N = N, T = T, D = D, n_classes = n_classes, important_sensors = important_sensors
        )

        self.X = torch.from_numpy(self.X).transpose(0,1)
        self.times = torch.from_numpy(self.times).transpose(0,1)
        self.y = torch.from_numpy(self.y).long()

    def __len__(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y 

def apply_gt_exp_to_matrix(X, gt_exp):
    Xgt = torch.zeros_like(X)
    for n in range(len(gt_exp)):
        for i, j in gt_exp[n]:
            Xgt[i,n,j] = 1
    return Xgt

def convert_torch(X, times, y):
    X = torch.from_numpy(X).transpose(0,1)
    times = torch.from_numpy(times).transpose(0,1)
    y = torch.from_numpy(y).long()

    return X, times, y

def get_all_spike_loaders(Ntrain = 1000, T = 500, D = 15, n_classes = 4, important_sensors = 2,
        Nval = 100, Ntest = 300):

    config_args = (T, D, n_classes, important_sensors)

    train_dataset = SpikeTrainDataset(Ntrain, *config_args)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    print('Train loaded')

    # Get validation tuple:
    Xval, timeval, yval, _ = generate_spike_dataset(Nval, *config_args)
    val_tuple = convert_torch(Xval, timeval, yval)
    print('Val loaded')

    # Get testing tuple:
    Xtest, timetest, ytest, gt_exps = generate_spike_dataset(Ntest, *config_args)
    test_tuple = convert_torch(Xtest, timetest, ytest)
    print('Test loaded')

    print_tuple(test_tuple)

    return train_dataset, val_tuple, test_tuple, apply_gt_exp_to_matrix(test_tuple[0], gt_exps)

def print_tuple(t):
    print('X', t[0].shape)
    print('time', t[1].shape)
    print('y', t[2].shape)
