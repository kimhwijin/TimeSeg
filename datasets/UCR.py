import torch
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
import os

file_dir = os.path.dirname(__file__)

import torch as th
from datasets.dataset import DataModule, Dataset

def UCR_config(dataset):
    dataset = "UCR_" + dataset

    if dataset == 'UCR_Yoga':
        seq_len     = 426
        num_class   = 2
    if dataset == 'UCR_Trace':
        seq_len     = 275
        num_class   = 4
    if dataset == 'UCR_ECG200':
        seq_len     = 96
        num_class   = 2
    if dataset == 'UCR_UMD':
        seq_len     = 150
        num_class   = 3
    if dataset == 'UCR_GunPointOldVersusYoung':
        seq_len     = 150
        num_class   = 2
    if dataset == 'UCR_GunPointMaleVersusFemale':
        seq_len     = 150
        num_class   = 2
    if dataset == 'UCR_GunPoint':
        seq_len     = 150
        num_class   = 2
    if dataset == 'UCR_ArrowHead':
        seq_len     = 251
        num_class   = 3
    if dataset == 'UCR_ShapeletSim':
        seq_len     = 500
        num_class   = 2
    if dataset == 'UCR_Coffee':
        seq_len     = 286
        num_class   = 2
    if dataset == 'UCR_Strawberry':
        seq_len     = 235
        num_class   = 2
    if dataset == 'UCR_DodgerLoopGame':
        seq_len     = 288
        num_class   = 2
    if dataset == "UCR_HandOutlines":
        seq_len     = 2709
        num_class   = 2

    if dataset == 'UCR_Chinatown':
        seq_len     = 24
        num_class   = 2
    if dataset == 'UCR_FreezerRegularTrain':
        seq_len     = 301
        num_class   = 2
        
    if dataset == 'UCR_FreezerSmallTrain':
        seq_len     = 301
        num_class   = 2
    if dataset == 'UCR_HouseTwenty':
        seq_len     = 2000
        num_class   = 2
    if dataset == 'UCR_WormsTwoClass':
        seq_len     = 900
        num_class   = 2
    if dataset == 'UCR_Wafer':
        seq_len     = 152
        num_class   = 2

    if dataset == 'UCR_TwoPatterns':
        seq_len     = 128
        num_class   = 4

    if dataset == 'UCR_ElectricDevices':
        seq_len     = 96
        num_class   = 7
    if dataset == 'UCR_ECG5000':
        seq_len     = 140
        num_class   = 5
    if dataset == 'UCR_FaceAll':
        seq_len     = 131
        num_class   = 14
    if dataset == 'UCR_FaceFour':
        seq_len     = 350
        num_class   = 4
    if dataset == 'UCR_BeetleFly':
        seq_len     = 512
        num_class   = 2
        
    return seq_len, num_class

class UCR(DataModule):
    def __init__(
        self,
        data,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "UCR"
        ),
        batch_size: int = 32,
        prop_val: float = 0.1,
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
        self.data_dir = os.path.join(data_dir, data)
        self.data = data
        self._mean = None
        self._std = None

    def prepare_data(self):
        if not os.path.exists(
            os.path.join(self.data_dir, f"{self.data}_TRAIN.tsv")
        ) or not os.path.exists(
            os.path.join(self.data_dir, "{}_TEST.tsv".format(self.data))
        ):
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
        if split == 'val':
            split = 'test'

        df    = pd.read_csv(
                    os.path.join(self.data_dir, f"{self.data}_{split.upper()}.tsv"),
                    sep='\t', 
                    header=None
                )
        x       = df[df.columns[1:]].values
        x       = KNNImputer(n_neighbors=5, weights="distance").fit_transform(x)

        x = torch.FloatTensor(x).unsqueeze(-1)
        y = torch.LongTensor(
            LabelEncoder().fit_transform(df[0].values)
        )

        if split == "train":
            self._mean = x.mean(dim=(0, 1), keepdim=True)
            self._std = x.std(dim=(0, 1), keepdim=True)
        
        EPS = 1e-5
        x = (x - self._mean) / (self._std + EPS)
        return {
            "x": x.float(),
            "y": y.long(),
            "mask": th.ones_like(x)
        }










