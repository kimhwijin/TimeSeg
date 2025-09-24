import os
import torch as th
import torch

from datasets.dataset import DataModule, Dataset
import sys, os

file_dir = os.path.dirname(__file__)

class FreqShape(DataModule):
    def __init__(
        self,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "FreqShape",
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
        if not os.path.join(
            self.data_dir, "split={}.pt".format(self.fold + 1)
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
        
        if self.data == None:
            self.data = torch.load(os.path.join(self.data_dir, f"split={self.fold+1}.pt"), weights_only=False)

        x, y = self.data[split]
        if split == 'test':
            return {
                "x": x.float(),
                "y": y.long(),
                'true': self.true_saliency()
            }
        return {
            "x": x.float(),
            "y": y.long(),
        }
    def true_saliency(self) -> th.Tensor:
        return self.data['gt_exps'].bool()