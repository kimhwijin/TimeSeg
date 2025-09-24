import os
import numpy as np

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import sys; 
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)

from models.layers import StateClassifier, TCNClassifier

from tint.models import Net
from torchmetrics import Accuracy, Precision, Recall, AUROC, F1Score, AveragePrecision
from typing import Callable, Union


def get_classifier(
    num_features,
    num_classes,
    max_len,
    model_type,
    data,
):  
    classifier = ClassifierNet(
        feature_size=num_features,
        n_state=num_classes,
        n_timesteps=max_len,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
        model_type=model_type,
        model_params={}
    )   
    return classifier


class ClassifierNet(Net):
    def __init__(
        self,
        feature_size: int,
        n_state: int,
        n_timesteps: int,
        hidden_size: int,
        rnn: str = "GRU",
        dropout: float = 0.5,
        regres: bool = True,
        bidirectional: bool = False,
        loss: Union[str, Callable] = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
        model_type: str = "state",
        model_params = {},
        pam: bool = False,
    ):
        
        if model_type == "state":
            classifier = StateClassifier(
                feature_size=feature_size,
                n_state=n_state,
                hidden_size=hidden_size,
                rnn=rnn,
                dropout=dropout,
                regres=regres,
                bidirectional=bidirectional,
            )
        elif model_type == "tcn":
            classifier = TCNClassifier(
                in_channels=feature_size,
                num_classes=n_state, 
                **model_params
            )
            
        super().__init__(
            layers=classifier,
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )
        self.save_hyperparameters()

        for stage in ["train", "val", "test"]:
            setattr(self, stage + "_acc", Accuracy(task="multiclass", num_classes=n_state, average='macro'))
            setattr(self, stage + "_pre", Precision(task="multiclass", num_classes=n_state, average='macro'))
            setattr(self, stage + "_rec", Recall(task="multiclass", num_classes=n_state, average='macro'))
            setattr(self, stage + "_auroc", AUROC(task="multiclass", num_classes=n_state, average='macro'))
            setattr(self, stage + "_auprc", AveragePrecision(task="multiclass", num_classes=n_state))
            setattr(self, stage + "_f1", F1Score(task="multiclass", num_classes=n_state))


    def forward(self, *args, **kwargs) -> th.Tensor:
        return self.net(*args, **kwargs)
    

    def step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x)
        
        loss = self.loss(y_hat, y)

        for metric in ["acc", "pre", "rec", "auroc", "auprc", "f1"]:
            getattr(self, stage + "_" + metric)(y_hat, y.long())
            self.log(stage + "_" + metric, getattr(self, stage + "_" + metric))
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x.float())
    
    def predict(self, *args, **kwargs) -> th.Tensor:
        return self.net(*args, **kwargs).softmax(-1)
