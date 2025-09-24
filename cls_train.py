import os
import torch
from argparse import ArgumentParser
import random
import torch as th
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import  CSVLogger

from models.classifier import get_classifier
from datasets import get_datamodule

def main(
    data,
    model_type,
    fold,
    seed,
    epoch,
):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    accelerator = device.split(":")[0]
    device_id = 1

    datamodule, num_features, max_len, num_classes = get_datamodule(data=data, fold=fold, seed=seed, model_type=model_type, batch_size=64)
    classifier = get_classifier(num_features, num_classes, max_len, model_type, data)

    if not os.path.exists("./model_ckpt/{}/".format(data)):
        os.makedirs("./model_ckpt/{}/".format(data))

    if os.path.exists(f"./model_ckpt/{data}/{model_type}_classifier_{fold}_{seed}.ckpt"):
        print(f"Skip ./model_ckpt/{data}/{model_type}_classifier_{fold}_{seed}.ckpt")
        return

    ckpt = ModelCheckpoint(
        dirpath="./model_ckpt/{}".format(data),
        filename="{}_classifier_{}_{}".format(model_type, fold, seed),
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=25)


    logger = CSVLogger(save_dir=f"./model_ckpt/{data}", name=f"{model_type}_classifier_{fold}_{seed}")

    trainer = Trainer(
        max_epochs=epoch,
        accelerator=accelerator,
        devices=device_id,
        callbacks=[ckpt, early_stop],
        logger=logger,
    )
    trainer.fit(classifier, datamodule=datamodule)
    results = trainer.test(ckpt_path="best", datamodule=datamodule)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data",type=str, default="MITECG")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_type", type=str, default="tcn")
    parser.add_argument("--epoch", type=int, default=100)
    return parser.parse_args()


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
    args = parse_args()
    set_seed(args.seed)

    main(
        data=args.data,
        model_type=args.model_type,
        fold=args.fold,
        seed=args.seed,
        epoch=args.epoch,
    )
