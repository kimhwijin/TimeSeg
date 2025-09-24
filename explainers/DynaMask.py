from pytorch_lightning import Trainer
import numpy as np

from tint.attr import DynaMask
from tint.attr.models import MaskNet

def explain(
    classifier,
    accelerator,
    device_id,
    x_test,
    data_mask,
    timesteps,
    device,
):
    trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            # deterministic=True,
        )
    mask = MaskNet(
        forward_func=classifier.predict,
        perturbation="fade_moving_average",
        keep_ratio=list(np.arange(0.1, 0.7, 0.1)),
        deletion_mode=True,
        size_reg_factor_init=0.1,
        size_reg_factor_dilation=10000,
        time_reg_factor=0.0,
        loss="cross_entropy",
    )
    explainer = DynaMask(classifier.predict)
    _attr = explainer.attribute(
        x_test,
        trainer=trainer,
        mask_net=mask,
        additional_forward_args=(data_mask, timesteps, False),
        batch_size=32,
        return_best_ratio=True,
    )
    print(f"Best keep ratio is {_attr[1]}")
    return _attr[0].to(device)
