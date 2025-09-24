from tint.models import MLP, RNN
from pytorch_lightning import Trainer

from tint.attr.models import ExtremalMaskNet
from tint.attr import ExtremalMask
import torch.nn as nn

def explain(
    classifier,
    accelerator,
    device_id,
    x_test,
    data_mask,
    timesteps,
    device,
    lambda_1,
    lambda_2,
    mask_lr,
):
    
    trainer = Trainer(
        max_epochs=500,
        accelerator=accelerator,
        devices=device_id,
    )
    mask = ExtremalMaskNet(
        forward_func=classifier.predict,
        model=nn.Sequential(
            RNN(
                input_size=x_test.shape[-1],
                rnn="gru",
                hidden_size=x_test.shape[-1],
                bidirectional=True,
            ),
            MLP([2 * x_test.shape[-1], x_test.shape[-1]]),
        ),
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        loss="cross_entropy",
        optim="adam",
        lr=mask_lr,
    )
    explainer = ExtremalMask(classifier.predict)
    _attr = explainer.attribute(
        x_test,
        additional_forward_args=(data_mask, timesteps, False),
        trainer=trainer,
        mask_net=mask,
        batch_size=100,
    )
    return _attr.to(device)