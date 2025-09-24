from tint.attr import TimeForwardTunnel
from captum.attr import Lime
import torch.nn as nn
import torch

def explain(
    classifier,
    x_test,
    y_test,
):
    explainer = TimeForwardTunnel(Lime(classifier.predict))
    return explainer.attribute(
        x_test,
        target = y_test,
        show_progress=True,
    ).abs()
