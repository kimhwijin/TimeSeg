import sys; 
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

def explain(
    classifier,
    datamodule,
    data,
    timesteps,
    num_features,
    skip_train_timex,
    fold,
    model_type,
    test_loader,
    device,
):
    skip_training = skip_train_timex # consider this

    generator_path = Path("./model_ckpt/generator/") / data / f"{model_type}_split_{fold}"
    generator_path.mkdir(parents=True, exist_ok=True)
    explainer = FIT(
        classifier,
        device=device,
        datamodule=datamodule,
        data_name=data,
        feature_size=num_features,
        path=generator_path,
        cv=fold,
    )

    if skip_training:
        explainer.load_generators()
    else:
        explainer.train_generators(300)

    fit = []

    for batch in tqdm(test_loader):
        x_batch = batch[0].to(device)
        data_mask = batch[1].to(device)
        batch_size = x_batch.shape[0]
        timesteps = timesteps[:batch_size, :]
        
        attr_batch = explainer.attribute(x_batch)
        
        fit.append(attr_batch)

    return torch.Tensor(np.concatenate(fit, axis=0)) 


    
    

import copy
import numpy as np

class BaseExplainer:
    def train_generators(self, num_epochs) :
        gen_result = self.explainer.train_generators(
            self.train_loader, self.valid_loader, num_epochs
        )
        self.explainer.test_generators(self.test_loader)
        
        return None
    
    def load_generators(self):
        self.explainer.load_generators()
        self.explainer.test_generators(self.test_loader)


class FIT(BaseExplainer):
    def __init__(
        self,
        model,
        device,
        datamodule,
        data_name,
        feature_size,
        path,
        cv
    ):
        from winit.explainer.fitexplainers import FITExplainer
        self.explainer = FITExplainer(
            device,
            feature_size,
            data_name,
            path,
        )
        
        self.explainer.set_model(model, False)
        
        self.datamodule = copy.deepcopy(datamodule)
        self.datamodule.setup()
        self.datamodule.batch_size = 100
        
        self.train_loader = self.datamodule.train_dataloader()
        self.valid_loader = self.datamodule.val_dataloader()
        self.test_loader = self.datamodule.test_dataloader()
        
    def attribute(self, x):
        return self.explainer.attribute(x)