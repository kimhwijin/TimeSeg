
import copy
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm

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
    explainer = WinIT(
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
    
    winit = []

    for batch in tqdm(test_loader):
        x_batch = batch[0].to(device)
        data_mask = batch[1].to(device)
        batch_size = x_batch.shape[0]
        timesteps = timesteps[:batch_size, :]
        
        attr_batch = explainer.attribute(x_batch)
        
        winit.append(attr_batch)
    
    return torch.Tensor(np.concatenate(winit, axis=0)) 


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
        
class WinIT(BaseExplainer):
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
        self.datamodule = copy.deepcopy(datamodule)
        self.datamodule.setup()
        self.datamodule.batch_size = 100
        
        self.train_loader = self.datamodule.train_dataloader()
        self.valid_loader = self.datamodule.val_dataloader()
        self.test_loader = self.datamodule.test_dataloader()
        
        from winit.explainer.winitexplainers import WinITExplainer
        self.explainer = WinITExplainer(
            device,
            feature_size,
            data_name,
            path,
            self.train_loader,
            random_state=42
        )
        
        self.explainer.set_model(model, True)
        
    def attribute(self, x):
        scores = self.explainer.attribute(x)
        num_samples, num_features, num_times, window_size = scores.shape

        aggregated_scores = np.zeros((num_samples, num_features, num_times))
        for t in range(num_times):
            relevant_windows = np.arange(t, min(t + window_size, num_times))
            relevant_obs = -relevant_windows + t - 1
            relevant_scores = scores[:, :, relevant_windows, relevant_obs]
            relevant_scores = np.nan_to_num(relevant_scores)

            aggregated_scores[:, :, t] = relevant_scores.mean(axis=-1)
        # scores# (bs, fts, ts, window_size)
        return aggregated_scores.reshape(-1, num_times, num_features)

    
    
