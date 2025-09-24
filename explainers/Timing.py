import torch
from captum.attr import (
    FeaturePermutation,
    GradientShap,
    IntegratedGradients,
    Occlusion,
    ShapleyValueSampling,
)
from torch.distributions import Beta
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import itertools
import numpy as np
import torch
import torch.nn as nn
import random

from tqdm import tqdm

def explain(
    classifier,
    timesteps,
    num_segments,
    min_seg_len,
    max_seg_len,
    test_loader,
    device,
):
    explainer = OUR(classifier.predict)

    our_results = []

    for batch in tqdm(test_loader):
        x_batch = batch[0].to(device)
        data_mask = batch[1].to(device)
        batch_size = x_batch.shape[0]
        timesteps = timesteps[:batch_size, :]

        from captum._utils.common import _run_forward
        
        
        attr_batch = torch.zeros_like(x_batch)
        
        with torch.autograd.set_grad_enabled(False):
            partial_targets = _run_forward(
                classifier,
                x_batch,
                additional_forward_args=(data_mask, timesteps, False),
            )
        partial_targets = torch.argmax(partial_targets, -1)
        B, T, D = x_batch.shape
        
        all_time_mask = torch.zeros(50, B, T, D).to(x_batch.device)
        for i in range(10):
            
            dims = torch.randint(0, D, (B, num_segments), device=device)
            seg_lens = torch.randint(min_seg_len, max_seg_len+1, (B, num_segments), device=device)

            t_starts = (torch.rand(B, num_segments, device=device) * (T - seg_lens)).long()
            time_mask = torch.ones_like(x_batch)
            batch_indices = torch.arange(B, device=device)

            for s in range(num_segments):

                max_len = seg_lens[:,s].max()

                base_range = torch.arange(max_len, device=device)
                base_range = base_range.unsqueeze(0)
                
                indices = t_starts[:,s].unsqueeze(-1) + base_range

                end_points = t_starts[:,s] + seg_lens[:,s]  # shape [B]
                end_points = end_points.unsqueeze(-1)           # shape [B, 1]

                valid_indices = (indices < end_points) & (indices < T)
                time_mask[batch_indices.view(1,-1,1), indices * valid_indices, dims[:,s].unsqueeze(-1)] = 0

            attr_batch += explainer.attribute_orig(
                x_batch,
                baselines=x_batch * 0,
                targets=partial_targets,
                additional_forward_args=(data_mask, timesteps, False),
                n_samples=50,
                num_segments=num_segments,
                min_seg_len=min_seg_len,
                max_seg_len=max_seg_len,
                time_mask=time_mask.unsqueeze(0).repeat(50, 1, 1, 1),
            ).abs()
            
            all_time_mask[i] = time_mask
        attr_batch = attr_batch /all_time_mask.sum(dim=0)

        our_results.append(attr_batch.detach().cpu())
    return torch.cat(our_results, dim=0)

class OUR:
    def __init__(self, model):
        self.model = model

    def attribute_random(
        self, inputs, baselines, targets, additional_forward_args, n_samples=50, prob=0.3,
    ):
        """
        inputs:  [B, T, D]
        baselines:  [B, T, D] (same shape as inputs)
        targets:  [B] (integer class targets)
        additional_forward_args: unused except . . . [2] for 'return_all'
        n_samples: number of interpolation steps
        """
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        # -------------------------------------------------
        # 1) Build interpolation from baseline --> inputs
        # -------------------------------------------------
        alphas = torch.linspace(0, 1 - 1 / n_samples, n_samples, device=inputs.device)
        alphas = alphas.view(-1, 1, 1, 1)  # shape: [n_samples, 1, 1, 1]

        # Start from "start_pos" so that alpha=0 means "baselines"
        start_pos = baselines

        # Expand to shape [n_samples, B, T, D]
        expanded_inputs = inputs.unsqueeze(0)  # [1, B, T, D]
        expanded_start = start_pos.unsqueeze(0)  # [1, B, T, D]

        # Interpolate
        interpolated_inputs = expanded_start + alphas * (expanded_inputs - expanded_start)

        # Example: 50% chance to fix each [t, d]
        fix_probability = prob  # tweak as needed
        rand_mask = torch.rand_like(interpolated_inputs)  # shape [n_samples, B, T, D]
        # Convert to {0,1} by comparing to fix_probability
        # 1 = keep interpolation, 0 = fix to actual input
        time_mask = (rand_mask > fix_probability).float()

        # Detach actual inputs so no gradient is assigned to them
        fixed_inputs = inputs.unsqueeze(0).detach()  # shape [1, B, T, D]
        # broadcast to match [n_samples, B, T, D]
        # The random mask has the same shape as `interpolated_inputs`
        # => we combine them:
        interpolated_inputs = time_mask * interpolated_inputs + (1 - time_mask) * fixed_inputs

        # Turn on gradient for only the interpolation portion
        interpolated_inputs.requires_grad = True

        # -------------------------------------------------
        # 3) Forward pass & gather target predictions
        # -------------------------------------------------
        predictions = self.model(
            interpolated_inputs.view(-1, inputs.shape[1], inputs.shape[2]),
            mask=None,
            timesteps=None,
            return_all=additional_forward_args[2],
        )

        # Make sure predictions has shape [n_samples, B, num_classes]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)

        predictions = predictions.view(n_samples, inputs.shape[0], -1)

        # Gather the logit of the correct class for each sample
        gathered = predictions.gather(
            dim=2,
            index=targets.unsqueeze(0)
            .unsqueeze(-1)
            .expand(n_samples, inputs.shape[0], 1),
        ).squeeze(
            -1
        )  # shape [n_samples, B]

        # Sum across all n_samples and batch for gradient
        total_for_target = gathered.sum()

        # -------------------------------------------------
        # 4) Compute gradients wrt `interpolated_inputs`
        # -------------------------------------------------
        grad = torch.autograd.grad(
            outputs=total_for_target,
            inputs=interpolated_inputs,
            retain_graph=True,
            allow_unused=True,
        )[
            0
        ]  # shape: [n_samples, B, T, D]
        grad[time_mask == 0] = 0

        grads = grad.sum(dim=0)  # Proper Riemann sum
        final_attr = grads * (inputs - baselines) / time_mask.sum(dim=0)

        return final_attr

    def attribute_random_time_segments_one_dim_same_for_batch(
        self,
        inputs: torch.Tensor,  # [B, T, D]
        baselines: torch.Tensor,  # [B, T, D]
        targets: torch.Tensor,  # [B]
        additional_forward_args,
        n_samples: int = 50,
        num_segments: int = 3,  # how many time segments (one dimension each) to fix per sample
        max_seg_len: int = None,  # optional maximum length for each time segment
        min_seg_len: int = None,
    ):
        """
        Generates random contiguous time segments (each segment picks ONE random dimension).
        BUT crucially, each sample i uses the SAME random segments for the *entire batch*.

        Steps:
        1) Interpolate from baselines -> inputs using n_samples alpha steps
        2) For each sample i (i.e. alpha step), create `num_segments` random slices
            - each slice picks a single dimension, plus time range [t_start : t_end)
            - fix that dimension/time range for ALL batch items
        3) Forward pass & gather target logit => sum => compute gradients
        4) Multiply by (inputs - baselines), optionally scale by how often (t,d) was free
        """
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        B, T, D = inputs.shape
        device = inputs.device

        data_mask = additional_forward_args[0]

        # -------------------------------------------------------
        # 1) Build interpolation from baseline -> inputs
        # -------------------------------------------------------
        alphas = torch.linspace(0, 1 - 1 / n_samples, n_samples, device=device).view(-1, 1, 1, 1)
        
        expanded_inputs = inputs.unsqueeze(0)
        expanded_baselines = baselines.unsqueeze(0)
        # Interpolate with batch-specific alphas
        interpolated_inputs = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)
        
        if max_seg_len is None:
            max_seg_len = T

        if min_seg_len is None:
            min_seg_len = 1

        # Generate batch-specific masks
        dims = torch.randint(0, D, (n_samples, B, num_segments), device=device)
        seg_lens = torch.randint(min_seg_len, max_seg_len+1, (n_samples, B, num_segments), device=device)
        
        # t_starts = torch.randint(0, T-max_seg_len+1, (n_samples, B, num_segments), device=device)
        t_starts = (torch.rand(n_samples, B, num_segments, device=device) * (T - seg_lens)).long()

        # Initialize mask
        time_mask = torch.ones_like(interpolated_inputs)

        # Create indices tensor
        batch_indices = torch.arange(B, device=device)
        sample_indices = torch.arange(n_samples, device=device)

        # Create mask via scatter
        for s in range(num_segments):
            # indices = t_starts[:,:,s].unsqueeze(-1) + torch.arange(seg_lens[:,:,s].max(), device=device).unsqueeze(0).unsqueeze(0)
            # valid_indices = indices < T
            # print(seg_lens)
            max_len = seg_lens[:,:,s].max()
            # print(max_len)
            # 2) base_range = [0, 1, 2, ..., max_len-1], shape [max_len]
            base_range = torch.arange(max_len, device=device)
            base_range = base_range.unsqueeze(0).unsqueeze(0)
            
            indices = t_starts[:,:,s].unsqueeze(-1) + base_range

            end_points = t_starts[:,:,s] + seg_lens[:,:,s]  # shape [n_samples, B]
            end_points = end_points.unsqueeze(-1)           # shape [n_samples, B, 1]

            valid_indices = (indices < end_points) & (indices < T)
            time_mask[sample_indices.view(-1,1,1), batch_indices.view(1,-1,1), indices * valid_indices, dims[:,:,s].unsqueeze(-1)] = 0

        # Combine masked inputs
        fixed_inputs = expanded_inputs.detach()
        masked_inputs = time_mask * interpolated_inputs + (1 - time_mask) * fixed_inputs
        masked_inputs.requires_grad = True

        # -------------------------------------------------------
        # 3) Forward pass & gather target logits
        # -------------------------------------------------------
        predictions = self.model(
            masked_inputs.view(-1, T, D),
            mask=None,
            timesteps=None,
            return_all=additional_forward_args[2],
        )
        # Ensure shape => [n_samples, B, num_classes]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)
        predictions = predictions.view(n_samples, B, -1)

        # Gather only the target logit for each example
        gathered = predictions.gather(
            dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
        ).squeeze(-1)

        total_for_target = gathered.sum()
        
        grad = torch.autograd.grad(outputs=total_for_target, inputs=masked_inputs, retain_graph=True)[0]
        grad[time_mask == 0] = 0

        grads = grad.sum(dim=0)  # Proper Riemann sum
        final_attr = grads * (inputs - baselines) / (time_mask.sum(dim=0) + torch.finfo(torch.float16).eps)
            
        return final_attr
    
    def attribute_orig(
        self,
        inputs: torch.Tensor,  # [B, T, D]
        baselines: torch.Tensor,  # [B, T, D]
        targets: torch.Tensor,  # [B]
        additional_forward_args,
        n_samples: int = 50,
        num_segments: int = 3,  # how many time segments (one dimension each) to fix per sample
        max_seg_len: int = None,  # optional maximum length for each time segment
        min_seg_len: int = None,
        time_mask: torch.Tensor = None,
    ):
        """
        Generates random contiguous time segments (each segment picks ONE random dimension).
        BUT crucially, each sample i uses the SAME random segments for the *entire batch*.

        Steps:
        1) Interpolate from baselines -> inputs using n_samples alpha steps
        2) For each sample i (i.e. alpha step), create `num_segments` random slices
            - each slice picks a single dimension, plus time range [t_start : t_end)
            - fix that dimension/time range for ALL batch items
        3) Forward pass & gather target logit => sum => compute gradients
        4) Multiply by (inputs - baselines), optionally scale by how often (t,d) was free
        """
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        B, T, D = inputs.shape
        device = inputs.device

        data_mask = additional_forward_args[0]

        # -------------------------------------------------------
        # 1) Build interpolation from baseline -> inputs
        # -------------------------------------------------------
        alphas = torch.linspace(0, 1 - 1 / n_samples, n_samples, device=device).view(-1, 1, 1, 1)
        
        expanded_inputs = inputs.unsqueeze(0)
        expanded_baselines = baselines.unsqueeze(0)
        # Interpolate with batch-specific alphas
        interpolated_inputs = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)
        
        if max_seg_len is None:
            max_seg_len = T

        if min_seg_len is None:
            min_seg_len = 1

        # Combine masked inputs
        fixed_inputs = expanded_inputs.detach()
        masked_inputs = time_mask * interpolated_inputs + (1 - time_mask) * fixed_inputs
        masked_inputs.requires_grad = True

        # -------------------------------------------------------
        # 3) Forward pass & gather target logits
        # -------------------------------------------------------
        predictions = self.model(
            masked_inputs.view(-1, T, D),
            mask=None,
            timesteps=None,
            return_all=additional_forward_args[2],
        )
        # Ensure shape => [n_samples, B, num_classes]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)
        predictions = predictions.view(n_samples, B, -1)

        # Gather only the target logit for each example
        gathered = predictions.gather(
            dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
        ).squeeze(-1)

        total_for_target = gathered.sum()
        
        grad = torch.autograd.grad(outputs=total_for_target, inputs=masked_inputs, retain_graph=True)[0]
        grad[time_mask == 0] = 0

        grads = grad.sum(dim=0)  # Proper Riemann sum
        final_attr = grads * (inputs - baselines)
            
        return final_attr
    
    def attribute_random_synthetic(
        self,
        inputs: torch.Tensor,  # [B, T, D]
        baselines: torch.Tensor,  # [B, T, D]
        targets: torch.Tensor,  # [B]
        additional_forward_args,
        n_samples: int = 50,
        num_segments: int = 3,  # how many time segments (one dimension each) to fix per sample
        max_seg_len: int = None,  # optional maximum length for each time segment
        min_seg_len: int = None,
    ):
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        B, T, D = inputs.shape
        device = inputs.device

        data_mask = additional_forward_args[0]

        # -------------------------------------------------------
        # 1) Build interpolation from baseline -> inputs
        # -------------------------------------------------------
        alphas = torch.linspace(0, 1 - 1 / n_samples, n_samples, device=device).view(-1, 1, 1, 1)
        
        expanded_inputs = inputs.unsqueeze(0)
        expanded_baselines = baselines.unsqueeze(0)
        # Interpolate with batch-specific alphas
        interpolated_inputs = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)
        
        if max_seg_len is None:
            max_seg_len = T
            
        max_seg_len = min(T, max_seg_len)

        if min_seg_len is None:
            min_seg_len = 1

        # Generate batch-specific masks
        dims = torch.randint(0, D, (n_samples, B, num_segments), device=device)
        seg_lens = torch.randint(min_seg_len, max_seg_len+1, (n_samples, B, num_segments), device=device)
        
        # t_starts = torch.randint(0, T-max_seg_len+1, (n_samples, B, num_segments), device=device)
        t_starts = (torch.rand(n_samples, B, num_segments, device=device) * (T - seg_lens)).long()

        # Initialize mask
        time_mask = torch.ones_like(interpolated_inputs)

        # Create indices tensor
        batch_indices = torch.arange(B, device=device)
        sample_indices = torch.arange(n_samples, device=device)

        # Create mask via scatter
        for s in range(num_segments):
            # indices = t_starts[:,:,s].unsqueeze(-1) + torch.arange(seg_lens[:,:,s].max(), device=device).unsqueeze(0).unsqueeze(0)
            # valid_indices = indices < T
            # print(seg_lens)
            max_len = seg_lens[:,:,s].max()
            # print(max_len)
            # 2) base_range = [0, 1, 2, ..., max_len-1], shape [max_len]
            base_range = torch.arange(max_len, device=device)
            base_range = base_range.unsqueeze(0).unsqueeze(0)
            
            indices = t_starts[:,:,s].unsqueeze(-1) + base_range

            end_points = t_starts[:,:,s] + seg_lens[:,:,s]  # shape [n_samples, B]
            end_points = end_points.unsqueeze(-1)           # shape [n_samples, B, 1]

            valid_indices = (indices < end_points) & (indices < T)
            time_mask[sample_indices.view(-1,1,1), batch_indices.view(1,-1,1), indices * valid_indices, dims[:,:,s].unsqueeze(-1)] = 0

        # Combine masked inputs
        fixed_inputs = expanded_inputs.detach()
        masked_inputs = time_mask * interpolated_inputs + (1 - time_mask) * fixed_inputs
        masked_inputs.requires_grad = True

        # -------------------------------------------------------
        # 3) Forward pass & gather target logits
        # -------------------------------------------------------
        predictions = self.model(
            masked_inputs.view(-1, T, D),
            mask=None,
            timesteps=None,
            return_all=additional_forward_args[2],
        )
        # Ensure shape => [n_samples, B, num_classes]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)
        predictions = predictions.view(n_samples, B, -1)

        # Gather only the target logit for each example
        gathered = predictions.gather(
            dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
        ).squeeze(-1)
        
        total_for_target = gathered.sum()
        
        grad = torch.autograd.grad(outputs=total_for_target, inputs=masked_inputs, retain_graph=True)[0]
        grad[time_mask == 0] = 0

        grads = grad.sum(dim=0)  # Proper Riemann sum
        final_attr = grads * (inputs - baselines) / (time_mask.sum(dim=0) + torch.finfo(torch.float16).eps)
            
        return final_attr
