from captum.attr import GradientShap
import torch as th

def explain(
    classifier,
    test_loader,
    timesteps,
    device
):
    explainer = GradientShap(classifier.predict)

    gradientshap = []

    # Iterate over the DataLoader to process data in batches
    for batch in test_loader:
        x_batch = batch[0].to(device)  # Move batch to the appropriate device if necessary
        data_mask = batch[1].to(device)
        batch_size = x_batch.shape[0]
        timesteps = timesteps[:batch_size, :]
        
        from captum._utils.common import _run_forward
        with th.autograd.set_grad_enabled(False):
            partial_targets = _run_forward(
                classifier,
                x_batch,
                additional_forward_args=(data_mask, timesteps, False),
            )
        partial_targets = th.argmax(partial_targets, -1)

        
        attr_batch = explainer.attribute(
                x_batch,
                baselines=(th.cat([x_batch * 0, x_batch])),
                target=partial_targets,
                n_samples=50,
                stdevs=0.0001,
                additional_forward_args=(data_mask, timesteps, False),
            ).abs()
        
        
        gradientshap.append(attr_batch.cpu())  # Move to CPU if necessary
    return th.cat(gradientshap, dim=0)