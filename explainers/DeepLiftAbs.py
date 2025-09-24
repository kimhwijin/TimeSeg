from captum.attr import DeepLift
import torch as th

def explain(
    classifier,
    test_loader,
    timesteps,
    device,
):

    explainer = DeepLift(classifier) # change forward function to self.net(*args, **kwargs).softmax(-1)

    deeplift = []

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
            baselines=x_batch * 0,
            target=partial_targets,
            additional_forward_args=(data_mask, timesteps, False),
        ).abs()
        
        deeplift.append(attr_batch.cpu())
    
    return th.cat(deeplift, dim=0)