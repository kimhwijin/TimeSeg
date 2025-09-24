from captum.attr import IntegratedGradients
import torch
def explain(
    classifier,
    test_loader,
    timesteps,
    num_classes,
    device,
):
     
    explainer = IntegratedGradients(classifier.predict)
    
    integrated_gradients = []

    for batch in test_loader:
        x_batch = batch[0].to(device)
        data_mask = batch[1].to(device)
        batch_size = x_batch.shape[0]
        timesteps = timesteps[:batch_size, :]
        
        from captum._utils.common import _run_forward
        with torch.autograd.set_grad_enabled(False):
            partial_targets = _run_forward(
                classifier,
                x_batch,
                # additional_forward_args=(data_mask, timesteps, False),
            )
        partial_targets = torch.argmax(partial_targets, -1)

        attr_batch = explainer.attribute(
            x_batch,
            baselines=x_batch * 0,
            target=partial_targets,
            # additional_forward_args=(data_mask, timesteps, False),
            # task="binary" if num_classes <= 2 else 'multiclass',
            # temporal_additional_forward_args=temporal_additional_forward_args,
        ).abs()
    
        integrated_gradients.append(attr_batch.cpu())
    
    return torch.cat(integrated_gradients, dim=0)