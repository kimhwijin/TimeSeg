from tint.attr import TemporalAugmentedOcclusion, TimeForwardTunnel

def explain(
    classifier,
    x_train,
    data_mask,
    x_test,
    y_test,
    temporal_additional_forward_args,
):
    
    explainer = TimeForwardTunnel(
        TemporalAugmentedOcclusion(
            classifier.predict, data=x_train, n_sampling=10, is_temporal=True
        )
    )
    return explainer.attribute(
        x_test,
        sliding_window_shapes=(1,),
        target=y_test,
        attributions_fn=abs,
        additional_forward_args=(data_mask, None, False),
        temporal_additional_forward_args=temporal_additional_forward_args,
        show_progress=True,
    ).abs()