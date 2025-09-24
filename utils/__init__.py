import sys
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)


from utils.utils import safe_nb, EarlyStopping, nb_transform_fn

def get_exp_name(
    loss,
    backbone,
    weights,
    seg_dist,
    dataset,
    target_type, 
    predictor_type, 
    predictor_pretrain,
    mask_type
):     
    exp_name = []

    if predictor_type == 'predictor':
        exp_name.append(f"Pr:pred:{predictor_pretrain}")
    elif predictor_type == 'blackbox':
        exp_name.append('Pr:blackbox')
    
    exp_name.append(f"Da:{dataset}")
    exp_name.append(f"Lo:{loss}")
    exp_name.append(f"Bc:{backbone}")
    exp_name.append(f"Sd:{seg_dist}")
    exp_name.append(f"We:" + ",".join(list(map(str, weights))))
    exp_name.append(f"Tg:{target_type}")
    exp_name.append(f"Ms:{mask_type}")
    exp_name = '_'.join(exp_name)
    print(exp_name)
    return exp_name