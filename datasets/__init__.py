import sys
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)

import os

from datasets.epilepsy import Epilepsy
from datasets.UCR import UCR, UCR_config
from datasets.MITECG import MITECG
from datasets.SeqCombSingle import SeqCombSingle
from datasets.FreqShape import FreqShape
from datasets.FreqShapeVar import FreqShapeVar
from datasets.LowVarDetectSingle import LowVarDetectSingle

def get_datamodule(data, fold, seed, batch_size, model_type=None):
    if data == 'MITECG':
        datamodule = MITECG(fold=fold, seed=seed, batch_size=batch_size)
        num_features = 1
        num_classes = 2
        max_len = 360

    elif data == 'epilepsy':
        datamodule = Epilepsy(fold=fold, seed=seed, batch_size=batch_size)

        num_features = 1
        num_classes = 2
        max_len = 178
    
    elif data in os.listdir(os.path.join(p, "data", "UCR")):
        datamodule = UCR(data=data, fold=fold, seed=seed, batch_size=batch_size)
        num_features = 1
        max_len, num_classes = UCR_config(data)

    elif data == 'SeqCombSingle':
        datamodule = SeqCombSingle(fold=fold, seed=seed, batch_size=batch_size)

        num_features = 1
        num_classes = 4
        max_len = 200
    elif data == 'FreqShape':
        datamodule = FreqShape(fold=fold, seed=seed, batch_size=batch_size)
        num_features = 1
        num_classes = 4
        max_len = 50

    elif data == 'FreqShapeVar':
        datamodule = FreqShapeVar(fold=fold, seed=seed, batch_size=batch_size)
        num_features = 1
        num_classes = 5
        max_len = 50

    elif data == 'LowVarDetectSingle':
        datamodule = LowVarDetectSingle(fold=fold, seed=seed, batch_size=batch_size)
        num_features = 1
        num_classes = 3
        max_len = 200
        
    return datamodule, num_features, max_len, num_classes
    
