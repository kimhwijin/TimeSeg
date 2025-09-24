import sys
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)
from distributions.CategoricalToCategorical import CategoricalToCategorical


def get_dist(seg_dist, seq_len):
    if seg_dist == 'cat_cat':
        d_start, d_end = seq_len, seq_len
        distribution_class = CategoricalToCategorical
        distribution_kwargs = {
            'seq_len': seq_len
        }

    return distribution_class, distribution_kwargs, (d_start, d_end)