import sys
from pathlib import Path
p = str(Path(__file__).absolute().parent.parent)
sys.path.append(p)


from distributions.CategoricalToNegativeBinomial import CategoricalToNegativeBinomial
from distributions.CategoricalToCategorical import CategoricalToCategorical
from distributions.CauchyCategoricalToCategorical import CauchyCategoricalToCategorical



def get_dist(seg_dist, seq_len, nb_transform):
    if seg_dist == 'cat_cat':
        d_start, d_end = seq_len, seq_len
        distribution_class = CategoricalToCategorical
    if seg_dist == 'cauchy_cat_cat':
        d_start, d_end = seq_len, seq_len
        distribution_class = CauchyCategoricalToCategorical
    if seg_dist == 'cat_nb':
        d_start, d_end = seq_len, 2
        distribution_class = CategoricalToNegativeBinomial

    if 'nb' in seg_dist:
        distribution_kwargs = {
            'seq_len': seq_len,
            'nb_transform': nb_transform
        }
    else:
        distribution_kwargs = {
            'seq_len': seq_len
        }

    return distribution_class, distribution_kwargs, (d_start, d_end)