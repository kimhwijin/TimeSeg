# TimeSeg: An Information-Theoretic Segment-Wise Explainer for Time-Series Predictions

<p align="center">
  <a href="https://iclr.cc/virtual/2026/poster/10008670"><img src="https://img.shields.io/badge/ICLR%202026-Accepted-brightgreen" alt="ICLR 2026"></a>
  <a href="https://openreview.net/forum?id=alt9mSWULk"><img src="https://img.shields.io/badge/OpenReview-alt9mSWULk-8c1b13" alt="OpenReview"></a>
</p>

> 🎉 **TimeSeg has been accepted to ICLR 2026!**
> 
![Model Overview](https://github.com/user-attachments/assets/2e1c3991-aa23-4038-99e7-3720e169cdaf)

## Abstract

Explaining predictions of black-box time-series models remains a challenging problem due to the dynamically evolving patterns within individual sequences and their complex temporal dependencies. Unfortunately, existing explanation methods largely focus on point-wise explanations, which fail to capture broader temporal context, while methods that attempt to highlight interpretable temporal patterns (e.g., achieved by incorporating a regularizer or fixed-length patches) often lack principled definitions of meaningful segments. This limitation frequently leads to fragmented and confusing explanations for end users.

As such, the notion of segment-wise explanations has remained underexplored, with little consensus on what constitutes an interpretable segment or how such segments should be identified. To bridge this gap, we define segment-wise explanation for black-box time-series models as the task of selecting contiguous subsequences that maximize their joint mutual information with the target prediction. Building on this formulation, we propose **TimeSeg**, a novel information-theoretic framework that employs reinforcement learning to sequentially identify predictive temporal segments at a per-instance level.

By doing so, TimeSeg produces segment-wise explanations that capture holistic temporal patterns rather than fragmented points, providing class-predictive patterns in a human-interpretable manner. Extensive experiments on both synthetic and real-world datasets demonstrate that TimeSeg produces more coherent and human-understandable explanations, while achieving performance that matches or surpasses existing methods on downstream tasks using the identified segments.

## Method Overview

TimeSeg is a **post-hoc, strict black-box** explainer: it requires only the model's inputs and outputs (no gradients, embeddings, or architectural knowledge). Given an input sequence, it dynamically selects a set of contiguous, variable-length segments that are jointly most predictive of the black-box output.

- **Information-theoretic objective.** Segment selection is defined as maximizing the mutual information (MI) between the selected segments and the black-box prediction, with a sparsity penalty that keeps explanations compact and non-overlapping.
- **Sequential reformulation.** The intractable joint MI is decomposed into conditional MI (CMI) terms via the chain rule, turning an exponential combinatorial search ($O(2^T)$) into a tractable sequential decision process.
- **Reinforcement learning.** The explainer is a stochastic policy trained with **PPO** in an actor–critic setup. A two-step start/end policy guarantees valid, non-empty segments; each CMI reward is the cross-entropy gap of the black-box with vs. without the newly added segment.
- **Adaptive length.** An instance-specific termination rule (threshold $\tau$) stops selection when the marginal information gain becomes negligible, so the number of segments $K$ adapts per instance (up to $K_{\max}$).

## Requirements

- Python 3.9
- PyTorch 2.6.0, TorchRL / TensorDict (RL), PyTorch Lightning
- scikit-learn, NumPy, pandas, matplotlib
- `time_interpret` (baseline explainers), `tslearn`, `stumpy`, `fastdtw` (segment analysis)

See `requirements.txt` for exact pinned versions.

# Quick Start

## 1) Create and activate a conda env (Python 3.9)

```bash
conda create --name time-segment python==3.9
conda activate time-segment
```

## 2) Install python dependencies

```bash
pip install -r requirements.txt
git clone https://github.com/TimeSynth/TimeSynth.git
cd TimeSynth
python setup.py install
cd ..
```

## 3) Pre-train Black-box Model

```bash
bash ./scripts/blackbox_train.sh
```

This pre-trains the black-box (a Temporal Convolutional Network) with default settings:

```bash
python blackbox_train.py \
    --model_type tcn \
    --fold 0 \
    --data GunPoint \
    --seed 42 \
    --epoch 1
```

The trained checkpoint is saved under `./model_ckpt/{dataset}/` (e.g., `./model_ckpt/MITECG/`) and is used automatically by the main pipeline.

## 4) Run TimeSeg (train the explainer)

```bash
bash ./scripts/main.sh
```

This trains the TimeSeg explainer (policy/value networks) and automatically runs the test step at the end:

```bash
python main.py \
    --train_type   ppo \
    --dataset      GunPoint \
    --split        0 \
    --mask_type    mean \
    --epochs       5 \
    --ppo_epochs   4 \
    --max_segment  5 \
    --seg_dist     cat_cat \
    --batch_size   256 \
    --rollout_len  1024 \
    --weights      1.0,0.3 \
    --threshold    0.3
```

Key arguments: `--max_segment` = $K_{\max}$, `--threshold` = termination $\tau$, `--weights` = (CE reward, length penalty $\lambda$), `--seg_dist` = segment-index distribution (`cat_cat` is the default Cat–Cat policy), `--mask_type` = value used for masked-out points (`mean` / `zero`).

# Datasets

TimeSeg is evaluated on synthetic datasets (with ground-truth explanatory segments) and real-world datasets. Synthetic and preprocessing setups follow TimeX (Queen et al., 2023).

**Synthetic** (class-defining motifs inserted into a NARMA noise base; ground truth = motif positions):

| Dataset          | #Samples | Length | Dim | Classes |
| ---------------- | -------: | -----: | --: | ------: |
| SeqComb-UV       |    6,100 |    200 |   1 |       4 |
| FreqShapes-V     |    6,100 |     50 |   1 |       5 |
| LowVarDetect-UV  |    6,100 |    200 |   1 |       2 |

**Real-world:**

| Dataset   | #Samples | Length | Dim | Classes | Segment-level GT       |
| --------- | -------: | -----: | --: | ------: | ---------------------- |
| MIT-ECG   |   90,337 |    360 |   1 |       2 | ✅ QRS interval        |
| Epilepsy  |   11,500 |    178 |   1 |       2 | ❌                     |
| Wafer     |    7,164 |    152 |   1 |       2 | ❌                     |
| GunPoint  |      400 |    150 |   1 |       2 | ❌                     |

Corresponding dataset loaders live under `./datasets/` (`SeqCombSingle.py`, `FreqShapeVar.py`, `LowVarDetectSingle.py`, `MITECG.py`, `epilepsy.py`, `UCR.py`).

# Results

TimeSeg matches or surpasses state-of-the-art explainers **while operating in a strict black-box setting**, unlike IG (needs gradients) and TimeX++ (needs internal embeddings).

- **Overlap with ground truth (MIT-ECG):** F1 **0.739** / IoU **0.621**, vs. the second-best TimeX++ at 0.593 / 0.460.
- **Explanation fidelity (unannotated datasets):** retaining only the selected segments causes a ≤ 2% AUROC drop for TimeSeg, whereas the second-best method drops ≥ 31%.
- **Segment quality:** contiguity as low as 1–2%, i.e., few boundaries and coherent segments.
- **Robustness:** consistent behavior across TCN, RNN, and Transformer black-box backbones, and a natural multivariate extension via channel selection.

Full tables, ablations ($\lambda$, $K_{\max}$, $\tau$), and qualitative examples are in the paper.

## Implementation Details

- **Black-box $g_\theta$:** Temporal Convolutional Network (TCN), 6 conv blocks, kernel size 3, dilated + residual, dropout 0.1; Adam (lr $10^{-3}$).
- **Policy $\pi_\phi$ / Value $V_\psi$:** 3-layer 1D CNNs (hidden dim 128), factorized start/end policies.
- **PPO:** clip $\epsilon = 0.2$, discount $\gamma = 0.99$, entropy coef 0.01, rollout 1,024, 4 PPO epochs.
- **Defaults:** $\lambda = 0.3$, $\tau = 0.3$, $K_{\max} = 5$.
- **Hardware (reference):** Intel Xeon CPU + NVIDIA RTX A6000 GPU.

# Citation

If you find TimeSeg useful in your research, please consider citing our paper:

```bibtex
@inproceedings{kim2026timeseg,
  title     = {TimeSeg: An Information-Theoretic Segment-Wise Explainer for Time-Series Predictions},
  author    = {Kim, Hwijin and Kim, Jaeho and Lee, Changhee},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=alt9mSWULk}
}
```

# Acknowledgements

Our synthetic datasets and preprocessing build on [TimeX](https://github.com/mims-harvard/TimeX) (Queen et al., 2023), and baseline explainers use the [`time_interpret`](https://github.com/josephenguehard/time_interpret) library.

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00358602) and by IITP grants funded by the Korea government (MSIT): the Artificial Intelligence Graduate School Program (No. RS-2019-II190079, Korea University), the AI Star Fellowship (No. RS-2025-02304828), and the AI Research Hub Project (No. RS-2024-00457882).
