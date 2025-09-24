# TimeSeg: An Information-Theoretic Segment-Level Explainer for Time-Series Predictions

![Model Overview](https://github.com/user-attachments/assets/2e1c3991-aa23-4038-99e7-3720e169cdaf)

## Overall explanation process

Explaining predictions of black-box time-series models remains a challenging problem due to the dynamically evolving patterns within individual sequences and their complex temporal dependencies. Unfortunately, existing explanation methods largely focus on point-wise explanations, which fail to capture broader temporal context, while methods that attempt to highlight interpretable temporal patterns~(\eg achieved by incorporating a regularizer or fixed-length patches) often lack principled definitions of meaningful segments. This limitation frequently leads to fragmented and confusing explanations for end users. 
As such, the notion of segment-level explanations has remained underexplored, with little consensus on what constitutes an \textit{interpretable} segment or how such segments should be identified. To bridge this gap, we define segment-level explanation for black-box time-series models as the task of selecting contiguous subsequences that maximize their joint mutual information with the target prediction. Building on this formulation, we propose \name, a novel information-theoretic framework that employs reinforcement learning to sequentially identify predictive temporal segments at a per-instance level. 
By doing so, \name produces segment-wise explanations that capture holistic temporal patterns rather than fragmented points, providing class-predictive patterns in a human-interpretable manner. Extensive experiments on both synthetic and real‑world datasets demonstrate that \name produces more coherent and human-understandable explanations, while achieving performance that matches or surpasses existing methods on downstream tasks using the identified segments.



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
This will pre-train the black-box with default settings.
The trained checkpoint is saved under ./model_ckpt/{dataset}/ (e.g., ./model_ckpt/MITECG/) and will be used automatically by the main pipeline.

## 4) Run TimeSeg (train the explainer)
```bash
bash ./scripts/main.sh
```
This command trains the TimeSeg explainer (policy/value networks) and automatically runs the test step at the end.


