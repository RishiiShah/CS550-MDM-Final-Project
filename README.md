# Amazon Video Games Recommender System

CS550 — Massive Data Mining and Learning | Final Project  
Rishabh Sanjeev Shah rs2780   
Om Nirav Patel op187  
Vivek Sai Manthri vm614  
Rutgers University

---

## Overview

A full recommender system pipeline built on the Amazon Video Games 5-core dataset. Covers two tasks:

- **Task A — Rating Prediction**: predict the numerical rating a user would give an item (evaluated by MAE and RMSE)
- **Task B — Top-N Recommendation**: rank and return the 10 items a user is most likely to enjoy (evaluated by P@10, R@10, F-measure, NDCG@10)

An interactive Flask demo (port 5001) exposes all models through four tabs: Recommendations, Evaluation, Fairness, and Explainability.

---

## Dataset

| Property | Value |
|---|---|
| Source | Amazon Video Games 5-core (McAuley et al.) |
| File | `data/Video_Games_5.json.gz` |
| Total records | 497,565 |
| Users | 55,220 |
| Items | 17,408 |
| Train ratings | 412,281 |
| Test ratings | 85,284 |
| Matrix density | 0.0518% |
| Avg ratings / user | 9.01 |
| Avg ratings / item | 28.58 |
| Avg rating value | 4.22 / 5 |

**Split strategy**: per-user temporal split — the most recent 20% of each user's reviews form the test set. This prevents any future signal from leaking into training.

---

## Models

### Task A — Rating Prediction

#### Matrix Factorization (MF) — Primary
Biased SGD matrix factorization (SVD-style). Learns user factors **P**, item factors **Q**, user biases **b_u**, and item biases **b_i** jointly via vectorized mini-batch SGD.

```
x̂_ui = μ + b_u + b_i + P_u · Q_i
```

Current selected configuration: `n_factors=32`, `n_epochs=60`, `lr=0.003`, `reg=0.03`

#### SVD++ — Baseline
Extends MF with implicit feedback from high-confidence interactions (ratings ≥ 4★). The implicit term augments the user factor with a weighted sum over item vectors for positively-rated items.

```
x̂_ui = μ + b_u + b_i + Q_i · (P_u + |N(u)|^{-0.5} · Σ_{j∈N(u)} y_j)
```

Only ratings ≥ 4★ are used as implicit positives — including 1–3★ reviews would treat negative experiences as preference signal.

Current selected configuration: `n_factors=40`, `n_epochs=55`, `lr=0.004`, `reg=0.025`

---

### Task B — Top-N Recommendation

#### SLIM (Sparse Linear Methods) — Primary
Learns a sparse item-item similarity matrix **W** by solving per-item ElasticNet regressions with non-negativity and zero-diagonal constraints. Recommendations are user rating vectors multiplied through **W**.

```
min_{w_j} ½‖r_j − X_j w_j‖² + α(ρ‖w_j‖₁ + ½(1−ρ)‖w_j‖²)
subject to: w_j ≥ 0,  w_{jj} = 0
```

Hyperparameters: `alpha=0.01`, `l1_ratio=0.1`, `max_iter=500`

#### Bayesian Personalized Ranking (BPR) — Baseline
Pairwise ranking model that optimises the probability that a user prefers observed item *i* over unobserved item *j*. Only ratings ≥ 4★ are treated as positive interactions.

Current selected configuration: `n_factors=48`, `n_epochs=40`, `lr=0.01`, `reg=0.01`

#### TextHybrid (SLIM + TF-IDF) — Hybrid
Reranks a pool of SLIM candidates using a content similarity score derived from TF-IDF review vectors. A user profile is constructed as the mean TF-IDF vector over their rated items. Final score blends the normalized SLIM score with cosine similarity to the user profile.

```
final_score = (1 − α) · SLIM_norm + α · content_sim
```

Hyperparameters: `max_features=5000`, `alpha=0.3`, `ngram_range=(1,2)`

---

## Results

Current status: this run is stable and reproducible, and active tuning is focused on improving Task-A MAE and BPR ranking quality while preserving SLIM performance.

### Task A — Rating Prediction (lower is better)

| Model | MAE | RMSE |
|---|---|---|
| **MF (primary)** | **0.7868** | **1.0804** |
| SVD++ (baseline) | 0.8125 | 1.1058 |
| MF/SVD++ blend | 0.7868 | 1.0804 |

MF outperforms SVD++. With only ~9.0 ratings per user on average, the implicit feedback term in SVD++ has insufficient signal to generalise and adds noise instead of benefit.

### Task B — Top-10 Recommendation (higher is better)

| Model | P@10 | R@10 | F-measure | NDCG@10 |
|---|---|---|---|---|
| BPR (baseline) | 0.0012 | 0.0084 | 0.0020 | 0.0045 |
| **SLIM (primary)** | **0.0108** | **0.0846** | **0.0184** | **0.0518** |
| TextHybrid (SLIM + TF-IDF) | 0.0108 | 0.0850 | 0.0185 | 0.0518 |
| SLIM-Balanced | 0.0108 | 0.0851 | 0.0185 | 0.0526 |
| SLIM-Discovery | 0.0100 | 0.0783 | 0.0171 | 0.0496 |

SLIM dominates. The Video Games domain has strong item-item co-purchase patterns (franchise, platform, genre) that sparse item-item linear regression captures directly. BPR trained on 4★/5★ only is the correct implicit-feedback baseline but is limited by dataset sparsity. TextHybrid shows marginal gains over SLIM from TF-IDF review content.

> **Note on low absolute values**: P@10 ≈ 1% and R@10 ≈ 7.7% are expected and normal for full-catalogue evaluation on sparse datasets. The median user has only 5 training ratings, and test sets contain 1–2 items per user. These numbers align with published results on similar Amazon review datasets.

---

## Sparsity Handling

The rating matrix density is 0.0518% — extremely sparse. Key mitigations:

- **5-core filtering (user-side)**: only users with ≥ 5 interactions are retained
- **Temporal split**: avoids data leakage; models train on older interactions only
- **SLIM item-item similarity**: captures co-purchase patterns even with sparse user histories
- **Popularity fallback**: when SLIM's co-occurrence scores are all zero for a user (no overlap with any item's neighborhood), the recommendation list is padded with globally-popular unseen items — ensuring every user always receives exactly 10 recommendations
- **BPR cold-start fallback**: completely unknown users (not seen during training) receive popularity-ranked items

Current artifact snapshot also records the latest tuned selection:

- MF best config: `n_factors=32`, `n_epochs=60`, `lr=0.003`, `reg=0.03`, weights=`mild`
- SVD++ best config: `n_factors=40`, `n_epochs=55`, `lr=0.004`, `reg=0.025`, weights=`mild`
- BPR best config: `n_factors=48`, `n_epochs=40`, `lr=0.01`, `reg=0.01`
- Task-A blend alpha: `1.0`

---

## Optional Tasks

### Fairness Analysis
The Fairness tab reports three dimensions:

- **Popularity bias**: distribution of recommended items by interaction-count quartile (long-tail vs. popular)
- **Catalogue coverage**: fraction of the item catalogue that appears in at least one recommendation
- **User-group fairness**: P@10 and R@10 broken down by user activity level (cold / medium / active)

### Explainability
The Explainability tab decomposes a prediction into three components:

- **CF score**: MF-predicted rating from user/item latent factors
- **Content score**: cosine similarity between user TF-IDF profile and item TF-IDF vector, mapped to [1, 5]
- **Blended score**: dynamic weighted combination — weight shifts toward CF for active users and toward content for cold-start users
- **Counterfactual analysis**: shows what the blended score would be if either the CF or content component were replaced with a neutral value of 4.0

---

## Project Structure

```
finalproject/
├── app.py                    # Flask demo (port 5001)
├── train_and_evaluate.py     # Full training + evaluation pipeline
├── requirements.txt
├── models/
│   ├── matrix_factorization.py   # Task A primary
│   ├── svdpp.py                  # Task A baseline
│   ├── slim.py                   # Task B primary
│   ├── bpr.py                    # Task B baseline
│   └── text_hybrid.py            # Task B hybrid
├── utils/
│   ├── preprocess.py             # Data loading, train/test split, id maps
│   └── evaluate.py               # MAE, RMSE, P@K, R@K, F, NDCG@K
├── data/
│   └── Video_Games_5.json.gz
└── artifacts/
    ├── results.json              # Saved evaluation metrics
    ├── train.csv / test.csv      # Processed splits
    ├── model.pkl                 # Trained MF
    ├── svdpp_model.pkl           # Trained SVD++
    ├── bpr_model.pkl             # Trained BPR
    ├── slim_model.pkl            # Trained SLIM
    └── text_hybrid.pkl           # Trained TextHybrid
```

---

## Setup and Usage

### Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train all models and evaluate
```bash
python train_and_evaluate.py
```

This runs the full pipeline (data loading → training → evaluation → artifact saving). Approximate runtime: 25–35 minutes depending on hardware (SLIM is the bottleneck at ~17k ElasticNet fits).

### Run the Flask demo
```bash
python app.py
# Open http://127.0.0.1:5001
```

---

## Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.11
flask>=3.0
tqdm>=4.66
```

---

## References

- Ning, X. & Karypis, G. (2011). *SLIM: Sparse Linear Methods for Top-N Recommender Systems*. ICDM.
- Rendle, S. et al. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback*. UAI.
- Koren, Y. (2008). *Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model*. KDD.
- He, X. et al. (2017). *Neural Collaborative Filtering*. WWW. *(sampled-negative evaluation protocol)*
- McAuley, J. et al. (2015). *Image-Based Recommendations on Styles and Substitutes*. SIGIR. *(dataset)*
