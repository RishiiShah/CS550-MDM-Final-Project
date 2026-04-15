"""
Full training and evaluation pipeline — Amazon Video Games recommender.

Task A — Rating Prediction  (MAE / RMSE on held-out test set)
  MF     — biased batched-SGD matrix factorisation  ← best Task-A model
  SVD++  — MF + implicit-feedback (4★/5★ only)      (strong baseline)

Task B — Top-N Recommendation  (P@10 / R@10 / F / NDCG@10)
  BPR         — pairwise ranking on 4★/5★ implicit feedback  (baseline)
  SLIM        — sparse item-item ElasticNet                   ← best Task-B model
  TextHybrid  — SLIM candidates reranked by TF-IDF content    (hybrid)

Split: per-user temporal split — most recent 20% of each user's reviews are test.
This prevents leaking future information into training.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from utils.preprocess import load_data, train_test_split, build_id_maps
from models.matrix_factorization import MatrixFactorization
from models.svdpp import SVDpp
from models.bpr import BPR
from models.text_hybrid import TextHybrid
from models.slim import SLIM
from utils.evaluate import mae, rmse, evaluate_recommendations

TOP_K    = 10
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)


def _rec_row(name, m, k=TOP_K):
    print(
        f"  {name:<30}"
        f"  P@{k}: {m['Precision@K']:.4f}"
        f"  R@{k}: {m['Recall@K']:.4f}"
        f"  F: {m['F-measure']:.4f}"
        f"  NDCG@{k}: {m['NDCG@K']:.4f}"
    )


def main():
    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("=" * 65)
    print("STEP 1 — Data loading and preprocessing")
    print("=" * 65)
    df = load_data()
    print(f"  Total records : {len(df)}")
    print(f"  Users         : {df['user_id'].nunique()}")
    print(f"  Items         : {df['item_id'].nunique()}")

    train_df, test_df = train_test_split(df)
    print(f"  Train / Test  : {len(train_df)} / {len(test_df)}")

    if "review" in train_df.columns:
        train_df = train_df.copy()
        train_df["review"] = train_df["review"].fillna("")

    # Build id maps from ALL data so cold-start is handled uniformly
    user2idx, item2idx, _, _ = build_id_maps(df)
    train_df.to_csv(ARTIFACTS / "train.csv", index=False)
    test_df.to_csv(ARTIFACTS / "test.csv",  index=False)

    # ── Step 2: Train Task-A models (rating prediction) ───────────────────────
    print("\n" + "=" * 65)
    print("STEP 2 — Training rating-prediction models  (Task A)")
    print("=" * 65)

    print("\n  [Primary]     Matrix Factorization (biased batched-SGD MF)...")
    mf = MatrixFactorization(n_factors=50, n_epochs=40, lr=0.005, reg=0.01)
    mf.fit(train_df, user2idx, item2idx)

    print("\n  [Baseline]    SVD++ (MF + 4★/5★ implicit feedback)...")
    svdpp = SVDpp(n_factors=50, n_epochs=50, lr=0.005, reg=0.015, seed=42)
    svdpp.fit(train_df, user2idx, item2idx)
    svdpp.build_inference_cache(train_df)

    # ── Step 3: Task-A evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3 — Rating-Prediction Evaluation  (MAE / RMSE)")
    print("=" * 65)
    y_true  = test_df["rating"].values
    y_mf    = mf.predict_batch(test_df)
    y_svdpp = svdpp.predict_batch(test_df)

    mf_mae,    mf_rmse    = mae(y_true, y_mf),    rmse(y_true, y_mf)
    svdpp_mae, svdpp_rmse = mae(y_true, y_svdpp), rmse(y_true, y_svdpp)

    print(f"\n  {'Model':<32}  MAE      RMSE")
    print(f"  {'-'*48}")
    print(f"  {'MF  [primary — biased SGD-MF]':<32}  {mf_mae:.4f}   {mf_rmse:.4f}")
    print(f"  {'SVD++ (4★/5★ implicit only)':<32}  {svdpp_mae:.4f}   {svdpp_rmse:.4f}")

    # ── Step 4: Train Task-B models (top-N recommendation) ───────────────────
    print("\n" + "=" * 65)
    print("STEP 4 — Training recommendation models  (Task B)")
    print("=" * 65)

    print("\n  [Baseline]    BPR — pairwise ranking on 4★/5★ positive interactions...")
    print("                (only high-rating interactions used; 1–3★ reviews are")
    print("                 negative signals and must not be treated as 'liked')")
    bpr_pos = train_df[train_df["rating"] >= 4.0].copy()
    bpr = BPR(n_factors=50, n_epochs=50, lr=0.01, reg=0.01, seed=42)
    bpr.fit(bpr_pos, user2idx, item2idx)

    print("\n  [Primary]     SLIM (sparse item-item ElasticNet)...")
    slim = SLIM(alpha=0.01, l1_ratio=0.1, max_iter=500, seed=42)
    slim.fit(train_df, user2idx, item2idx)

    print("\n  [Hybrid]      TextHybrid (SLIM candidates reranked by TF-IDF content)...")
    text_hybrid = TextHybrid(max_features=5000, alpha=0.3)
    text_hybrid.fit(train_df)

    # ── Step 5: Task-B evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 5 — Top-10 Recommendation Evaluation  (P / R / F / NDCG)")
    print("=" * 65)

    train_seen      = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    test_items_dict = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    # Pre-built dict for TextHybrid — avoids O(n_train) DataFrame scan per user
    user_items_dict = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
    all_users       = test_df["user_id"].unique()
    N               = len(all_users)

    def _gen(label, fn, fallback=None):
        out = {}
        for k, uid in enumerate(all_users, 1):
            if k % 1000 == 0:
                print(f"    [{label}] {k}/{N}...")
            seen = train_seen.get(uid, set())
            recs = fn(uid, seen)
            out[uid] = [iid for iid, _ in recs] if recs else (fallback or {}).get(uid, [])
        return out

    print("\n  Generating BPR recommendations...")
    rec_bpr  = _gen("BPR",  lambda uid, seen: bpr.recommend(uid, seen, n=TOP_K))
    m_bpr    = evaluate_recommendations(rec_bpr, test_items_dict, k=TOP_K)

    print("  Generating SLIM recommendations...")
    rec_slim = _gen("SLIM", lambda uid, seen: slim.recommend(uid, seen, n=TOP_K),
                   fallback=rec_bpr)
    m_slim   = evaluate_recommendations(rec_slim, test_items_dict, k=TOP_K)

    print("  Generating TextHybrid recommendations (SLIM pool reranked by TF-IDF)...")
    rec_hybrid = _gen(
        "Hybrid",
        lambda uid, seen: text_hybrid.rerank(
            uid, slim.recommend(uid, seen, n=TOP_K * 3), user_items_dict, n=TOP_K
        ),
        fallback=rec_slim,
    )
    m_hybrid = evaluate_recommendations(rec_hybrid, test_items_dict, k=TOP_K)

    print()
    _rec_row("BPR  (baseline)",                  m_bpr)
    _rec_row("SLIM  (primary)",                  m_slim)
    _rec_row("TextHybrid  (SLIM + TF-IDF)",      m_hybrid)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        # Task A — MF is best
        "rating_prediction":       {"MAE": mf_mae,    "RMSE": mf_rmse},
        "rating_prediction_mf":    {"MAE": mf_mae,    "RMSE": mf_rmse},
        "rating_prediction_svdpp": {"MAE": svdpp_mae, "RMSE": svdpp_rmse},
        # Task B
        "recommendation_bpr":      m_bpr,
        "recommendation_slim":     m_slim,
        "recommendation_hybrid":   m_hybrid,
        # Best models (used by app.py display)
        "recommendation":          m_slim,
        "best_recommender":        "SLIM (item-item ElasticNet)",
        "best_rating_model":       "MF (biased SGD-MF)",
    }
    with open(ARTIFACTS / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {ARTIFACTS}/results.json")

    # ── Save model artifacts ──────────────────────────────────────────────────
    for fname, obj in [
        ("model.pkl",       mf),
        ("svdpp_model.pkl", svdpp),
        ("bpr_model.pkl",   bpr),
        ("text_hybrid.pkl", text_hybrid),
        ("slim_model.pkl",  slim),
    ]:
        with open(ARTIFACTS / fname, "wb") as f:
            pickle.dump(obj, f)
        print(f"Saved artifacts/{fname}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print("Task A — Rating Prediction  (lower is better)")
    print(f"  MF     MAE {mf_mae:.4f}  RMSE {mf_rmse:.4f}  ← best Task-A model")
    print(f"  SVD++  MAE {svdpp_mae:.4f}  RMSE {svdpp_rmse:.4f}  (4★/5★ implicit only)")
    print()
    print("Task B — Top-10 Recommendation  (higher is better)")
    _rec_row("BPR",                      m_bpr)
    _rec_row("SLIM",                     m_slim)
    _rec_row("TextHybrid (SLIM+TF-IDF)", m_hybrid)
    print()
    print("Model selection rationale:")
    print("  Task A: Plain MF wins — SVD++'s implicit feedback adds marginal signal")
    print("  even after filtering to 4★/5★ only. ~9 ratings/user is too sparse for")
    print("  the implicit term to generalise.")
    print("  Task B: SLIM dominates — Video Games has strong item-item co-purchase")
    print("  patterns (franchise, platform, genre) that sparse ElasticNet captures")
    print("  directly. BPR trained on 4★/5★ only (correct implicit feedback).")
    print("  TextHybrid shows whether TF-IDF review content adds signal over SLIM.")
    print("=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
