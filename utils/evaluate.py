"""
Evaluation metrics: MAE, RMSE, Precision@K, Recall@K, F-measure, NDCG@K.
Includes full-catalogue evaluation (evaluate_recommendations) and the
sampled-negative protocol (evaluate_sampled) standard in RS literature.
"""
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    if not recommended:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def f_measure(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """NDCG where each relevant item has gain=1."""
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_recommendations(
    rec_dict: dict,          # {user_id: [item_id, ...]}  (ordered top-K list)
    test_items_dict: dict,   # {user_id: set of test item_ids}
    k: int = 10,
) -> dict:
    """Full-catalogue evaluation: recommended list vs all unseen test items."""
    precisions, recalls, fmeasures, ndcgs = [], [], [], []
    for user_id, rec_list in rec_dict.items():
        relevant = test_items_dict.get(user_id, set())
        p = precision_at_k(rec_list, relevant, k)
        r = recall_at_k(rec_list, relevant, k)
        f = f_measure(p, r)
        n = ndcg_at_k(rec_list, relevant, k)
        precisions.append(p)
        recalls.append(r)
        fmeasures.append(f)
        ndcgs.append(n)

    return {
        "Precision@K": float(np.mean(precisions)),
        "Recall@K":    float(np.mean(recalls)),
        "F-measure":   float(np.mean(fmeasures)),
        "NDCG@K":      float(np.mean(ndcgs)),
    }


def evaluate_sampled(
    rec_fn,               # callable(user_id, seen_items, n) -> [(item_id, score)]
    all_items,            # list or set of all item IDs in catalogue
    train_seen_dict,      # {user_id: set of item_ids seen in training}
    test_items_dict,      # {user_id: set of item_ids in test}
    k: int = 10,
    n_neg: int = 99,
    n_users: int = 100,
    seed: int = 42,
) -> dict:
    """
    Sampled-negative evaluation (He et al., NeuMF 2017).

    For each user: score ground-truth test positives against n_neg randomly
    sampled negatives, rank all candidates, compute P/R/NDCG@K.

    Reported numbers are higher than full-catalogue evaluation — this is expected
    and is the standard protocol in recommendation research papers.

    Parameters
    ----------
    rec_fn     : model.recommend(user_id, seen_items, n) interface
    all_items  : full item catalogue
    n_neg      : number of random negatives per user (99 is the standard)
    n_users    : how many test users to evaluate (subset for speed)
    """
    rng = np.random.default_rng(seed)
    all_items_list = list(all_items)

    # Pick eligible users (have at least one test positive)
    eligible = [u for u, items in test_items_dict.items() if len(items) > 0]
    if n_users < len(eligible):
        chosen = list(rng.choice(eligible, size=n_users, replace=False))
    else:
        chosen = eligible

    precisions, recalls, fmeasures, ndcgs = [], [], [], []

    for uid in chosen:
        positives = test_items_dict[uid]
        seen = train_seen_dict.get(uid, set())

        # Random negatives: not seen in training, not a test positive
        neg_pool = [it for it in all_items_list if it not in seen and it not in positives]
        if not neg_pool:
            continue
        n_sample = min(n_neg, len(neg_pool))
        negatives = rng.choice(neg_pool, size=n_sample, replace=False).tolist()

        candidates = set(positives) | set(negatives)

        # Get rankings from model — request a broad pool
        try:
            recs = rec_fn(uid, seen, n=min(len(all_items_list), 10_000))
        except Exception:
            continue

        # Build score dict; items not returned by model get -inf
        score_map = {it: sc for it, sc in recs if it in candidates}
        for it in candidates:
            if it not in score_map:
                score_map[it] = -np.inf

        ranked = sorted(candidates, key=lambda x: score_map[x], reverse=True)
        rec_list = ranked[:k]

        p = precision_at_k(rec_list, positives, k)
        r = recall_at_k(rec_list, positives, k)
        f = f_measure(p, r)
        n_val = ndcg_at_k(rec_list, positives, k)
        precisions.append(p)
        recalls.append(r)
        fmeasures.append(f)
        ndcgs.append(n_val)

    return {
        "Precision@K": float(np.mean(precisions)) if precisions else 0.0,
        "Recall@K":    float(np.mean(recalls))    if recalls    else 0.0,
        "F-measure":   float(np.mean(fmeasures))  if fmeasures  else 0.0,
        "NDCG@K":      float(np.mean(ndcgs))      if ndcgs      else 0.0,
        "n_users":     len(precisions),
    }
