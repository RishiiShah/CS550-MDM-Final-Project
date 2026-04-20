"""
Evaluation metrics: MAE, RMSE, Precision@K, Recall@K, F-measure, NDCG@K.
Includes full-catalogue evaluation (evaluate_recommendations) and the
sampled-negative protocol (evaluate_sampled) standard in RS literature.
"""
from collections import Counter

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


def popularity_bias_metrics(
    rec_dict: dict,
    item_popularity: dict,
    all_items,
    k: int = 10,
    head_fraction: float = 0.10,
    tail_fraction: float = 0.50,
) -> dict:
    """
    Quantify how strongly a recommender amplifies popular items.

    Parameters
    ----------
    rec_dict         : {user_id: [item_id, ...]} top-k recommendations
    item_popularity  : {item_id: interaction_count in train}
    all_items        : iterable of all catalogue item_ids
    """
    all_items = list(all_items)
    if not all_items:
        return {
            "AvgPopularity@K": 0.0,
            "CatalogAvgPopularity": 0.0,
            "PopularityLiftPct": 0.0,
            "HeadShare@K": 0.0,
            "TailShare@K": 0.0,
            "Coverage@K": 0.0,
            "ExposureEntropy": 0.0,
        }

    # Ensure all catalogue items have a popularity value (0 for unseen-in-train).
    pop_map = {it: float(item_popularity.get(it, 0.0)) for it in all_items}

    sorted_items = sorted(all_items, key=lambda x: pop_map[x], reverse=True)
    n_items = len(sorted_items)
    n_head = max(1, int(np.ceil(n_items * head_fraction)))
    n_tail = max(1, int(np.ceil(n_items * tail_fraction)))
    if n_head + n_tail > n_items:
        n_tail = max(1, n_items - n_head)
    head_set = set(sorted_items[:n_head])
    tail_candidates = [it for it in reversed(sorted_items) if it not in head_set]
    tail_set = set(tail_candidates[:n_tail])

    rec_items = []
    for recs in rec_dict.values():
        rec_items.extend([it for it in recs[:k] if it in pop_map])

    if not rec_items:
        catalog_avg = float(np.mean(list(pop_map.values())))
        return {
            "AvgPopularity@K": 0.0,
            "CatalogAvgPopularity": catalog_avg,
            "PopularityLiftPct": 0.0,
            "HeadShare@K": 0.0,
            "TailShare@K": 0.0,
            "Coverage@K": 0.0,
            "ExposureEntropy": 0.0,
        }

    rec_pop = np.array([pop_map.get(it, 0.0) for it in rec_items], dtype=np.float64)
    avg_pop_rec = float(np.mean(rec_pop))
    avg_pop_catalog = float(np.mean(list(pop_map.values())))
    if avg_pop_catalog > 0:
        pop_lift = (avg_pop_rec - avg_pop_catalog) / avg_pop_catalog * 100.0
    else:
        pop_lift = 0.0

    rec_total = float(len(rec_items))
    head_share = float(sum(1 for it in rec_items if it in head_set) / rec_total)
    tail_share = float(sum(1 for it in rec_items if it in tail_set) / rec_total)

    unique_rec = set(rec_items)
    coverage = float(len(unique_rec) / n_items)

    # Exposure entropy over surfaced items; higher means less concentrated exposure.
    exposure_counts = np.array(list(Counter(rec_items).values()), dtype=np.float64)
    p = exposure_counts / exposure_counts.sum()
    entropy = float(-(p * np.log2(p + 1e-12)).sum())

    return {
        "AvgPopularity@K": avg_pop_rec,
        "CatalogAvgPopularity": avg_pop_catalog,
        "PopularityLiftPct": pop_lift,
        "HeadShare@K": head_share,
        "TailShare@K": tail_share,
        "Coverage@K": coverage,
        "ExposureEntropy": entropy,
    }


def user_activity_fairness_metrics(
    rec_dict: dict,
    test_items_dict: dict,
    train_seen_dict: dict,
    k: int = 10,
) -> dict:
    """
    Measure quality parity across user-activity cohorts.

    Users are grouped by training-history length terciles:
      - cold  : shortest histories
      - warm  : medium histories
      - power : longest histories
    """
    eligible_users = [
        u for u in rec_dict.keys()
        if u in test_items_dict and len(test_items_dict.get(u, set())) > 0
    ]
    if not eligible_users:
        empty_groups = {
            "cold": {
                "n_users": 0,
                "avg_train_history": 0.0,
                "Precision@K": 0.0,
                "Recall@K": 0.0,
                "F-measure": 0.0,
                "NDCG@K": 0.0,
            },
            "warm": {
                "n_users": 0,
                "avg_train_history": 0.0,
                "Precision@K": 0.0,
                "Recall@K": 0.0,
                "F-measure": 0.0,
                "NDCG@K": 0.0,
            },
            "power": {
                "n_users": 0,
                "avg_train_history": 0.0,
                "Precision@K": 0.0,
                "Recall@K": 0.0,
                "F-measure": 0.0,
                "NDCG@K": 0.0,
            },
        }
        return {
            "groups": empty_groups,
            "gaps": {
                "Precision@K": 0.0,
                "Recall@K": 0.0,
                "NDCG@K": 0.0,
            },
        }

    history_sizes = np.array(
        [len(train_seen_dict.get(u, set())) for u in eligible_users],
        dtype=np.float64,
    )
    q1, q2 = np.quantile(history_sizes, [1 / 3, 2 / 3])

    buckets = {"cold": [], "warm": [], "power": []}
    for u in eligible_users:
        h = len(train_seen_dict.get(u, set()))
        if h <= q1:
            buckets["cold"].append(u)
        elif h <= q2:
            buckets["warm"].append(u)
        else:
            buckets["power"].append(u)

    groups_out = {}
    for name, users in buckets.items():
        if not users:
            groups_out[name] = {
                "n_users": 0,
                "avg_train_history": 0.0,
                "Precision@K": 0.0,
                "Recall@K": 0.0,
                "F-measure": 0.0,
                "NDCG@K": 0.0,
            }
            continue

        p_vals, r_vals, f_vals, n_vals = [], [], [], []
        for uid in users:
            rec_list = rec_dict.get(uid, [])
            relevant = test_items_dict.get(uid, set())
            p = precision_at_k(rec_list, relevant, k)
            r = recall_at_k(rec_list, relevant, k)
            f = f_measure(p, r)
            n = ndcg_at_k(rec_list, relevant, k)
            p_vals.append(p)
            r_vals.append(r)
            f_vals.append(f)
            n_vals.append(n)

        groups_out[name] = {
            "n_users": int(len(users)),
            "avg_train_history": float(np.mean([len(train_seen_dict.get(u, set())) for u in users])),
            "Precision@K": float(np.mean(p_vals)),
            "Recall@K": float(np.mean(r_vals)),
            "F-measure": float(np.mean(f_vals)),
            "NDCG@K": float(np.mean(n_vals)),
        }

    p_group = [v["Precision@K"] for v in groups_out.values() if v["n_users"] > 0]
    r_group = [v["Recall@K"] for v in groups_out.values() if v["n_users"] > 0]
    n_group = [v["NDCG@K"] for v in groups_out.values() if v["n_users"] > 0]

    gaps = {
        "Precision@K": float(max(p_group) - min(p_group)) if p_group else 0.0,
        "Recall@K": float(max(r_group) - min(r_group)) if r_group else 0.0,
        "NDCG@K": float(max(n_group) - min(n_group)) if n_group else 0.0,
    }

    return {
        "groups": groups_out,
        "gaps": gaps,
    }
