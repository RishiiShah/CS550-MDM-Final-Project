"""
Data preprocessing: load JSON/JSON.gz, split 80/20 per user into train/test.

Two-pass streaming loader is used when the file is .gz — avoids building a
full multi-million-row DataFrame in memory.  For plain .json files the single-
pass path is used instead (file already fits in RAM).
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE   = "data/Video_Games_5.json.gz"
RANDOM_SEED = 42


def load_data(path: str = DATA_FILE,
              min_reviews: int = 5,
              user_sample_frac: float = 1.0,
              seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Stream-load an Amazon review file (.json or .json.gz).

    Parameters
    ----------
    path             : path to .json or .json.gz file
    min_reviews      : keep only users with at least this many reviews
    user_sample_frac : fraction of qualifying users to sample (1.0 = all)
    seed             : random seed for user sampling
    """
    path = str(path)
    is_gz = path.endswith(".gz")
    opener = (lambda p: gzip.open(p, "rt", encoding="utf-8")) if is_gz \
             else (lambda p: open(p, encoding="utf-8"))

    # ── Pass 1: count reviews per user ────────────────────────────────────────
    print(f"Pass 1: counting reviews per user in {path}...")
    user_counts: dict = {}
    with opener(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            uid = json.loads(line).get("reviewerID")
            if uid:
                user_counts[uid] = user_counts.get(uid, 0) + 1
            if (i + 1) % 500_000 == 0:
                print(f"  Counted {i + 1:,} lines...")

    valid_users = [u for u, c in user_counts.items() if c >= min_reviews]
    print(f"  Users with >= {min_reviews} reviews: {len(valid_users):,}")

    # Optional down-sampling (useful for very large datasets)
    np.random.seed(seed)
    if user_sample_frac < 1.0:
        sample_size = max(1, int(len(valid_users) * user_sample_frac))
        sampled_users = set(
            np.random.choice(valid_users, size=sample_size, replace=False).tolist()
        )
        print(f"  Sampled {sample_size:,} users ({user_sample_frac:.0%})")
    else:
        sampled_users = set(valid_users)
        print(f"  Using all {len(sampled_users):,} qualifying users")

    # ── Pass 2: load only records for selected users ──────────────────────────
    print("Pass 2: loading records for selected users...")
    records = []
    with opener(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("reviewerID") not in sampled_users:
                continue
            records.append({
                "user_id":   item.get("reviewerID"),
                "item_id":   item.get("asin"),
                "rating":    float(item.get("overall", 0)),
                "review":    (item.get("reviewText") or "").replace("\n", " "),
                "title":     (item.get("summary")    or "").replace("\n", " "),
                "timestamp": item.get("unixReviewTime", 0),
            })

    df = pd.DataFrame(records)
    df.dropna(subset=["user_id", "item_id", "rating"], inplace=True)

    # Remove duplicated events before splitting/evaluation.
    # Keying by (user, item, timestamp) avoids contradictory copies of the same event.
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], keep="first")
    removed = before - len(df)

    df.reset_index(drop=True, inplace=True)
    print(f"  Loaded {len(df):,} records — "
          f"{df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")
    if removed > 0:
        print(f"  Removed {removed:,} exact duplicate interactions")
    return df


def train_test_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Per-user temporal split: sort each user's reviews by timestamp and
    hold out the most recent test_ratio as the test set.

    This prevents data leakage — the model is never trained on reviews
    that happened after the reviews it is evaluated on.
    """
    train_rows, test_rows = [], []
    # Stable sort keeps deterministic ordering for tied timestamps.
    work_df = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    for uid, group in work_df.groupby("user_id", sort=False):
        sorted_idx = group.index.tolist()
        n_test = max(1, int(len(sorted_idx) * test_ratio))
        test_set = set(sorted_idx[-n_test:])
        for idx in sorted_idx:
            (test_rows if idx in test_set else train_rows).append(idx)
    train_df = df.loc[train_rows].reset_index(drop=True)
    test_df  = df.loc[test_rows].reset_index(drop=True)
    return train_df, test_df


def build_id_maps(df: pd.DataFrame):
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: it for it, i in item2idx.items()}
    return user2idx, item2idx, idx2user, idx2item


if __name__ == "__main__":
    df = load_data()
    train_df, test_df = train_test_split(df)
    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")
