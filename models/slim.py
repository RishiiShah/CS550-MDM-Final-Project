"""
SLIM (Sparse Linear Methods) recommender.

Learns a sparse item-item weight matrix W by solving per-item ElasticNet
regressions:
    min_{w_j} 0.5*||r_j - X_j w_j||^2 + alpha * (l1_ratio*||w_j||_1 + 0.5*(1-l1_ratio)*||w_j||_2^2)
    subject to: w_j >= 0,  w_{jj} = 0

Scores for user u:  s_u = r_u @ W
"""
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix, hstack
from sklearn.linear_model import ElasticNet
from tqdm import tqdm


class SLIM:
    def __init__(
        self,
        alpha: float = 0.01,
        l1_ratio: float = 0.1,
        max_iter: int = 500,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.seed = seed

    def _score_user(self, user_id: str, seen_items: set):
        """Return raw SLIM scores with seen items masked to 0.0."""
        u = self.user2idx.get(user_id)
        if u is None:
            return None

        r_u = self.R_train[u, :]
        scores = (r_u @ self.W).toarray().ravel()

        seen_idxs = [self.item2idx[iid] for iid in seen_items if iid in self.item2idx]
        if seen_idxs:
            scores[np.array(seen_idxs, dtype=np.int64)] = 0.0
        return scores

    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame, user2idx: dict, item2idx: dict):
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.idx2item = {v: k for k, v in item2idx.items()}

        n_users = len(user2idx)
        n_items = len(item2idx)

        # item_counts: item_id (str) -> int
        self.item_counts = train_df.groupby("item_id").size().to_dict()

        # Build sparse rating matrix (float32) in CSC format
        u_indices = train_df["user_id"].map(user2idx).values
        i_indices = train_df["item_id"].map(item2idx).values
        ratings = train_df["rating"].values.astype(np.float32)

        # Filter out rows where mapping produced NaN (unknown ids)
        valid = (~np.isnan(u_indices.astype(float))) & (~np.isnan(i_indices.astype(float)))
        u_indices = u_indices[valid].astype(np.int32)
        i_indices = i_indices[valid].astype(np.int32)
        ratings = ratings[valid]

        R = csc_matrix(
            (ratings, (u_indices, i_indices)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        # W stored as lil for efficient column assignment, convert to csr at end
        from scipy.sparse import lil_matrix
        W = lil_matrix((n_items, n_items), dtype=np.float32)

        en = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            positive=True,
            fit_intercept=False,
        )

        R_csc = R  # already CSC

        for j in tqdm(range(n_items), desc="SLIM fitting items"):
            y = R_csc[:, j].toarray().ravel()
            if y.sum() == 0:
                continue

            # Build X_j: all columns except j
            X_j = hstack([R_csc[:, :j], R_csc[:, j + 1:]])  # (n_users, n_items-1)

            en.fit(X_j, y)
            coef = en.coef_  # length n_items-1

            # Reinsert 0 at position j to get full n_items-length coef
            full_coef = np.concatenate([coef[:j], [0.0], coef[j:]])

            # Store as column j of W
            nz_idx = np.nonzero(full_coef)[0]
            if len(nz_idx) > 0:
                W[nz_idx, j] = full_coef[nz_idx].reshape(-1, 1)

        # Ensure diagonal is zero
        for j in range(n_items):
            W[j, j] = 0.0

        self.W = csr_matrix(W)
        self.R_train = csr_matrix(R)

        # Precompute item popularity ranking (interaction count, descending).
        # Used as fallback when co-occurrence scores are zero for sparse users.
        pop_counts = np.zeros(n_items, dtype=np.float32)
        for iid, cnt in self.item_counts.items():
            idx = item2idx.get(iid)
            if idx is not None:
                pop_counts[idx] = cnt
        self._pop_counts = pop_counts
        self._log_pop = np.log1p(pop_counts)

        # Quartile bins from least popular (0) to most popular (3), used by
        # distribution-aware reranking to avoid over-concentrating head items.
        q1, q2, q3 = np.quantile(pop_counts, [0.25, 0.50, 0.75])
        bins = np.zeros(n_items, dtype=np.int8)
        bins[pop_counts > q1] = 1
        bins[pop_counts > q2] = 2
        bins[pop_counts > q3] = 3
        self._pop_bins = bins

        self._popular_idxs = np.argsort(-pop_counts)  # most popular first

        print(f"  SLIM W matrix: {self.W.shape}, nnz={self.W.nnz}")

    # ------------------------------------------------------------------
    def recommend(self, user_id: str, seen_items: set, n: int = 10):
        """
        Return list of (item_id, score) sorted by descending score.
        When fewer than n items have positive SLIM scores (sparse users),
        the list is padded with globally-popular unseen items (score = 0.0).
        """
        scores = self._score_user(user_id, seen_items)
        if scores is None:
            return []

        # Top-n by score (only positive-score items from SLIM)
        top_idx = np.argsort(scores)[::-1][:n]
        results = []
        for i in top_idx:
            if scores[i] > 0:
                results.append((self.idx2item[i], float(scores[i])))

        # Popularity fallback: pad to n for users with insufficient co-occurrence signal
        if len(results) < n:
            in_results = {iid for iid, _ in results}
            needed = n - len(results)
            for pop_i in self._popular_idxs:
                if needed == 0:
                    break
                iid = self.idx2item[pop_i]
                if iid not in seen_items and iid not in in_results:
                    results.append((iid, 0.0))
                    in_results.add(iid)
                    needed -= 1

        return results

    def recommend_balanced(
        self,
        user_id: str,
        seen_items: set,
        n: int = 10,
        popularity_penalty: float = 0.45,
        novelty_boost: float = 0.20,
    ):
        """
        Balanced SLIM scoring that penalizes very popular items and mildly
        boosts lower-popularity items to reduce popularity amplification.
        """
        scores = self._score_user(user_id, seen_items)
        if scores is None:
            return []

        penalty = np.power(self._log_pop + 1.0, popularity_penalty)
        balanced = scores / penalty

        if novelty_boost > 0:
            pop_norm = self._pop_counts / (self._pop_counts.max() + 1e-8)
            balanced = balanced * (1.0 + novelty_boost * (1.0 - pop_norm))

        top_idx = np.argsort(balanced)[::-1][:n]
        results = []
        for i in top_idx:
            if balanced[i] > 0:
                results.append((self.idx2item[i], float(balanced[i])))

        # Backfill with less-popular unseen items first before global head items.
        if len(results) < n:
            in_results = {iid for iid, _ in results}
            needed = n - len(results)
            for pop_i in self._popular_idxs[::-1]:
                if needed == 0:
                    break
                iid = self.idx2item[pop_i]
                if iid not in seen_items and iid not in in_results:
                    results.append((iid, 0.0))
                    in_results.add(iid)
                    needed -= 1

        return results

    def recommend_distributed(
        self,
        user_id: str,
        seen_items: set,
        n: int = 10,
        popularity_penalty: float = 0.35,
        diversity_lambda: float = 0.12,
        candidate_factor: int = 8,
    ):
        """
        Distribution-aware reranking: trades off score with popularity-bin
        exposure to avoid filling top-k from only high-popularity items.
        """
        scores = self._score_user(user_id, seen_items)
        if scores is None:
            return []

        penalty = np.power(self._log_pop + 1.0, popularity_penalty)
        base = scores / penalty

        pool_size = min(len(base), max(n * candidate_factor, n))
        candidate_idx = np.argsort(base)[::-1][:pool_size]
        candidate_idx = [i for i in candidate_idx if base[i] > 0]
        if not candidate_idx:
            return self.recommend_balanced(user_id, seen_items, n=n)

        selected = []
        selected_set = set()
        bin_counts = np.zeros(4, dtype=np.int32)

        while len(selected) < n and len(selected_set) < len(candidate_idx):
            best_i, best_sc = None, -np.inf
            for i in candidate_idx:
                if i in selected_set:
                    continue
                pop_bin = self._pop_bins[i]
                mmr_score = base[i] - diversity_lambda * bin_counts[pop_bin]
                if mmr_score > best_sc:
                    best_sc = mmr_score
                    best_i = i

            if best_i is None:
                break

            selected.append((self.idx2item[best_i], float(base[best_i])))
            selected_set.add(best_i)
            bin_counts[self._pop_bins[best_i]] += 1

        if len(selected) < n:
            in_results = {iid for iid, _ in selected}
            needed = n - len(selected)
            for pop_i in self._popular_idxs[::-1]:
                if needed == 0:
                    break
                iid = self.idx2item[pop_i]
                if iid not in seen_items and iid not in in_results:
                    selected.append((iid, 0.0))
                    in_results.add(iid)
                    needed -= 1

        return selected
