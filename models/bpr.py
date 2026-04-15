"""
Bayesian Personalized Ranking (BPR) via vectorized mini-batch SGD.
Optimises pairwise ranking loss; suitable for implicit/explicit top-N recommendation.

Note: user bias (bu) is NOT included. In BPR's pairwise objective,
  x_ui - x_uj = (bu + bi[i] + Pu·Qi) - (bu + bi[j] + Pu·Qj)
              = bi[i] - bi[j] + Pu·(Qi - Qj)
bu cancels completely, so it provides zero gradient signal and no benefit.
"""
import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class BPR:
    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 50,
        lr: float = 0.01,
        reg: float = 0.01,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    def fit(self, train_df: pd.DataFrame, user2idx: dict, item2idx: dict):
        """
        Train BPR on implicit feedback.  Caller should pre-filter train_df
        to positive interactions (e.g. rating >= 4) before passing it here.
        """
        np.random.seed(self.seed)
        self.user2idx = user2idx
        self.item2idx = item2idx
        n_users = len(user2idx)
        n_items = len(item2idx)

        # ── Initialise parameters ────────────────────────────────────────────────
        self.P  = np.random.normal(0, 0.1, (n_users, self.n_factors))  # user factors
        self.Q  = np.random.normal(0, 0.1, (n_items, self.n_factors))  # item factors
        self.bi = np.zeros(n_items)                                     # item biases
        self.idx2item = {v: k for k, v in item2idx.items()}            # cached once

        # Precompute item popularity ranking for cold-start fallback
        item_counts = train_df.groupby("item_id").size()
        pop_counts  = np.zeros(n_items, dtype=np.float32)
        for iid, cnt in item_counts.items():
            idx = item2idx.get(iid)
            if idx is not None:
                pop_counts[idx] = cnt
        self._popular_idxs = np.argsort(-pop_counts)

        # ── Map training pairs to indices once ───────────────────────────────────
        u_raw  = train_df["user_id"].map(user2idx).values
        i_raw  = train_df["item_id"].map(item2idx).values
        valid  = (~pd.isna(u_raw)) & (~pd.isna(i_raw))
        u_arr  = u_raw[valid].astype(np.int64)
        i_arr  = i_raw[valid].astype(np.int64)
        n_pairs = len(u_arr)
        batch_size = 512

        for epoch in range(self.n_epochs):
            order  = np.random.permutation(n_pairs)
            u_shuf = u_arr[order]
            i_shuf = i_arr[order]
            j_shuf = np.random.randint(0, n_items, size=n_pairs)

            total_loss = 0.0
            n_batches  = int(np.ceil(n_pairs / batch_size))

            for b in range(n_batches):
                s = b * batch_size
                e = min(s + batch_size, n_pairs)

                u_b = u_shuf[s:e]
                i_b = i_shuf[s:e]
                j_b = j_shuf[s:e]

                x_ui  = self.bi[i_b] + np.einsum("bf,bf->b", self.P[u_b], self.Q[i_b])
                x_uj  = self.bi[j_b] + np.einsum("bf,bf->b", self.P[u_b], self.Q[j_b])
                xuij  = np.clip(x_ui - x_uj, -30.0, 30.0)

                sig   = _sigmoid(xuij)
                err   = 1.0 - sig
                total_loss += -np.log(sig + 1e-10).sum()

                err_col = err[:, None]

                dP  =  err_col * (self.Q[i_b] - self.Q[j_b]) - self.reg * self.P[u_b]
                dQi =  err_col *  self.P[u_b]                 - self.reg * self.Q[i_b]
                dQj = -err_col *  self.P[u_b]                 - self.reg * self.Q[j_b]
                dbi =  err                                     - self.reg * self.bi[i_b]
                dbj = -err                                     - self.reg * self.bi[j_b]

                np.add.at(self.P,  u_b, self.lr * dP)
                np.add.at(self.Q,  i_b, self.lr * dQi)
                np.add.at(self.Q,  j_b, self.lr * dQj)
                np.add.at(self.bi, i_b, self.lr * dbi)
                np.add.at(self.bi, j_b, self.lr * dbj)

            avg_loss = total_loss / n_pairs
            print(f"  BPR Epoch {epoch+1}/{self.n_epochs}  Avg BPR Loss: {avg_loss:.4f}")

    def recommend(self, user_id: str, seen_items: set, n: int = 10) -> list:
        """
        Return top-n (item_id, score) for unseen items.
        Cold-start users (unknown user_id) get popularity-ranked fallback.
        """
        if not hasattr(self, 'idx2item'):
            self.idx2item = {v: k for k, v in self.item2idx.items()}
        u = self.user2idx.get(user_id)

        # Cold-start fallback: user never seen in training → return popular items
        if u is None:
            results = []
            for pop_i in self._popular_idxs:
                if len(results) == n:
                    break
                iid = self.idx2item[pop_i]
                if iid not in seen_items:
                    results.append((iid, 0.0))
            return results

        item_idxs = np.array(
            [idx for iid, idx in self.item2idx.items() if iid not in seen_items],
            dtype=np.int64,
        )
        if len(item_idxs) == 0:
            return []

        scores    = self.bi[item_idxs] + self.Q[item_idxs] @ self.P[u]
        top_order = np.argsort(-scores)[:n]
        return [(self.idx2item[item_idxs[o]], float(scores[o])) for o in top_order]
