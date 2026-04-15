"""
SVD++ model — extends biased MF with implicit feedback.

Predicted rating:
  x_ui = μ + b_u + b_i + q_i · (p_u + |N(u)|^{-0.5} * Σ_{j∈N(u)} y_j)

N(u) is restricted to items the user rated ≥ 4★.  Using all rated items
(including 1-star) as implicit positives pollutes the signal — the model
learns "user engaged with bad items too", which adds noise and is why SVD++
beat plain MF less than expected before this fix.
"""
import numpy as np
import pandas as pd


class SVDpp:
    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 50,
        lr: float = 0.005,
        reg: float = 0.015,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    def fit(self, train_df: pd.DataFrame, user2idx: dict, item2idx: dict):
        np.random.seed(self.seed)
        self.user2idx = user2idx
        self.item2idx = item2idx
        n_users = len(user2idx)
        n_items = len(item2idx)

        self.global_mean = float(train_df["rating"].mean())
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.pu = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.qi = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.y  = np.random.normal(0, 0.1, (n_items, self.n_factors))

        u_arr = train_df["user_id"].map(user2idx).values
        i_arr = train_df["item_id"].map(item2idx).values
        r_arr = train_df["rating"].values.astype(np.float64)
        n_samples = len(r_arr)

        # N(u): only positively-rated items (≥4★) as implicit feedback.
        # Including 1-star reviews as "implicit likes" pollutes the signal.
        pos_mask = r_arr >= 4.0
        Nu: dict[int, list] = {}
        for u_idx, i_idx in zip(u_arr[pos_mask], i_arr[pos_mask]):
            Nu.setdefault(int(u_idx), []).append(int(i_idx))
        Nu = {u: np.array(v, dtype=np.int64) for u, v in Nu.items()}
        self._Nu = Nu  # saved for inference cache

        Nu_sqrt_inv = {
            u: (1.0 / np.sqrt(len(items)) if len(items) > 0 else 0.0)
            for u, items in Nu.items()
        }

        for epoch in range(self.n_epochs):
            # Pre-compute sum_y for each user
            sum_y = np.zeros((n_users, self.n_factors))
            for u_idx, items_list in Nu.items():
                if len(items_list) > 0:
                    sum_y[u_idx] = Nu_sqrt_inv[u_idx] * self.y[items_list].sum(axis=0)

            order = np.random.permutation(n_samples)
            total_sq_err = 0.0

            for idx in order:
                u = int(u_arr[idx])
                i = int(i_arr[idx])
                r = r_arr[idx]
                implicit_u = sum_y[u]

                pred = (self.global_mean + self.bu[u] + self.bi[i]
                        + self.qi[i] @ (self.pu[u] + implicit_u))
                err  = r - pred
                total_sq_err += err * err

                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                pu_old = self.pu[u].copy()
                self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                self.qi[i] += self.lr * (err * (pu_old + implicit_u) - self.reg * self.qi[i])

                if u in Nu and len(Nu[u]) > 0:
                    sqrt_inv = Nu_sqrt_inv[u]
                    self.y[Nu[u]] += self.lr * (
                        err * sqrt_inv * self.qi[i] - self.reg * self.y[Nu[u]]
                    )

            train_rmse = np.sqrt(total_sq_err / n_samples)
            print(f"  SVD++ Epoch {epoch+1}/{self.n_epochs}  Train RMSE: {train_rmse:.4f}")

    def predict_one(self, user_id: str, item_id: str) -> float:
        u = self.user2idx.get(user_id)
        i = self.item2idx.get(item_id)
        if u is None or i is None:
            return self.global_mean

        if hasattr(self, '_sum_y_cache') and self._sum_y_cache is not None:
            implicit_u = self._sum_y_cache[u]
        else:
            implicit_u = self._compute_implicit(u)

        pred = (self.global_mean + self.bu[u] + self.bi[i]
                + self.qi[i] @ (self.pu[u] + implicit_u))
        return float(np.clip(pred, 1.0, 5.0))

    def _compute_implicit(self, u_idx: int) -> np.ndarray:
        items_list = self._Nu.get(u_idx) if hasattr(self, '_Nu') else None
        if items_list is None or len(items_list) == 0:
            return np.zeros(self.n_factors)
        return (1.0 / np.sqrt(len(items_list))) * self.y[items_list].sum(axis=0)

    def build_inference_cache(self, train_df: pd.DataFrame):
        """Build cached sum_y for fast batch prediction after training.
        Uses the same ≥4★ positive filter as fit()."""
        u_arr = train_df["user_id"].map(self.user2idx).values
        i_arr = train_df["item_id"].map(self.item2idx).values
        r_arr = train_df["rating"].values.astype(np.float64)

        # Rebuild N(u) with positive-only filter (must match fit())
        pos_mask = r_arr >= 4.0
        Nu: dict[int, list] = {}
        for u_idx, i_idx in zip(u_arr[pos_mask], i_arr[pos_mask]):
            Nu.setdefault(int(u_idx), []).append(int(i_idx))
        self._Nu = {u: np.array(v, dtype=np.int64) for u, v in Nu.items()}

        n_users = len(self.user2idx)
        cache   = np.zeros((n_users, self.n_factors))
        for u_idx, items_list in self._Nu.items():
            if len(items_list) > 0:
                cache[u_idx] = (1.0 / np.sqrt(len(items_list))) * self.y[items_list].sum(axis=0)
        self._sum_y_cache = cache

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        u_raw = df["user_id"].map(self.user2idx).values
        i_raw = df["item_id"].map(self.item2idx).values
        preds = np.full(len(df), self.global_mean, dtype=np.float64)
        valid = (~pd.isna(u_raw)) & (~pd.isna(i_raw))
        if valid.any():
            u_v = u_raw[valid].astype(np.int64)
            i_v = i_raw[valid].astype(np.int64)
            if hasattr(self, '_sum_y_cache') and self._sum_y_cache is not None:
                implicit = self._sum_y_cache[u_v]
            else:
                implicit = np.zeros((int(valid.sum()), self.n_factors))
            preds[valid] = np.clip(
                self.global_mean + self.bu[u_v] + self.bi[i_v]
                + np.sum(self.qi[i_v] * (self.pu[u_v] + implicit), axis=1),
                1.0, 5.0,
            )
        return preds
