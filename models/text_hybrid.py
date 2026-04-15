"""
TF-IDF content-based reranker for hybrid recommendations.

Builds a user profile from their reviewed items' TF-IDF vectors and
rescores BPR candidates by blending BPR score with content similarity.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack


class TextHybrid:
    def __init__(self, max_features: int = 5000, alpha: float = 0.3):
        """
        Parameters
        ----------
        max_features : int
            Maximum number of TF-IDF features.
        alpha : float
            Weight for content score; (1 - alpha) is weight for BPR score.
        """
        self.max_features = max_features
        self.alpha = alpha
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
        )

    def fit(self, train_df: pd.DataFrame):
        """
        Build TF-IDF matrix over items using their reviews from train_df.

        Item text = concatenation of all reviews for that item.
        """
        # Aggregate reviews per item; replace NaN with empty string
        item_reviews = (
            train_df.assign(review=train_df["review"].fillna(""))
            .groupby("item_id")["review"]
            .apply(lambda texts: " ".join(texts))
            .reset_index()
        )

        self.item_ids = item_reviews["item_id"].tolist()
        self._item_id_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}

        corpus = item_reviews["review"].tolist()
        self.item_tfidf = self.vectorizer.fit_transform(corpus)  # sparse (n_items × max_features)

    def rerank(
        self,
        user_id: str,
        bpr_candidates: list,
        user_items_dict: dict,
        n: int = 10,
    ) -> list:
        """
        Rerank SLIM candidates using content-based similarity.

        Parameters
        ----------
        user_id : str
            The user to rerank for.
        bpr_candidates : list of (item_id, score)
            Candidate list from SLIM.recommend() (or BPR).
        user_items_dict : dict  {user_id -> list[item_id]}
            Pre-built lookup of training items per user.  Pass
            ``train_df.groupby("user_id")["item_id"].apply(list).to_dict()``
            once before the evaluation loop — avoids an O(n_train) scan per user.
        n : int
            Number of items to return.

        Returns
        -------
        list of (item_id, final_score)
        """
        if not bpr_candidates:
            return []

        # ── 1. Build user profile ─────────────────────────────────────────────
        user_items = user_items_dict.get(user_id, [])

        user_profile = None
        valid_indices = [
            self._item_id_to_idx[iid]
            for iid in user_items
            if iid in self._item_id_to_idx
        ]
        if valid_indices:
            # Mean of TF-IDF vectors (sparse rows)
            profile_matrix = self.item_tfidf[valid_indices]
            # Convert to dense for mean computation, then back
            user_profile = np.asarray(profile_matrix.mean(axis=0))  # (1, max_features)

        # ── 2. Compute content scores for each candidate ──────────────────────
        candidate_ids = [iid for iid, _ in bpr_candidates]
        bpr_scores_arr = np.array([score for _, score in bpr_candidates], dtype=np.float64)

        content_scores = np.zeros(len(candidate_ids), dtype=np.float64)
        if user_profile is not None:
            for idx, iid in enumerate(candidate_ids):
                item_idx = self._item_id_to_idx.get(iid)
                if item_idx is not None:
                    item_vec = self.item_tfidf[item_idx]
                    # cosine_similarity expects 2-D arrays
                    sim = cosine_similarity(user_profile, item_vec)[0, 0]
                    content_scores[idx] = float(sim)

        # ── 3. Normalize SLIM/BPR scores to [0, 1] via min-max ──────────────────
        # Min-max preserves the full relative ordering across all candidates.
        bpr_min = bpr_scores_arr.min()
        bpr_max = bpr_scores_arr.max()
        bpr_range = bpr_max - bpr_min
        if bpr_range > 1e-10:
            bpr_norm = (bpr_scores_arr - bpr_min) / bpr_range
        else:
            bpr_norm = np.zeros_like(bpr_scores_arr)

        # ── 4. Combine scores ─────────────────────────────────────────────────
        final_scores = (1.0 - self.alpha) * bpr_norm + self.alpha * content_scores

        # ── 5. Return top-n by final score ────────────────────────────────────
        top_order = np.argsort(-final_scores)[:n]
        return [(candidate_ids[o], float(final_scores[o])) for o in top_order]
