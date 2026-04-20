"""
Flask demo — Amazon Video Games Recommender.
Run:  python app.py  →  http://127.0.0.1:5001

Models and data are loaded once at startup.  All tabs are served from the same
process; heavy endpoints (fairness, sampled-eval) are computed on first request
and cached in memory for the lifetime of the server.
"""
import importlib
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from sklearn.metrics.pairwise import cosine_similarity

# ── Backward-compat shims ─────────────────────────────────────────────────────
for _old, _new in [
    ("matrix_factorization", "models.matrix_factorization"),
    ("svdpp",                "models.svdpp"),
    ("bpr",                  "models.bpr"),
    ("slim",                 "models.slim"),
    ("text_hybrid",          "models.text_hybrid"),
]:
    try:
        sys.modules[_old] = importlib.import_module(_new)
    except ImportError:
        pass

app       = Flask(__name__)
ARTIFACTS = "artifacts"
TOP_K     = 10

# ── Load artifacts once ───────────────────────────────────────────────────────
print("Loading models and data…")

def _load(fname):
    p = os.path.join(ARTIFACTS, fname)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _load_json(fname, default=None):
    p = os.path.join(ARTIFACTS, fname)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)

MF_MODEL          = _load("model.pkl")
SVDPP_MODEL       = _load("svdpp_model.pkl")
BPR_MODEL         = _load("bpr_model.pkl")
SLIM_MODEL        = _load("slim_model.pkl")
TEXT_HYBRID_MODEL = _load("text_hybrid.pkl")

TRAIN_DF = pd.read_csv(os.path.join(ARTIFACTS, "train.csv"))
TEST_DF  = pd.read_csv(os.path.join(ARTIFACTS, "test.csv"))

with open(os.path.join(ARTIFACTS, "results.json")) as f:
    RESULTS = json.load(f)
RATING_BUNDLE = _load_json("rating_bundle.json", default={})

# Pre-computed lookups
ITEM_COUNTS = TRAIN_DF.groupby("item_id").size().to_dict()
ITEM_TITLES = (
    pd.concat([TRAIN_DF, TEST_DF])[["item_id", "title"]]
    .dropna()
    .drop_duplicates("item_id")
    .set_index("item_id")["title"]
    .to_dict()
)
ITEM_REVIEWS = (
    TRAIN_DF.dropna(subset=["review"])
    .groupby("item_id")["review"]
    .apply(lambda x: x.drop_duplicates().head(3).tolist())
    .to_dict()
)
TRAIN_SEEN       = TRAIN_DF.groupby("user_id")["item_id"].apply(set).to_dict()
TEST_ITEMS       = TEST_DF.groupby("user_id")["item_id"].apply(set).to_dict()
ALL_ITEMS        = set(TRAIN_DF["item_id"].unique())
# Pre-built for TextHybrid / content scoring — avoids O(n_train) scan per request
USER_ITEMS_DICT  = TRAIN_DF.groupby("user_id")["item_id"].apply(list).to_dict()
ALL_USERS   = sorted(MF_MODEL.user2idx.keys()) if MF_MODEL else []
ACTIVE_USERS = (
    TRAIN_DF["user_id"].value_counts()
    .loc[lambda x: x >= 10]
    .index.tolist()[:300]
)

# Lazy cache for expensive computations
_fairness_cache     = None
_sampled_eval_cache = None

print(f"  {len(ALL_USERS):,} users  ·  {len(ALL_ITEMS):,} items  ·  {len(TRAIN_DF):,} train ratings")
print("Ready at http://127.0.0.1:5001")


# ─────────────────────────────────────────────────────────────────────────────
# Score decomposition helpers  (borrowed concept from friend's hybrid approach)
# ─────────────────────────────────────────────────────────────────────────────

def _dynamic_alpha(n_history: int) -> float:
    """CF weight grows with history size: cold-start → content, power-user → CF."""
    if n_history <= 5:
        return 0.30
    elif n_history <= 20:
        return 0.50
    else:
        return 0.70


def _compute_content_score(user_id: str, item_id: str):
    """
    Return (cosine_sim, content_score_in_[1,5]) using the TF-IDF model.
    content_score = global_mean + sim * (5 - global_mean)
    so a perfect match yields 5.0 and no match yields global_mean.
    """
    if TEXT_HYBRID_MODEL is None or MF_MODEL is None:
        return 0.0, float(MF_MODEL.global_mean if MF_MODEL else 3.5)

    item_idx = TEXT_HYBRID_MODEL._item_id_to_idx.get(item_id)
    if item_idx is None:
        return 0.0, float(MF_MODEL.global_mean)

    user_items = USER_ITEMS_DICT.get(user_id, [])
    valid_idxs = [
        TEXT_HYBRID_MODEL._item_id_to_idx[iid]
        for iid in user_items
        if iid in TEXT_HYBRID_MODEL._item_id_to_idx
    ]
    if not valid_idxs:
        return 0.0, float(MF_MODEL.global_mean)

    user_profile = np.asarray(TEXT_HYBRID_MODEL.item_tfidf[valid_idxs].mean(axis=0))
    item_vec     = TEXT_HYBRID_MODEL.item_tfidf[item_idx]
    sim          = float(cosine_similarity(user_profile, item_vec)[0, 0])
    gm           = float(MF_MODEL.global_mean)
    content_score = float(np.clip(gm + sim * (5.0 - gm), 1.0, 5.0))
    return sim, content_score


def _apply_calibration(score: float, key: str) -> float:
    calib = RATING_BUNDLE.get("calibration", {})
    params = calib.get(key, {})
    a = float(params.get("a", 1.0))
    b = float(params.get("b", 0.0))
    return float(np.clip(a * score + b, 1.0, 5.0))


def _predict_task_a_score(user_id: str, item_id: str, model_pref: str = "auto"):
    mf_raw = float(MF_MODEL.predict_one(user_id, item_id)) if MF_MODEL else 3.5
    svd_raw = float(SVDPP_MODEL.predict_one(user_id, item_id)) if SVDPP_MODEL else mf_raw
    mf_cal = _apply_calibration(mf_raw, "mf")
    svd_cal = _apply_calibration(svd_raw, "svdpp")
    alpha = float(RATING_BUNDLE.get("calibration", {}).get("blend_alpha", 1.0))
    blend = float(np.clip(alpha * mf_cal + (1.0 - alpha) * svd_cal, 1.0, 5.0))

    pref = (model_pref or "auto").lower()
    if pref == "mf":
        return mf_cal, "MF"
    if pref in {"svdpp", "svd++"}:
        return svd_cal, "SVD++"
    if pref == "blend":
        return blend, "MF/SVD++ blend"

    best = RATING_BUNDLE.get("best_rating_model", "")
    if "blend" in best.lower():
        return blend, "MF/SVD++ blend"
    if "SVD++" in best:
        return svd_cal, "SVD++"
    return mf_cal, "MF"


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video Games Recommender</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f0f4f8;color:#1a202c;min-height:100vh}
header{background:linear-gradient(135deg,#232f3e 0%,#37475a 100%);color:#fff;
       padding:18px 40px;box-shadow:0 2px 8px rgba(0,0,0,.3)}
header h1{font-size:1.5rem;font-weight:700}
header p{font-size:.85rem;opacity:.7;margin-top:3px}

/* Tabs */
.tab-nav{display:flex;gap:0;background:#1a2535;padding:0 40px}
.tab-btn{background:none;border:none;color:#a0aec0;padding:12px 22px;font-size:.9rem;
         cursor:pointer;border-bottom:3px solid transparent;transition:.2s}
.tab-btn:hover{color:#fff}
.tab-btn.active{color:#ff9900;border-bottom-color:#ff9900;font-weight:600}
.tab-pane{display:none}.tab-pane.active{display:block}

.container{max-width:1100px;margin:0 auto;padding:28px 24px}
.panel{background:#fff;border-radius:12px;padding:26px;
       box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:24px}
.panel h2{font-size:1.05rem;font-weight:600;margin-bottom:16px;color:#2d3748}
.panel h3{font-size:.95rem;font-weight:600;margin:20px 0 12px;color:#2d3748}

/* Metric grid */
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
              gap:14px;margin-bottom:24px}
.metric-card{background:#fff;border-radius:10px;padding:18px 20px;
             box-shadow:0 1px 4px rgba(0,0,0,.08)}
.metric-card .lbl{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:#718096}
.metric-card .val{font-size:1.7rem;font-weight:700;color:#2d3748;margin-top:2px}
.metric-card .sub{font-size:.76rem;color:#a0aec0;margin-top:2px}
.metric-card.best .val{color:#276749}

/* Forms */
.form-row{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap}
label{display:block;font-size:.78rem;font-weight:600;color:#4a5568;margin-bottom:5px}
select,input{border:1px solid #e2e8f0;border-radius:8px;padding:9px 12px;font-size:.88rem;
             background:#f7fafc;color:#2d3748;outline:none;transition:.2s}
select{min-width:200px}
select:focus,input:focus{border-color:#ff9900;box-shadow:0 0 0 3px rgba(255,153,0,.15)}
button.primary{background:#ff9900;color:#fff;border:none;border-radius:8px;
               padding:10px 20px;font-size:.88rem;font-weight:600;cursor:pointer;transition:.2s}
button.primary:hover{background:#e68900;transform:translateY(-1px)}
button.secondary{background:#718096;color:#fff;border:none;border-radius:8px;
                 padding:10px 20px;font-size:.88rem;font-weight:600;cursor:pointer;transition:.2s}
button.secondary:hover{background:#4a5568}

/* Rec list */
.rec-item{border-bottom:1px solid #edf2f7;padding:14px 0}
.rec-item:last-child{border:none}
.rec-header{display:flex;align-items:flex-start;gap:14px}
.rank-badge{width:30px;height:30px;border-radius:50%;background:#ff9900;color:#fff;
            font-weight:700;font-size:.8rem;display:flex;align-items:center;
            justify-content:center;flex-shrink:0}
.rank-badge.top3{background:#232f3e}
.rec-body{flex:1}
.rec-title{font-weight:600;font-size:.92rem;color:#2d3748}
.rec-meta{font-size:.76rem;color:#a0aec0;margin-top:3px}
.rec-right{text-align:right;flex-shrink:0}
.stars{color:#ff9900;font-size:1rem;letter-spacing:1px}
.stars .empty{color:#e2e8f0}
.score-val{font-size:.85rem;font-weight:600;color:#2d3748}
.tag{display:inline-block;padding:1px 7px;border-radius:3px;font-size:.68rem;
     font-weight:600;text-transform:uppercase;background:#e6fffa;color:#234e52;margin-left:6px}

/* Reviews toggle */
.reviews-toggle{font-size:.76rem;color:#3182ce;cursor:pointer;margin-top:6px;
                display:inline-block;user-select:none}
.reviews-block{display:none;margin-top:8px;border-left:3px solid #e2e8f0;padding-left:12px}
.reviews-block p{font-size:.8rem;color:#4a5568;margin-bottom:6px;line-height:1.5}

/* Predict result */
.pred-result{margin-top:14px;font-size:1rem;font-weight:600;background:#fff8e6;
             border:1px solid #fcd34d;border-radius:8px;padding:12px 16px;display:none}
.model-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.7rem;
             font-weight:600;background:#ebf4ff;color:#2b6cb0;margin-left:6px}

/* Tables */
table.data-table{width:100%;border-collapse:collapse;font-size:.85rem}
table.data-table th{background:#f7fafc;color:#4a5568;font-weight:600;text-align:left;
                    padding:9px 12px;border-bottom:2px solid #e2e8f0}
table.data-table td{padding:9px 12px;border-bottom:1px solid #edf2f7;color:#2d3748}
table.data-table tr:last-child td{border:none}
table.data-table td.best{color:#276749;font-weight:700}
table.data-table td.num{text-align:right;font-variant-numeric:tabular-nums}

/* Bar chart (CSS only) */
.bar-wrap{margin:12px 0}
.bar-label{font-size:.8rem;color:#4a5568;margin-bottom:4px;display:flex;justify-content:space-between}
.bar-track{background:#edf2f7;border-radius:4px;height:18px;position:relative;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;transition:width .5s ease;display:flex;
          align-items:center;padding-left:8px;font-size:.72rem;color:#fff;font-weight:600}
.bar-fill.neutral{background:#4299e1}
.bar-fill.good{background:#48bb78}
.bar-fill.warn{background:#ed8936}
.ref-line{position:absolute;top:0;bottom:0;width:2px;background:rgba(0,0,0,.25)}

/* Fairness group table */
.group-bar{height:12px;background:#4299e1;border-radius:3px;display:inline-block;vertical-align:middle;margin-right:8px}

/* Explainability */
.explain-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin:16px 0}
.explain-card{border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center}
.explain-card .model-name{font-size:.75rem;font-weight:600;color:#718096;text-transform:uppercase;margin-bottom:8px}
.explain-card .big-val{font-size:1.6rem;font-weight:700;color:#2d3748}
.explain-card .sub-val{font-size:.8rem;color:#a0aec0;margin-top:4px}
.explain-card.highlight{border-color:#ff9900;background:#fff8e6}

/* Loading spinner */
.spinner{display:inline-block;width:16px;height:16px;border:2px solid #e2e8f0;
         border-top-color:#ff9900;border-radius:50%;animation:spin .7s linear infinite;
         vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-msg{color:#718096;font-size:.9rem;margin:20px 0}

@media(max-width:640px){.metrics-grid{grid-template-columns:1fr 1fr}
  .explain-grid{grid-template-columns:1fr}.tab-btn{padding:10px 12px;font-size:.8rem}}
</style>
</head>
<body>

<header>
  <h1>Amazon Video Games Recommender</h1>
  <p>{{ n_users|format_num }} users &nbsp;·&nbsp; {{ n_items|format_num }} items &nbsp;·&nbsp;
     {{ n_train|format_num }} training ratings &nbsp;·&nbsp;
     Models: MF · SVD++ · BPR · SLIM · TextHybrid</p>
</header>

<nav class="tab-nav">
  <button class="tab-btn active" onclick="showTab('recs')">Recommendations</button>
  <button class="tab-btn" onclick="showTab('eval')">Evaluation</button>
  <button class="tab-btn" onclick="showTab('fairness')">Fairness</button>
  <button class="tab-btn" onclick="showTab('explain')">Explainability</button>
</nav>

<!-- ══════════════════════════════════════════════════════════════════════════
     TAB 1 — RECOMMENDATIONS
══════════════════════════════════════════════════════════════════════════ -->
<div id="tab-recs" class="tab-pane active">
<div class="container">

  <div class="metrics-grid">
    <div class="metric-card">
      <div class="lbl">MAE — MF</div>
      <div class="val">{{ results.rating_prediction_mf.MAE|r4 }}</div>
      <div class="sub">Task A best model</div>
    </div>
    <div class="metric-card">
      <div class="lbl">RMSE — MF</div>
      <div class="val">{{ results.rating_prediction_mf.RMSE|r4 }}</div>
      <div class="sub">Task A best model</div>
    </div>
    <div class="metric-card best">
      <div class="lbl">NDCG@10 — SLIM</div>
      <div class="val">{{ results.recommendation_slim["NDCG@K"]|r4 }}</div>
      <div class="sub">Task B best model</div>
    </div>
    <div class="metric-card">
      <div class="lbl">NDCG@10 — BPR</div>
      <div class="val">{{ results.recommendation_bpr["NDCG@K"]|r4 }}</div>
      <div class="sub">Pairwise ranking baseline</div>
    </div>
    <div class="metric-card">
      <div class="lbl">P@10 — SLIM</div>
      <div class="val">{{ (results.recommendation_slim["Precision@K"]*100)|r2 }}%</div>
      <div class="sub">Task B best model</div>
    </div>
  </div>

  <div class="panel">
    <h2>Top-10 Recommendations</h2>
    <div class="form-row">
      <div>
        <label>User ID</label>
        <div style="display:flex;gap:8px">
          <input id="uid-input" list="uid-list" placeholder="Type or select a User ID"
                 style="min-width:260px" autocomplete="off">
          <datalist id="uid-list">
            {% for u in active_users %}
            <option value="{{ u }}">
            {% endfor %}
          </datalist>
          <button class="secondary" onclick="randomUser()">Random</button>
        </div>
      </div>
      <div>
        <label>Recommender model</label>
        <select id="rec-model">
          <option value="slim">SLIM (best Task-B)</option>
          <option value="hybrid">TextHybrid (SLIM + TF-IDF)</option>
          <option value="mf">MF (rating-based ranking)</option>
          <option value="bpr">BPR (pairwise baseline)</option>
        </select>
      </div>
      <button class="primary" onclick="getRecommendations()">Recommend</button>
    </div>
    <div id="rec-list" style="margin-top:16px"></div>
  </div>

  <div class="panel" id="history-panel" style="display:none">
    <h2>User's rated items</h2>
    <table class="data-table" id="history-table">
      <thead><tr><th>#</th><th>Item ID</th><th>Title</th><th>Rating</th><th>Reviews</th></tr></thead>
      <tbody id="history-body"></tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Predict a Rating</h2>
    <div class="form-row">
      <div><label>User ID</label>
           <input id="pred-user" placeholder="A31Y9ELLA1JUB0" style="min-width:220px"></div>
      <div><label>Item ID (ASIN)</label>
           <input id="pred-item" placeholder="B000F4YN2K" style="min-width:160px"></div>
      <div><label>Model</label>
           <select id="pred-model">
             <option value="mf">MF (best Task-A)</option>
             <option value="svdpp">SVD++ (implicit feedback)</option>
           </select></div>
      <button class="primary" onclick="predictRating()">Predict</button>
    </div>
    <div class="pred-result" id="pred-result"></div>
  </div>

</div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════════
     TAB 2 — EVALUATION
══════════════════════════════════════════════════════════════════════════ -->
<div id="tab-eval" class="tab-pane">
<div class="container">

  <div class="panel">
    <h2>Task A — Rating Prediction &nbsp;<small style="font-weight:400;color:#718096">(lower MAE/RMSE is better)</small></h2>
    <table class="data-table">
      <thead><tr><th>Model</th><th class="num">MAE</th><th class="num">RMSE</th><th>Notes</th></tr></thead>
      <tbody>
        {% set ta = [
          ("MF",    "MF [primary — biased SGD-MF]",      results.rating_prediction_mf),
          ("SVD++", "SVD++ (4★/5★ implicit feedback)",   results.rating_prediction_svdpp)
        ] %}
        {% set best_mae  = [results.rating_prediction_mf.MAE,
                            results.rating_prediction_svdpp.MAE]|min %}
        {% set best_rmse = [results.rating_prediction_mf.RMSE,
                            results.rating_prediction_svdpp.RMSE]|min %}
        {% for key, label, r in ta %}
        <tr>
          <td>{{ label }}</td>
          <td class="num {{ 'best' if r.MAE == best_mae else '' }}">{{ r.MAE|r4 }}</td>
          <td class="num {{ 'best' if r.RMSE == best_rmse else '' }}">{{ r.RMSE|r4 }}</td>
          <td style="font-size:.8rem;color:#718096">
            {% if key == "SVD++" %}MF + implicit-feedback signal (only positively-rated items)
            {% else %}Best Task-A model: strong item/user biases dominate on Video Games{% endif %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Task B — Top-10 Recommendation &nbsp;<small style="font-weight:400;color:#718096">(higher is better)</small></h2>
    <table class="data-table">
      <thead><tr><th>Model</th><th class="num">P@10</th><th class="num">R@10</th>
                 <th class="num">F-measure</th><th class="num">NDCG@10</th><th>Notes</th></tr></thead>
      <tbody>
        {% set hybrid_r = results.get("recommendation_hybrid") %}
        {% set tb = [
          ("BPR",    "BPR (baseline — 4★/5★ implicit)",          results.recommendation_bpr),
          ("SLIM",   "SLIM [primary — item-item ElasticNet]",    results.recommendation_slim),
          ("Hybrid", "TextHybrid (SLIM + TF-IDF rerank)",        hybrid_r or results.recommendation_slim)
        ] %}
        {% set best_ndcg = [results.recommendation_bpr["NDCG@K"],
                            results.recommendation_slim["NDCG@K"],
                            (hybrid_r or results.recommendation_slim)["NDCG@K"]]|max %}
        {% for key, label, r in tb %}
        <tr>
          <td>{{ label }}</td>
          <td class="num">{{ r["Precision@K"]|r4 }}</td>
          <td class="num">{{ r["Recall@K"]|r4 }}</td>
          <td class="num">{{ r["F-measure"]|r4 }}</td>
          <td class="num {{ 'best' if r['NDCG@K'] == best_ndcg else '' }}">{{ r["NDCG@K"]|r4 }}</td>
          <td style="font-size:.8rem;color:#718096">
            {% if key == "SLIM" %}Best Task-B: item-item co-purchase patterns (franchise/platform)
            {% elif key == "Hybrid" %}SLIM top-3N pool reranked by TF-IDF review cosine similarity
            {% else %}Pairwise ranking on 4★/5★ interactions only{% endif %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    <p style="font-size:.78rem;color:#a0aec0;margin-top:12px">
      Full-catalogue evaluation: recommendations ranked against all unseen items.
      Absolute values are low because Video Games is a long-tail catalogue —
      relative ordering between models is what matters.
    </p>
  </div>

  <div class="panel">
    <h2>Sampled Evaluation &nbsp;<small style="font-weight:400;color:#718096">
      (99 random negatives per user — standard in literature)</small></h2>
    <p style="font-size:.85rem;color:#4a5568;margin-bottom:16px">
      For each user: rank test positives against 99 randomly sampled negatives.
      Numbers are higher than full-catalogue evaluation — this is expected and
      directly comparable with published recommendation benchmarks.
    </p>
    <button class="primary" onclick="runSampledEval()">Run (50 users · ~30 sec)</button>
    <div id="sampled-result" style="margin-top:16px"></div>
  </div>

</div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════════
     TAB 3 — FAIRNESS
══════════════════════════════════════════════════════════════════════════ -->
<div id="tab-fairness" class="tab-pane">
<div class="container">
  <div class="panel">
    <h2>Fairness &amp; Bias Analysis</h2>
    <p style="font-size:.85rem;color:#4a5568;margin-bottom:16px">
      Evaluates popularity bias (do we over-recommend popular items?),
      catalogue coverage (how many items does the system surface?),
      and user-group fairness parity (P@10 / R@10 / NDCG@10 across activity cohorts).
      Based on SLIM recommendations for a stratified sample of up to 300 users.
    </p>
    <button class="primary" onclick="runFairness()">Run Analysis</button>
    <div id="fairness-result" style="margin-top:20px"></div>
  </div>
</div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════════
     TAB 4 — EXPLAINABILITY
══════════════════════════════════════════════════════════════════════════ -->
<div id="tab-explain" class="tab-pane">
<div class="container">
  <div class="panel">
    <h2>Score Explainability</h2>
    <p style="font-size:.85rem;color:#4a5568;margin-bottom:16px">
      Select a user and any unseen item to see: predicted ratings from all three
      Task-A models, the item's rank in each Task-B recommender, and a
      counterfactual table showing which model would have recommended it.
    </p>
    <div class="form-row">
      <div>
        <label>User ID</label>
        <select id="exp-user" style="min-width:250px" onchange="loadUnseenItems()">
          {% for u in active_users[:100] %}
          <option value="{{ u }}">{{ u }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label>Unseen Item (ASIN)</label>
        <select id="exp-item" style="min-width:200px">
          <option>Loading…</option>
        </select>
      </div>
      <button class="primary" onclick="runExplain()">Explain</button>
    </div>

    <details style="margin-top:16px;font-size:.83rem;color:#4a5568">
      <summary style="cursor:pointer;font-weight:600">How scores are computed</summary>
      <pre style="margin-top:10px;background:#f7fafc;padding:14px;border-radius:8px;
                  line-height:1.7;overflow-x:auto">
Task A — Rating Prediction
  MF [primary] : global_mean + user_bias + item_bias + P_u · Q_i
  SVD++        : MF + 4★/5★ implicit feedback (|N(u)|^-0.5 · Σ y_j · Q_i)

Task B — Top-N Ranking
  BPR  [baseline] : item_bias + P_u · Q_i  (trained on 4★/5★ only)
  SLIM [primary]  : r_u @ W  (sparse item-item ElasticNet, full catalogue)
  TextHybrid      : SLIM top-3N pool reranked by TF-IDF cosine similarity

Score Decomposition (Explainability tab)
  CF score     : MF prediction for user-item pair
  Content score: global_mean + TF-IDF_cosine_sim × (5 − global_mean)
  Blend        : α×CF + β×Content  (α grows 0.30→0.70 with history size)</pre>
    </details>

    <div id="explain-result" style="margin-top:20px"></div>
  </div>
</div>
</div>


<!-- ══════════════════════════════════════════════════════════════════════════
     JAVASCRIPT
══════════════════════════════════════════════════════════════════════════ -->
<script>
const ALL_USERS = {{ active_users | tojson }};

// ── Tab switching ─────────────────────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.currentTarget.classList.add('active');
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function starsHTML(score) {
  const n = Math.round(Math.min(5, Math.max(1, score)));
  return Array.from({length:5}, (_,i) =>
    `<span class="${i < n ? '' : 'empty'}">&#9733;</span>`
  ).join('');
}

function loading(msg='Loading…') {
  return `<div class="loading-msg"><span class="spinner"></span>${msg}</div>`;
}

function fmtNum(n) {
  return n.toLocaleString();
}

// ── Tab 1: Recommendations ────────────────────────────────────────────────────
function randomUser() {
  const uid = ALL_USERS[Math.floor(Math.random() * ALL_USERS.length)];
  document.getElementById('uid-input').value = uid;
  getRecommendations();
}

async function getRecommendations() {
  const uid   = document.getElementById('uid-input').value.trim();
  const model = document.getElementById('rec-model').value;
  if (!uid) return;
  const list = document.getElementById('rec-list');
  list.innerHTML = loading('Fetching recommendations…');
  document.getElementById('history-panel').style.display = 'none';

  const resp = await fetch(`/recommend?user_id=${encodeURIComponent(uid)}&model=${model}`);
  const data = await resp.json();
  if (data.error) { list.innerHTML = `<p style="color:red">${data.error}</p>`; return; }

  list.innerHTML = data.recommendations.map((r, i) => {
    const reviewsHtml = r.reviews && r.reviews.length
      ? r.reviews.map(rv => `<p>${rv.substring(0,280)}${rv.length>280?'…':''}</p>`).join('')
      : '<p style="color:#a0aec0">No reviews available.</p>';
    const rid = `rv-${i}`;
    return `
    <div class="rec-item">
      <div class="rec-header">
        <div class="rank-badge ${i<3?'top3':''}">${i+1}</div>
        <div class="rec-body">
          <div class="rec-title">${r.title || r.item_id}</div>
          <div class="rec-meta">
            <code>${r.item_id}</code>
            ${r.in_test ? '<span class="tag">in test set</span>' : ''}
            &nbsp;·&nbsp; ${fmtNum(r.review_count)} reviews
          </div>
          <span class="reviews-toggle" onclick="toggleReviews('${rid}')">▶ Sample reviews</span>
          <div id="${rid}" class="reviews-block">${reviewsHtml}</div>
        </div>
        <div class="rec-right">
          <div class="stars">${starsHTML(r.predicted_rating)}</div>
          <div class="score-val">${r.predicted_rating.toFixed(2)} / 5</div>
          <div style="font-size:.72rem;color:#a0aec0">MF predicted rating</div>
        </div>
      </div>
    </div>`;
  }).join('');

  fetchUserHistory(uid);
}

function toggleReviews(id) {
  const el = document.getElementById(id);
  const tog = el.previousElementSibling;
  if (el.style.display === 'block') {
    el.style.display = 'none'; tog.textContent = '▶ Sample reviews';
  } else {
    el.style.display = 'block'; tog.textContent = '▼ Sample reviews';
  }
}

async function fetchUserHistory(uid) {
  const resp = await fetch(`/user_history?user_id=${encodeURIComponent(uid)}`);
  const data = await resp.json();
  if (!data.history || !data.history.length) return;
  document.getElementById('history-body').innerHTML = data.history.map((h, i) => `
    <tr>
      <td>${i+1}</td>
      <td><code>${h.item_id}</code></td>
      <td>${h.title}</td>
      <td><span class="stars">${starsHTML(h.rating)}</span> ${h.rating.toFixed(1)}</td>
      <td>${fmtNum(h.review_count)}</td>
    </tr>`).join('');
  document.getElementById('history-panel').style.display = 'block';
}

async function predictRating() {
  const uid   = document.getElementById('pred-user').value.trim();
  const iid   = document.getElementById('pred-item').value.trim();
  const model = document.getElementById('pred-model').value;
  const box   = document.getElementById('pred-result');
  if (!uid || !iid) {
    box.style.display='block'; box.textContent='Enter both User ID and Item ID.'; return;
  }
  const resp = await fetch(`/predict?user_id=${encodeURIComponent(uid)}&item_id=${encodeURIComponent(iid)}&model=${model}`);
  const data = await resp.json();
  box.style.display = 'block';
  if (data.error) { box.textContent = data.error; return; }
  box.innerHTML = `Predicted rating for <strong>${uid}</strong> on <strong>${iid}</strong>: `
    + `<strong>${data.predicted_rating.toFixed(2)} / 5.0</strong>`
    + ` <span class="model-badge">${data.model}</span>`;
}

// ── Tab 2: Sampled Evaluation ─────────────────────────────────────────────────
async function runSampledEval() {
  const box = document.getElementById('sampled-result');
  box.innerHTML = loading('Running sampled evaluation (50 users × 3 models — BPR, SLIM, TextHybrid)…');
  const resp = await fetch('/sampled_eval');
  const data = await resp.json();
  if (data.error) { box.innerHTML = `<p style="color:red">${data.error}</p>`; return; }

  const rows = data.results.map(r => `
    <tr>
      <td>${r.model}</td>
      <td class="num">${r.precision}</td>
      <td class="num">${r.recall}</td>
      <td class="num">${r.f_measure}</td>
      <td class="num">${r.ndcg}</td>
      <td class="num">${r.n_users}</td>
    </tr>`).join('');
  box.innerHTML = `
    <table class="data-table">
      <thead><tr><th>Model</th><th class="num">P@10</th><th class="num">R@10</th>
             <th class="num">F-measure</th><th class="num">NDCG@10</th><th class="num">Users</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p style="font-size:.78rem;color:#a0aec0;margin-top:10px">
      Protocol: 99 random negatives + ground-truth positives per user, fixed seed=42.
    </p>`;
}

// ── Tab 3: Fairness ───────────────────────────────────────────────────────────
async function runFairness() {
  const box = document.getElementById('fairness-result');
  box.innerHTML = loading('Computing fairness metrics (up to 300 stratified users)…');
  const resp = await fetch('/fairness');
  const data = await resp.json();
  if (data.error) { box.innerHTML = `<p style="color:red">${data.error}</p>`; return; }

  const biasColor = Math.abs(data.bias_pct) > 50 ? 'warn' : 'good';
  const covColor  = data.coverage_pct > 5 ? 'good' : 'warn';

  // Tier bars
  const maxTier = Math.max(...data.tiers.map(t => t.pct));
  const tierBars = data.tiers.map(t => {
    const c = t.pct > 35 ? 'warn' : t.pct < 15 ? 'good' : 'neutral';
    return `
    <div class="bar-wrap">
      <div class="bar-label"><span>${t.label}</span><span>${t.pct.toFixed(1)}%</span></div>
      <div class="bar-track">
        <div class="bar-fill ${c}" style="width:${Math.min(100, t.pct / maxTier * 100)}%"></div>
        <div class="ref-line" style="left:25%" title="Uniform 25%"></div>
      </div>
    </div>`;
  }).join('');

  // Group fairness (ranking quality parity)
  const maxNdcg = Math.max(...data.groups.map(g => g.ndcg));
  const groupRows = data.groups.map(g => {
    const barW = maxNdcg > 0 ? Math.round(g.ndcg / maxNdcg * 120) : 0;
    return `<tr>
      <td>${g.group}</td>
      <td>${g.n_users}</td>
      <td class="num">${g.precision.toFixed(4)}</td>
      <td class="num">${g.recall.toFixed(4)}</td>
      <td>
        <span class="group-bar" style="width:${barW}px"></span>
        ${g.ndcg.toFixed(4)}
      </td>
    </tr>`;
  }).join('');

  box.innerHTML = `
    <div class="metrics-grid" style="grid-template-columns:repeat(4,1fr)">
      <div class="metric-card">
        <div class="lbl">Avg Popularity — Catalogue</div>
        <div class="val">${data.avg_pop_all.toFixed(1)}</div>
        <div class="sub">reviews per item</div>
      </div>
      <div class="metric-card">
        <div class="lbl">Avg Popularity — Recommended</div>
        <div class="val">${data.avg_pop_rec.toFixed(1)}</div>
        <div class="sub">${data.avg_pop_rec > data.avg_pop_all ? '▲' : '▼'} vs catalogue</div>
      </div>
      <div class="metric-card">
        <div class="lbl">Popularity Bias</div>
        <div class="val">${data.bias_pct > 0 ? '+' : ''}${data.bias_pct.toFixed(1)}%</div>
        <div class="sub">0% = perfectly unbiased</div>
      </div>
      <div class="metric-card">
        <div class="lbl">Catalogue Coverage</div>
        <div class="val">${data.coverage_pct.toFixed(1)}%</div>
        <div class="sub">${fmtNum(data.n_unique_rec)} of ${fmtNum(data.n_total)} items</div>
      </div>
    </div>

    <h3>Recommendation Distribution by Popularity Tier</h3>
    <p style="font-size:.8rem;color:#718096;margin-bottom:12px">
      Dashed line = 25% uniform baseline. Bars far above it indicate over-recommendation
      of popular items; bars far below indicate neglect of long-tail items.
    </p>
    ${tierBars}

    <h3 style="margin-top:20px">Recommendation Quality by User Activity Group</h3>
    <p style="font-size:.8rem;color:#718096;margin-bottom:12px">
      Gap is measured on NDCG@10 across groups.
    </p>
    <table class="data-table">
      <thead><tr><th>Group</th><th>Users</th><th class="num">P@10</th><th class="num">R@10</th><th>NDCG@10</th></tr></thead>
      <tbody>${groupRows}</tbody>
    </table>
    <p style="font-size:.8rem;margin-top:10px;color:${data.gap > 0.02 ? '#c05621' : '#276749'}">
      ${data.gap > 0.02
        ? 'NDCG@10 gap of ' + data.gap.toFixed(4) + ' indicates activity-group disparity.'
        : 'NDCG@10 gap of ' + data.gap.toFixed(4) + ' is relatively small.'}
    </p>`;
}

// ── Tab 4: Explainability ─────────────────────────────────────────────────────
async function loadUnseenItems() {
  const uid = document.getElementById('exp-user').value;
  const resp = await fetch(`/unseen_items?user_id=${encodeURIComponent(uid)}&n=100`);
  const data = await resp.json();
  const sel = document.getElementById('exp-item');
  sel.innerHTML = data.items.map(it =>
    `<option value="${it.item_id}">${it.item_id}${it.title ? ' — ' + it.title.substring(0,40) : ''}</option>`
  ).join('');
}

async function runExplain() {
  const uid = document.getElementById('exp-user').value;
  const iid = document.getElementById('exp-item').value;
  const box = document.getElementById('explain-result');
  box.innerHTML = loading('Computing scores and ranks across all models…');
  const resp = await fetch(`/explain?user_id=${encodeURIComponent(uid)}&item_id=${encodeURIComponent(iid)}`);
  const data = await resp.json();
  if (data.error) { box.innerHTML = `<p style="color:red">${data.error}</p>`; return; }

  const avgRating = ((data.mf_rating + data.svdpp_rating) / 2).toFixed(2);
  const consensusColor = avgRating >= 4 ? '#276749' : avgRating >= 3 ? '#c07820' : '#9b2c2c';
  const CUTOFF = 10;

  // Rating cards
  const ratingCards = [
    ['MF',    data.mf_rating,    'Primary (best MAE)'],
    ['SVD++', data.svdpp_rating, '4★/5★ implicit feedback'],
  ].map(([name, val, sub]) => {
    const isHigh = val >= 4.0;
    return `<div class="explain-card ${isHigh ? 'highlight' : ''}">
      <div class="model-name">${name}</div>
      <div class="big-val">${val.toFixed(2)}</div>
      <div class="stars">${starsHTML(val)}</div>
      <div class="sub-val">${sub}</div>
    </div>`;
  }).join('');

  // Rank cards
  const rankCards = [
    ['BPR',  data.bpr_rank,  'Baseline'],
    ['SLIM', data.slim_rank, 'Best Task-B'],
  ].map(([name, rank, sub]) => {
    const inTop = typeof rank === 'number' && rank <= CUTOFF;
    return `<div class="explain-card ${inTop ? 'highlight' : ''}">
      <div class="model-name">${name}</div>
      <div class="big-val">${typeof rank === 'number' ? '#' + rank : rank}</div>
      <div class="sub-val">${sub} &nbsp; ${inTop ? '✅ Top-10' : '❌ Outside top 10'}</div>
    </div>`;
  }).join('');

  // Counterfactual table
  const cfRows = [
    ['BPR',  data.bpr_rank],
    ['SLIM', data.slim_rank],
  ].map(([name, rank]) => {
    const rec = typeof rank === 'number' && rank <= CUTOFF ? '✅ Yes' : '❌ No';
    return `<tr><td>${name}</td>
      <td class="num">${typeof rank === 'number' ? '#' + rank : rank}</td>
      <td>${rec}</td></tr>`;
  }).join('');

  const anyRec = [data.bpr_rank, data.slim_rank]
    .some(r => typeof r === 'number' && r <= CUTOFF);

  // ── Score decomposition (borrowed from friend's pipeline approach) ────────
  const THRESHOLD = 4.0;
  const alphaB    = data.alpha_cf.toFixed(2);
  const betaB     = (1 - data.alpha_cf).toFixed(2);

  const cfColor  = data.cf_raw      >= THRESHOLD ? '#276749' : '#c07820';
  const cntColor = data.content_score >= THRESHOLD ? '#276749' : '#c07820';
  const blColor  = data.blended     >= THRESHOLD ? '#276749' : '#c07820';

  const decompSection = `
    <h3>Score Decomposition &nbsp;
      <small style="font-weight:400;color:#718096">MF (CF) + TF-IDF Content → dynamic blend</small>
    </h3>
    <p style="font-size:.82rem;color:#718096;margin-bottom:14px">
      α=${alphaB} × CF + β=${betaB} × Content &nbsp;—&nbsp;
      based on <strong>${data.n_history}</strong> training interactions
      <span style="color:#a0aec0">(α: 0.30 cold-start → 0.70 power user)</span>
    </p>
    <div class="explain-grid">
      <div class="explain-card">
        <div class="model-name">① MF (CF) Score</div>
        <div class="big-val" style="color:${cfColor}">${data.cf_raw.toFixed(3)}</div>
        <div class="stars">${starsHTML(data.cf_raw)}</div>
        <div class="sub-val">global_mean + user_bias + item_bias + P·Q</div>
      </div>
      <div class="explain-card">
        <div class="model-name">② TF-IDF Content Score</div>
        <div class="big-val" style="color:${cntColor}">${data.content_score.toFixed(3)}</div>
        <div class="stars">${starsHTML(data.content_score)}</div>
        <div class="sub-val">cosine_sim = ${data.content_sim.toFixed(4)}</div>
      </div>
      <div class="explain-card ${data.blended >= THRESHOLD ? 'highlight' : ''}">
        <div class="model-name">③ Blended Score</div>
        <div class="big-val" style="color:${blColor}">${data.blended.toFixed(3)}</div>
        <div class="stars">${starsHTML(data.blended)}</div>
        <div class="sub-val">${alphaB}×CF + ${betaB}×Content</div>
      </div>
    </div>

    <h3>Per-Component Counterfactual
      <small style="font-weight:400;color:#718096">
        — if this component were boosted to 4.0, holding all else fixed
      </small>
    </h3>
    <p style="font-size:.82rem;color:#718096;margin-bottom:10px">
      Inspired by Zhang et al. (2021) <em>Counterfactual Explainable Recommendation</em>.
      For components <em>below</em> 4.0: would boosting them to the threshold tip this item into the top-10?
      Components already ≥ 4.0 are not a limiting factor.
    </p>
    <table class="data-table" style="max-width:580px">
      <thead>
        <tr>
          <th>Component</th>
          <th class="num">Current</th>
          <th class="num">Hypothetical (= 4.0)</th>
          <th class="num">Resulting Blend</th>
          <th>Would Recommend?</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>MF (CF) Score</td>
          <td class="num">${data.cf_raw.toFixed(3)}</td>
          ${data.cf_raw >= THRESHOLD
            ? `<td class="num" colspan="2" style="color:#276749">Already ≥ 4.0 — not a limiting factor</td>
               <td>✅ Not limiting</td>`
            : `<td class="num">4.000</td>
               <td class="num">${data.cf_if_4.toFixed(3)}</td>
               <td>${data.cf_if_4 >= THRESHOLD ? '✅ Yes' : '❌ No'}</td>`}
        </tr>
        <tr>
          <td>TF-IDF Content</td>
          <td class="num">${data.content_score.toFixed(3)}</td>
          ${data.content_score >= THRESHOLD
            ? `<td class="num" colspan="2" style="color:#276749">Already ≥ 4.0 — not a limiting factor</td>
               <td>✅ Not limiting</td>`
            : `<td class="num">4.000</td>
               <td class="num">${data.content_if_4.toFixed(3)}</td>
               <td>${data.content_if_4 >= THRESHOLD ? '✅ Yes' : '❌ No'}</td>`}
        </tr>
      </tbody>
    </table>`;

  box.innerHTML = `
    <div style="background:#f7fafc;border-radius:10px;padding:14px 18px;margin-bottom:16px">
      <strong>${data.title || iid}</strong>
      <span style="color:#a0aec0;font-size:.8rem;margin-left:8px">
        ${fmtNum(data.review_count)} reviews &nbsp;·&nbsp; Model consensus:
        <span style="color:${consensusColor};font-weight:600">${avgRating} / 5</span>
      </span>
    </div>

    <h3>Task A — Predicted Ratings</h3>
    <div class="explain-grid">${ratingCards}</div>

    <h3>Task B — Item Rank in Each Recommender</h3>
    <p style="font-size:.8rem;color:#718096;margin-bottom:10px">
      Out of ${fmtNum(data.n_unseen)} unseen items for this user.
      SLIM and BPR ranks are exact (full matrix).
    </p>
    <div class="explain-grid">${rankCards}</div>

    <h3>Counterfactual — Would any model recommend this item?</h3>
    <table class="data-table" style="max-width:420px">
      <thead><tr><th>Model</th><th class="num">Rank</th><th>In Top-10?</th></tr></thead>
      <tbody>${cfRows}</tbody>
    </table>
    <p style="margin-top:12px;font-size:.88rem;font-weight:600;
              color:${anyRec ? '#276749' : '#9b2c2c'}">
      ${anyRec
        ? '✅ At least one model would recommend this item in its top 10.'
        : '❌ No model would recommend this item in the top 10 for this user.'}
    </p>

    <hr style="margin:24px 0;border:none;border-top:1px solid #e2e8f0">
    ${decompSection}`;
}

// ── Init ──────────────────────────────────────────────────────────────────────
window.onload = function() {
  const uid = ALL_USERS[Math.floor(Math.random() * Math.min(ALL_USERS.length, 100))];
  document.getElementById('uid-input').value = uid;
  getRecommendations();
  loadUnseenItems();
};
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Jinja filters
# ─────────────────────────────────────────────────────────────────────────────
@app.template_filter("r4")
def _r4(v): return f"{float(v):.4f}"

@app.template_filter("r2")
def _r2(v): return f"{float(v):.2f}"

@app.template_filter("format_num")
def _fmt(v): return f"{int(v):,}"


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(
        HTML,
        active_users=ACTIVE_USERS,
        results=RESULTS,
        n_users=len(ALL_USERS),
        n_items=len(ALL_ITEMS),
        n_train=len(TRAIN_DF),
    )


@app.route("/recommend")
def recommend():
    user_id    = request.args.get("user_id", "")
    model_name = request.args.get("model", "slim")

    seen       = TRAIN_SEEN.get(user_id, set())
    test_items = TEST_ITEMS.get(user_id, set())

    if model_name == "bpr" and BPR_MODEL:
        raw = BPR_MODEL.recommend(user_id, seen, n=TOP_K)
    elif model_name == "hybrid" and SLIM_MODEL and TEXT_HYBRID_MODEL:
        cands = SLIM_MODEL.recommend(user_id, seen, n=TOP_K * 3)
        raw = TEXT_HYBRID_MODEL.rerank(user_id, cands, USER_ITEMS_DICT, n=TOP_K)
        if not raw:
            raw = cands[:TOP_K]
    elif model_name == "mf" and MF_MODEL:
        raw = MF_MODEL.recommend(user_id, seen, n=TOP_K)
    else:
        raw = SLIM_MODEL.recommend(user_id, seen, n=TOP_K) if SLIM_MODEL else []
        if not raw and BPR_MODEL:
            raw = BPR_MODEL.recommend(user_id, seen, n=TOP_K)

    # Universal fallback for cold-start/empty-return paths.
    if not raw and BPR_MODEL:
        raw = BPR_MODEL.recommend(user_id, seen, n=TOP_K)

    recs = []
    for item_id, _ in raw:
        title   = str(ITEM_TITLES.get(item_id, item_id))[:80]
        pred_score, _ = _predict_task_a_score(user_id, item_id, model_pref="auto")
        pred = round(pred_score, 4)
        reviews = [str(r)[:300] for r in ITEM_REVIEWS.get(item_id, [])]
        recs.append({
            "item_id":          item_id,
            "title":            title,
            "predicted_rating": pred,
            "in_test":          item_id in test_items,
            "review_count":     int(ITEM_COUNTS.get(item_id, 0)),
            "reviews":          reviews,
        })

    return jsonify({"user_id": user_id, "recommendations": recs, "model": model_name})


@app.route("/user_history")
def user_history():
    user_id = request.args.get("user_id", "")
    rows = (
        TRAIN_DF[TRAIN_DF["user_id"] == user_id][["item_id", "title", "rating"]]
        .dropna(subset=["rating"])
        .sort_values("rating", ascending=False)
        .head(15)
    )
    history = [
        {
            "item_id":      row["item_id"],
            "title":        str(row.get("title", ""))[:70] if row.get("title") else "",
            "rating":       float(row["rating"]),
            "review_count": int(ITEM_COUNTS.get(row["item_id"], 0)),
        }
        for _, row in rows.iterrows()
    ]
    return jsonify({"history": history})


@app.route("/predict")
def predict():
    user_id    = request.args.get("user_id", "")
    item_id    = request.args.get("item_id", "")
    model_name = request.args.get("model", "auto")

    if MF_MODEL is None and SVDPP_MODEL is None:
        return jsonify({"error": "No rating model available"})

    score, label = _predict_task_a_score(user_id, item_id, model_pref=model_name)

    return jsonify({"user_id": user_id, "item_id": item_id,
                    "predicted_rating": round(float(score), 4), "model": label})


@app.route("/unseen_items")
def unseen_items():
    user_id = request.args.get("user_id", "")
    n       = int(request.args.get("n", 100))
    seen    = TRAIN_SEEN.get(user_id, set())
    items   = [
        {"item_id": it, "title": str(ITEM_TITLES.get(it, ""))[:50]}
        for it in sorted(ALL_ITEMS)
        if it not in seen
    ][:n]
    return jsonify({"items": items})


@app.route("/fairness")
def fairness():
    global _fairness_cache
    if _fairness_cache is not None:
        return jsonify(_fairness_cache)

    if SLIM_MODEL is None:
        return jsonify({"error": "SLIM model not available"})

    from utils.evaluate import precision_at_k, recall_at_k, ndcg_at_k

    # Stratified sampling so all activity tiers are represented.
    rng = np.random.default_rng(42)
    low_uids    = [u for u, s in TRAIN_SEEN.items() if len(s) <= 10]
    medium_uids = [u for u, s in TRAIN_SEEN.items() if 10 < len(s) <= 50]
    high_uids   = [u for u, s in TRAIN_SEEN.items() if len(s) > 50]
    def _sample(lst, k):
        k = min(k, len(lst))
        return rng.choice(lst, size=k, replace=False).tolist() if k else []
    sampled_uids = _sample(low_uids, 100) + _sample(medium_uids, 100) + _sample(high_uids, 100)

    all_rec_items = []
    group_metrics = {
      "Low (≤10 ratings)": {"p": [], "r": [], "n": []},
      "Medium (11–50)": {"p": [], "r": [], "n": []},
      "High (>50)": {"p": [], "r": [], "n": []},
    }

    for uid in sampled_uids:
        seen = TRAIN_SEEN.get(uid, set())
        recs = SLIM_MODEL.recommend(uid, seen, n=TOP_K)
        items = [it for it, _ in recs]
        all_rec_items.extend(items)

        relevant = TEST_ITEMS.get(uid, set())
        p = precision_at_k(items, relevant, k=TOP_K)
        r = recall_at_k(items, relevant, k=TOP_K)
        n = ndcg_at_k(items, relevant, k=TOP_K)

        n_hist = len(seen)
        if n_hist <= 10:
          group_metrics["Low (≤10 ratings)"]["p"].append(p)
          group_metrics["Low (≤10 ratings)"]["r"].append(r)
          group_metrics["Low (≤10 ratings)"]["n"].append(n)
        elif n_hist <= 50:
          group_metrics["Medium (11–50)"]["p"].append(p)
          group_metrics["Medium (11–50)"]["r"].append(r)
          group_metrics["Medium (11–50)"]["n"].append(n)
        else:
          group_metrics["High (>50)"]["p"].append(p)
          group_metrics["High (>50)"]["r"].append(r)
          group_metrics["High (>50)"]["n"].append(n)

    pop_all     = np.array(list(ITEM_COUNTS.values()), dtype=float)
    pop_rec     = np.array([ITEM_COUNTS.get(it, 1) for it in all_rec_items], dtype=float)
    avg_pop_all = float(np.mean(pop_all))
    avg_pop_rec = float(np.mean(pop_rec))
    bias_pct    = (avg_pop_rec - avg_pop_all) / avg_pop_all * 100
    unique_rec  = set(all_rec_items)
    coverage    = len(unique_rec) / len(ALL_ITEMS) * 100

    sorted_items = sorted(ALL_ITEMS, key=lambda x: ITEM_COUNTS.get(x, 0))
    n_it = len(sorted_items)
    tier_sets = [
        ("Long-tail (Q1 — least popular)", set(sorted_items[:n_it // 4])),
        ("Lower-mid (Q2)",                 set(sorted_items[n_it // 4: n_it // 2])),
        ("Upper-mid (Q3)",                 set(sorted_items[n_it // 2: 3 * n_it // 4])),
        ("Head (Q4 — most popular)",        set(sorted_items[3 * n_it // 4:])),
    ]
    total_recs = len(all_rec_items)
    tiers = [
        {"label": label,
         "pct": round(sum(1 for it in all_rec_items if it in s) / total_recs * 100, 1)}
        for label, s in tier_sets
    ]

    groups = []
    for g, m in group_metrics.items():
      n_users = len(m["p"])
      groups.append({
        "group": g,
        "n_users": n_users,
        "precision": round(float(np.mean(m["p"])), 4) if n_users else 0.0,
        "recall": round(float(np.mean(m["r"])), 4) if n_users else 0.0,
        "ndcg": round(float(np.mean(m["n"])), 4) if n_users else 0.0,
      })

    ndcgs = [g["ndcg"] for g in groups if g["n_users"] > 0]
    gap = round(max(ndcgs) - min(ndcgs), 4) if len(ndcgs) >= 2 else 0.0

    _fairness_cache = {
        "avg_pop_all":  round(avg_pop_all, 2),
        "avg_pop_rec":  round(avg_pop_rec, 2),
        "bias_pct":     round(bias_pct, 1),
        "coverage_pct": round(coverage, 1),
        "n_unique_rec": len(unique_rec),
        "n_total":      len(ALL_ITEMS),
        "tiers":        tiers,
        "groups":       groups,
        "gap":          gap,
    }
    return jsonify(_fairness_cache)


@app.route("/sampled_eval")
def sampled_eval():
    global _sampled_eval_cache
    if _sampled_eval_cache is not None:
        return jsonify(_sampled_eval_cache)

    from utils.evaluate import evaluate_sampled

    def _hybrid_rec(uid, seen, n):
        cands = SLIM_MODEL.recommend(uid, seen, n=n * 3) if SLIM_MODEL else []
        return TEXT_HYBRID_MODEL.rerank(uid, cands, USER_ITEMS_DICT, n=n) if TEXT_HYBRID_MODEL else cands

    model_map = [
        ("BPR",        BPR_MODEL.recommend if BPR_MODEL else None),
        ("SLIM",       SLIM_MODEL.recommend if SLIM_MODEL else None),
        ("TextHybrid", _hybrid_rec if (SLIM_MODEL and TEXT_HYBRID_MODEL) else None),
    ]
    rows = []
    for name, rec_fn in model_map:
        if rec_fn is None:
            continue
        r = evaluate_sampled(
            rec_fn=rec_fn,
            all_items=ALL_ITEMS,
            train_seen_dict=TRAIN_SEEN,
            test_items_dict=TEST_ITEMS,
            k=TOP_K, n_neg=99, n_users=50,
        )
        rows.append({
            "model":     name,
            "precision": f"{r['Precision@K']:.4f}",
            "recall":    f"{r['Recall@K']:.4f}",
            "f_measure": f"{r['F-measure']:.4f}",
            "ndcg":      f"{r['NDCG@K']:.4f}",
            "n_users":   r["n_users"],
        })

    _sampled_eval_cache = {"results": rows}
    return jsonify(_sampled_eval_cache)


@app.route("/explain")
def explain():
    user_id = request.args.get("user_id", "")
    item_id = request.args.get("item_id", "")

    seen     = TRAIN_SEEN.get(user_id, set())
    n_unseen = len(ALL_ITEMS) - len(seen)

    # ── Task A ratings ────────────────────────────────────────────────────────
    r_mf    = round(float(MF_MODEL.predict_one(user_id, item_id)),    4) if MF_MODEL    else 0.0
    r_svdpp = round(float(SVDPP_MODEL.predict_one(user_id, item_id)), 4) if SVDPP_MODEL else 0.0

    # ── Task B ranks ──────────────────────────────────────────────────────────
    def slim_rank():
        if SLIM_MODEL is None: return None
        u = SLIM_MODEL.user2idx.get(user_id)
        i = SLIM_MODEL.item2idx.get(item_id)
        if u is None or i is None: return None
        if item_id in seen: return "N/A (training item)"
        scores = (SLIM_MODEL.R_train[u, :] @ SLIM_MODEL.W).toarray().ravel()
        target_score = float(scores[i])
        for it in seen:
            j = SLIM_MODEL.item2idx.get(it)
            if j is not None: scores[j] = -np.inf
        return int(np.sum(scores > target_score)) + 1

    def bpr_rank():
        if BPR_MODEL is None: return None
        u = BPR_MODEL.user2idx.get(user_id)
        i = BPR_MODEL.item2idx.get(item_id)
        if u is None or i is None: return None
        if item_id in seen: return "N/A (training item)"
        scores = BPR_MODEL.bi + BPR_MODEL.P[u] @ BPR_MODEL.Q.T
        target_score = float(scores[i])
        for it in seen:
            j = BPR_MODEL.item2idx.get(it)
            if j is not None: scores[j] = -np.inf
        return int(np.sum(scores > target_score)) + 1

    # ── Score decomposition (MF + TF-IDF content blend) ──────────────────────
    u_mf = MF_MODEL.user2idx.get(user_id) if MF_MODEL else None
    i_mf = MF_MODEL.item2idx.get(item_id) if MF_MODEL else None
    if u_mf is not None and i_mf is not None and MF_MODEL:
        cf_raw = float(np.clip(
            MF_MODEL.global_mean + MF_MODEL.bu[u_mf] + MF_MODEL.bi[i_mf]
            + MF_MODEL.P[u_mf] @ MF_MODEL.Q[i_mf],
            1.0, 5.0,
        ))
    else:
        cf_raw = float(MF_MODEL.global_mean) if MF_MODEL else 3.5

    content_sim, content_score = _compute_content_score(user_id, item_id)
    n_history  = len(seen)
    alpha_cf   = _dynamic_alpha(n_history)
    blended    = float(np.clip(alpha_cf * cf_raw + (1 - alpha_cf) * content_score, 1.0, 5.0))
    cf_if_4    = float(np.clip(alpha_cf * 4.0   + (1 - alpha_cf) * content_score, 1.0, 5.0))
    content_if_4 = float(np.clip(alpha_cf * cf_raw + (1 - alpha_cf) * 4.0,        1.0, 5.0))

    return jsonify({
        "user_id":        user_id,
        "item_id":        item_id,
        "title":          str(ITEM_TITLES.get(item_id, item_id))[:80],
        "review_count":   int(ITEM_COUNTS.get(item_id, 0)),
        "n_unseen":       n_unseen,
        # Task A
        "mf_rating":      r_mf,
        "svdpp_rating":   r_svdpp,
        # Task B
        "slim_rank":      slim_rank(),
        "bpr_rank":       bpr_rank(),
        # Score decomposition
        "cf_raw":         round(cf_raw, 4),
        "content_sim":    round(content_sim, 4),
        "content_score":  round(content_score, 4),
        "n_history":      n_history,
        "alpha_cf":       round(alpha_cf, 2),
        "blended":        round(blended, 4),
        "cf_if_4":        round(cf_if_4, 4),
        "content_if_4":   round(content_if_4, 4),
    })


if __name__ == "__main__":
    print("Starting demo at http://127.0.0.1:5001")
    app.run(debug=False, port=5001)
