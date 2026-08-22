#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

data = np.load(ROOT / "data" / "dataset.npz", allow_pickle=True)
X, y, match = data["X"], data["y"].astype(int), data["match"]

matches = sorted(set(match))
rng = np.random.default_rng(42)
rng.shuffle(matches)
split = int(len(matches) * 0.8)
train_m, val_m = set(matches[:split]), set(matches[split:])
tr = np.array([m in train_m for m in match])
va = ~tr

model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric="logloss",
    early_stopping_rounds=40,
)
model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)

p = model.predict_proba(X[va])[:, 1]
print(f"train ticks={tr.sum()} val ticks={va.sum()} ({len(train_m)}/{len(val_m)} matches)")
print(f"auc={roc_auc_score(y[va], p):.4f} logloss={log_loss(y[va], p):.4f} brier={brier_score_loss(y[va], p):.4f}")

by_tick = {}
for m_i, tick, yi, pi in zip(match[va], X[va][:, 0] * 100, y[va], p):
    by_tick.setdefault(int(tick), []).append((yi, pi))
print("\ntick |  n  | acc | avg P(A)")
for tick in sorted(by_tick):
    arr = by_tick[tick]
    ys = np.array([a for a, _ in arr])
    ps = np.array([b for _, b in arr])
    print(f"{tick:4.0f} | {len(arr):3d} | {np.mean((ps > 0.5) == ys):.3f} | {ps.mean():.3f}")

model.save_model(OUT / "eval_model.json")
