#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import round_rows

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "replays"
OUT = ROOT / "outputs"

ap = argparse.ArgumentParser()
ap.add_argument("--map", default="577-m1")
ap.add_argument("--rounds", type=int, nargs="*", help="round numbers (1-based)")
args = ap.parse_args()

model = xgb.XGBClassifier()
model.load_model(OUT / "eval_model.json")
replay = json.loads((CACHE / f"{args.map}.json").read_text())
rows, labels, meta = round_rows(replay, args.map.split("-")[0])
X = np.array(rows, dtype=np.float32)
p = model.predict_proba(X)[:, 1]

round_ids = sorted({int(m.split("|")[1]) for m in meta})
targets = args.rounds or round_ids[:6]
ncol = 3
nrow = (len(targets) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3 * nrow), sharey=True, squeeze=False)
for ax, rid in zip(axes.flat, targets):
    mask = np.array([int(m.split("|")[1]) == rid for m in meta])
    ts = X[mask][:, 0] * 100
    pp = p[mask]
    winner = "A" if labels[np.argmax(mask)] == 1 else "B"
    ax.plot(ts, pp, lw=2)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylim(0, 1)
    ax.invert_xaxis()
    ax.set_title(f"R{rid} winner={winner} (A win prob)")
    ax.grid(alpha=0.3)
for ax in axes.flat[len(targets):]:
    ax.axis("off")
fig.suptitle(f"{args.map}: shogi-style eval graph (time flows right to left)")
fig.tight_layout()
out = OUT / f"eval_graph_{args.map}.png"
fig.savefig(out, dpi=120)
print(f"saved {out}")
