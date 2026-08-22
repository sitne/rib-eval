#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TMAX = 44
NPLAYERS = 10


class WinPredictor(nn.Module):
    def __init__(self, f=9, d=128, layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(f, d)
        self.pos = nn.Embedding(TMAX, d)
        self.atk = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=d * 4, dropout=dropout, batch_first=True
        )
        self.spatial = nn.TransformerEncoder(layer, layers)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, 1)

    def forward(self, x, at):
        B, T, P, _ = x.shape
        h = self.proj(x) + self.pos.weight[:T].view(1, T, 1, -1)
        atk_e = self.atk(at)[:, None, None, :].expand(-1, T, 1, -1)
        h = h.view(B * T, P, -1) + atk_e.reshape(B * T, 1, -1)
        h = self.spatial(h).view(B, T, P, -1)
        alive = x[..., 4:5]
        pooled = (h * alive).sum(2) / alive.sum(2).clamp(min=1.0)
        out, _ = self.gru(pooled)
        return self.head(out).squeeze(-1)


def predict(model, x, at, bs=256):
    dev = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xi = torch.as_tensor(x[i : i + bs], dtype=torch.float32, device=dev)
            ai = torch.as_tensor(at[i : i + bs].astype(np.int64), device=dev)
            outs.append(model(xi, ai).cpu().numpy())
    return np.concatenate(outs)


def run(tag, map_filter, data, epochs=20, batch=128, lr=1e-3, seed=42):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, mask, y, at, maps, matches = data
    sel = np.ones(len(y), dtype=bool) if map_filter is None else (maps == map_filter)

    Xs, Ms, ys, ats, mts = X[sel], mask[sel], y[sel], at[sel], matches[sel]
    uniq = sorted(set(mts))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    split = int(len(uniq) * 0.8)
    tr_m, va_m = set(uniq[:split]), set(uniq[split:])
    tr = np.array([m in tr_m for m in mts])
    va = ~tr

    xt = torch.tensor(Xs[tr], device=dev)
    mt = torch.tensor(Ms[tr], device=dev)
    yt = torch.tensor(ys[tr], dtype=torch.float32, device=dev)
    att = torch.tensor(ats[tr].astype(np.int64), device=dev)
    xv = torch.tensor(Xs[va], device=dev)
    mv = torch.tensor(Ms[va], device=dev)
    yv = ys[va]
    avv = torch.tensor(ats[va].astype(np.int64), device=dev)

    model = WinPredictor().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(reduction="none")

    def evaluate():
        return predict(model, Xs[va], ats[va])

    best_auc, best_state, patience = 0.0, None, 0
    n = len(xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            logits = model(xt[idx], att[idx])
            l = lossf(logits, yt[idx][:, None].expand_as(logits))
            l = (l * mt[idx]).sum() / mt[idx].sum()
            opt.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(l) * len(idx)
        vl = evaluate()
        vmask = mv.cpu().numpy().astype(bool)
        auc = roc_auc_score(np.repeat(yv[:, None], TMAX, 1)[vmask], vl[vmask])
        print(f"[{tag}] epoch {ep}: loss={tot / n:.4f} val_auc={auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, best_state, patience = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    model.eval()
    vl = evaluate()
    vmask = mv.cpu().numpy().astype(bool)
    yrep = np.repeat(yv[:, None], TMAX, 1)
    all_acc = ((vl > 0) == yrep)[vmask].mean()
    last_tick_counts = mv.cpu().numpy().sum(1).astype(int) - 1
    last_pred = vl[np.arange(len(vl)), last_tick_counts]
    last_acc = ((last_pred > 0) == yv).mean()
    print(f"[{tag}] BEST val_auc={best_auc:.4f} rounds={len(yv)} tick_acc={all_acc:.4f} last_tick_acc={last_acc:.4f}", flush=True)
    torch.save(model.state_dict(), OUT / f"transformer_{tag}.pt")
    return tag, best_auc, last_acc, int(va.sum()), len(yv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="ALL", help="map name, POOLED, or ALL")
    args = ap.parse_args()

    d = np.load(ROOT / "data" / "sequences.npz", allow_pickle=True)
    data = (d["X"], d["mask"], d["y"], d["attacker"], d["map"], d["match"])

    results = []
    if args.map == "POOLED":
        results.append(run("pooled", None, data))
    else:
        tags = sorted(set(d["map"])) if args.map == "ALL" else [args.map]
        for t in tags:
            results.append(run(f"map_{t}", t, data))
        if args.map == "ALL":
            results.append(run("pooled", None, data))
    print("\n=== summary ===")
    print(f"{'tag':16s} {'val_auc':>8s} {'last_acc':>9s} {'val_ticks':>10s} {'val_rounds':>11s}")
    for tag, auc, acc, nt, nr in results:
        print(f"{tag:16s} {auc:8.4f} {acc:9.4f} {nt:10d} {nr:11d}")


if __name__ == "__main__":
    main()
