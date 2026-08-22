#!/usr/bin/env python3
import argparse
import html
import json
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

import numpy as np
import torch

from train_transformer import WinPredictor, predict

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DATA = ROOT / "data"
TICK_MS = 5000.0


def load_model():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = WinPredictor().to(dev)
    m.load_state_dict(torch.load(OUT / "transformer_pooled.pt", map_location=dev, weights_only=True))
    m.eval()
    return m


def predict_all(model, X, at):
    wps = []
    for i in range(0, len(X), 512):
        logits = predict(model, X[i : i + 512], at[i : i + 512])
        wps.append(1.0 / (1.0 + np.exp(-logits)))
    return np.concatenate(wps)


def player_name(roster, actor_id):
    return roster.get(actor_id, {}).get("name", str(actor_id))


def clock(sec):
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def svg_curve(wp, kill_times=None, spike_times=None, defuse_times=None, swing_idx=None, width=760, height=150):
    pad = 10
    n = len(wp)
    xs = [pad + i * (width - 2 * pad) / (n - 1) for i in range(n)]
    ys = [height - pad - v * (height - 2 * pad) for v in wp]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    mid = height - pad - 0.5 * (height - 2 * pad)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="curve" preserveAspectRatio="none">']
    parts.append(
        f'<line x1="{pad}" y1="{mid:.1f}" x2="{width - pad}" y2="{mid:.1f}" stroke="#555" stroke-dasharray="4 4"/>'
    )
    for lst, color, op in ((spike_times, "#22c55e", "0.30"), (defuse_times, "#60a5fa", "0.30"), (kill_times or [], "#ff4655", "0.25")):
        if not lst:
            continue
        t0, step = lst
        for kt in t0:
            x = pad + (kt / step) * (width - 2 * pad)
            x = min(max(x, pad), width - pad)
            parts.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" y2="{height - 4}" stroke="{color}" stroke-opacity="{op}" stroke-width="7"/>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#38bdf8" stroke-width="2.5"/>')
    for si in swing_idx or []:
        parts.append(
            f'<circle cx="{xs[si]:.1f}" cy="{ys[si]:.1f}" r="5" fill="#fbbf24" stroke="#111" stroke-width="1.5"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { background:#0f1923; color:#ece8e1; font-family:'Segoe UI',system-ui,sans-serif; margin:0; padding:24px; }
h1 { font-size:26px; letter-spacing:.06em; margin:0 0 4px; }
h1 span { color:#ff4655; }
h2 { font-size:17px; letter-spacing:.08em; text-transform:uppercase; color:#8fa3ad; border-bottom:1px solid #233; padding-bottom:6px; margin-top:34px;}
.chips { display:flex; gap:12px; flex-wrap:wrap; margin:14px 0 6px; }
.chip { background:#1b2733; border:1px solid #2a3a48; border-radius:8px; padding:8px 14px; font-size:13px; }
.chip b { color:#38bdf8; font-size:16px; }
.grid2 { display:flex; gap:22px; flex-wrap:wrap; }
.card { background:#141f29; border:1px solid #233240; border-radius:12px; padding:16px 18px; flex:1 1 420px; min-width:380px;}
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th { text-align:left; color:#8fa3ad; font-weight:600; padding:6px 8px; border-bottom:1px solid #2a3a48; font-size:11.5px; letter-spacing:.05em; text-transform:uppercase;}
td { padding:6px 8px; border-bottom:1px solid #1c2833; }
tr:hover td { background:#182430; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.bar-wrap { position:relative; height:18px; min-width:120px; }
.bar { height:100%; border-radius:4px; opacity:.85; }
.pos .bar { background:#22c55e; } .neg .bar { background:#ef4444; }
.wpa-pos { color:#22c55e; font-weight:700;} .wpa-neg { color:#ef4444; font-weight:700;}
.featured { display:grid; grid-template-columns:repeat(auto-fit,minmax(560px,1fr)); gap:18px; margin-top:14px;}
.fcard { background:#141f29; border:1px solid #233240; border-radius:12px; padding:14px 16px; }
.fhead { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px; }
.fhead .t { font-weight:700; letter-spacing:.04em; color:#ece8e1; text-decoration:none; }
.fhead .t:hover { color:#38bdf8; }
.fhead .sub { color:#8fa3ad; font-size:12px; font-weight:400; }
td .sub, a.sub { color:#8fa3ad; font-size:11px; }
a.sub:hover { color:#38bdf8; }
.delta-chip { font-weight:800; font-size:15px; padding:2px 10px; border-radius:6px; }
.d-up { background:#052e16; color:#4ade80; border:1px solid #14532d;}
.d-down { background:#450a0a; color:#f87171; border:1px solid #7f1d1d;}
.curve { width:100%; height:auto; background:#0b1319; border-radius:8px; }
.fmeta { color:#8fa3ad; font-size:12px; margin-top:6px;}
.note { color:#8fa3ad; font-size:12.5px; line-height:1.6; max-width:900px; }
.legend { display:flex; gap:18px; font-size:12px; color:#8fa3ad; margin:8px 0;}
.swatch { display:inline-block; width:22px; height:10px; border-radius:3px; margin-right:6px; vertical-align:middle;}
"""


def fmt_date(iso):
    if not iso:
        return None
    try:
        return _date.fromisoformat(iso).strftime("%b %d, %Y")
    except ValueError:
        return iso


def match_label(match_id, match_info):
    info = (match_info or {}).get(match_id) or {}
    title = info.get("title")
    if not title:
        return f"Match {match_id}", ""
    parts = [title]
    d = fmt_date(info.get("date"))
    if d:
        parts.append(d)
    return " — ".join([parts[0], *parts[1:]]), f"rib #{match_id}"


def build_html(ranked, swings, metas, W, d, args, n_rounds, match_info=None, timing_rows=None, sig=None):
    max_abs = max(abs(v["wpa"]) for _, v in ranked) or 1.0

    def sig_html(name):
        s = (sig or {}).get(name)
        if not s:
            return ""
        cls = "wpa-pos" if s["d"] >= 0 else "wpa-neg"
        return (
            f"<br><span class='sub'><a class='sub' href='https://rib.gg/matches/{s['match']}' target='_blank'>"
            f"R{s['round']} @ {clock(s['t_ms'] / 1000)} · <span class='{cls}'>{s['d']:+.2f}</span> ({s['type']})</a></span>"
        )

    def player_rows(entries):
        rows = []
        for name, s in entries:
            avg = s["wpa"] / s["kills"] if s["kills"] else 0.0
            cls = "pos" if s["wpa"] >= 0 else "neg"
            bw = abs(s["wpa"]) / max_abs * 100
            rows.append(
                f"<tr class='{cls}'><td>{html.escape(name)}{sig_html(name)}</td>"
                f"<td class='num'><span class='wpa-{cls}'>{s['wpa']:+.2f}</span></td>"
                f"<td><div class='bar-wrap'><div class='bar' style='width:{bw:.1f}%'></div></div></td>"
                f"<td class='num'>{s['kills']}</td><td class='num'>{s['deaths']}</td>"
                f"<td class='num'>{avg:+.3f}</td></tr>"
            )
        return "\n".join(rows)

    n = args.top_players
    top_html = player_rows(ranked[:n])
    bot_html = player_rows(list(reversed(ranked[-n:])))

    seen, featured = set(), []
    for s in swings:
        key = (s["match"], s["round"])
        if key in seen:
            continue
        seen.add(key)
        if len(featured) >= args.top_swing_cards:
            break
        idx = next(
            i for i, r in enumerate(metas) if r["match"] == s["match"] and r["roundNum"] == s["round"]
        )
        valid = d["mask"][idx].astype(bool)
        wp = W[idx][valid]
        diffs = np.abs(np.diff(wp))
        si = list(np.where(diffs >= args.swing_threshold)[0] + 1)
        kill_ts = [
            (k["t"] - metas[idx]["startMs"]) / TICK_MS
            for k in metas[idx]["kills"]
            if k["t"] > metas[idx]["startMs"]
        ]
        spike_label = ""
        spike_ts, defuse_ts = [], []
        for pf in DATA.glob(f"replays/{s['match']}-m*.json"):
            try:
                prd = json.loads(pf.read_text())["replayData"]
            except (json.JSONDecodeError, KeyError):
                continue
            if prd.get("map") != s["map"]:
                continue
            rnd = next((rr for rr in prd["rounds"] if rr["roundNum"] == s["round"]), None)
            if rnd:
                start_ms = metas[idx]["startMs"]
                for e in rnd["events"]:
                    if e["t"] <= start_ms:
                        continue
                    tu = (e["t"] - start_ms) / TICK_MS
                    if e["type"] == "plant":
                        spike_ts.append(tu)
                        spike_label = f" · spike @ {clock((e['t'] - start_ms) / 1000)}"
                    elif e["type"] == "defuse":
                        defuse_ts.append(tu)
            break
        curve = svg_curve(
            wp,
            kill_times=(kill_ts, len(wp)),
            spike_times=(spike_ts, len(wp)) if spike_ts else None,
            defuse_times=(defuse_ts, len(wp)) if defuse_ts else None,
            swing_idx=si,
        )
        winner = "A" if d["y"][idx] == 1 else "B"
        label, sub_id = match_label(s["match"], match_info)
        cls = "d-up" if s["delta"] >= 0 else "d-down"
        featured.append(
            f"<div class='fcard'><div class='fhead'>"
            f"<span><a class='t' href='https://rib.gg/matches/{s['match']}' target='_blank'>{html.escape(label)}</a> "
            f"<span class='sub'>· {html.escape(s['map'])} · R{s['round']} · winner={winner} · {sub_id}</span></span>"
            f"<span class='delta-chip {cls}'>{s['delta']:+.2f}</span></div>"
            f"{curve}"
            f"<div class='fmeta'>P(A) {s['before']:.2f} → {s['after']:.2f} @ {clock(s['t_sec'])}{spike_label} · "
            f"red bands = kills · green = plant · blue = defuse · yellow dots = swings</div></div>"
        )

    total_kills = sum(s["kills"] for _, s in ranked)
    chips = (
        f"<div class='chip'><b>{n_rounds:,}</b> rounds</div>"
        f"<div class='chip'><b>{total_kills:,}</b> kills attributed</div>"
        f"<div class='chip'><b>{len(ranked):,}</b> players</div>"
        f"<div class='chip'><b>{len(swings):,}</b> swings ≥ {args.swing_threshold:.2f}</div>"
    )

    timing_html = ""
    if timing_rows:
        league_ale = sum(r["ale"] * r["kills"] for r in timing_rows) / sum(
            r["kills"] for r in timing_rows
        )
        rows = []

        def ph(v):
            return "—" if v is None else f"{v:.3f}"

        for t in timing_rows[: args.top_players]:
            exec_cls = "wpa-pos" if t["exec"] >= 1.0 else "wpa-neg"
            rows.append(
                f"<tr><td>{html.escape(t['name'])}</td>"
                f"<td class='num'>{t['kills']}</td>"
                f"<td class='num'><b>{t['ale']:.3f}</b></td>"
                f"<td class='num'>{ph(t['ale_e'])}</td>"
                f"<td class='num'>{ph(t['ale_m'])}</td>"
                f"<td class='num'>{ph(t['ale_l'])}</td>"
                f"<td class='num'><span class='{exec_cls}'>×{t['exec']:.2f}</span></td></tr>"
            )
        timing_html = f"""
<h2>Engagement timing — who picks high-stakes fights?</h2>
<div class="grid2"><div class="card">
<table><tr><th>player (≥50 K)</th><th class="num">K</th>
<th class="num">ALE all*</th><th class="num">E (&lt;25s)</th><th class="num">M (25–75s)</th><th class="num">L (75s+)</th>
<th class="num">execution</th></tr>
{''.join(rows)}
</table></div></div>
<p class="note">*<b>ALE</b> (average leverage of engagements): the median win-probability swing a typical kill
produces in the situations this player chose to fight in (alive-difference × round-time buckets), split by
phase — E/M/L columns show whether they hunt stakes early or late. It is independent of whether the player
won those duels: it measures <i>when</i> they engage, not the outcome. <b>Execution</b> = actual |ΔP| ÷ ALE —
phase-neutral by construction; converting the stakes they picked into real swings.
League average ALE ≈ {league_ale:.3f}. Role caveat: entry duelists naturally take earlier,
lower-leverage fights than lurkers — compare within roles.</p>"""

    doc = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<title>WPA Report — rib-eval</title><style>{CSS}</style></head><body>
<h1>rib-eval <span>//</span> Win Probability Added Report</h1>
<div class="chips">{chips}</div>

<h2>Top players by WPA</h2>
<div class="grid2">
<div class="card"><table><tr><th>player</th><th class="num">WPA</th><th></th><th class="num">K</th><th class="num">D</th><th class="num">avg/K</th></tr>
{top_html}</table></div>
<div class="card"><table><tr><th>player</th><th class="num">WPA</th><th></th><th class="num">K</th><th class="num">D</th><th class="num">avg/K</th></tr>
{bot_html}</table></div>
</div>

<h2>Biggest WP swings — auto-highlight candidates</h2>
<div class="legend"><span><span class="swatch" style="background:#38bdf8"></span>P(A wins) curve</span>
<span><span class="swatch" style="background:#ff4655;opacity:.35"></span>kills</span>
<span><span class="swatch" style="background:#22c55e;opacity:.5"></span>spike plant</span>
<span><span class="swatch" style="background:#60a5fa;opacity:.5"></span>defuse</span>
<span><span class="swatch" style="background:#fbbf24"></span>swing tick</span></div>
{timing_html}
<div class="featured">{''.join(featured)}</div>

<h2>Notes</h2>
<p class="note">WPA attributes each kill's surrounding Δ win-probability (+/-5s window) to the killer and victim;
plants, defuses and utility value are not yet credited. Probabilities are uncalibrated — treat absolute
percentages as directional. Model: pooled Transformer ({len(metas):,} rounds, holdout AUC ≈ 0.87).
Data: rib.gg 2D replay exports.</p>
</body></html>"""
    path = OUT / "wpa_report.html"
    path.write_text(doc, encoding="utf-8")
    print(f"report -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swing-threshold", type=float, default=0.12)
    ap.add_argument("--top-players", type=int, default=15)
    ap.add_argument("--top-swing-cards", type=int, default=8)
    args = ap.parse_args()

    d = np.load(DATA / "sequences.npz", allow_pickle=True)
    metas = [json.loads(l) for l in open(DATA / "rounds_meta.jsonl")]
    assert len(metas) == len(d["y"])

    model = load_model()
    W = predict_all(model, d["X"], d["attacker"])
    print(f"predicted {W.shape} win-prob curves")

    swings = []
    wpa = defaultdict(lambda: {"wpa": 0.0, "kills": 0, "deaths": 0})
    sig = {}
    kill_records = []
    X_all, mask_all = d["X"], d["mask"]
    for i, r in enumerate(metas):
        if i % 2000 == 0:
            print(f"... round {i}/{len(metas)}", flush=True)
        valid = mask_all[i].astype(bool)
        wp = W[i][valid]
        times = r["startMs"] + np.arange(len(wp)) * TICK_MS
        Xi = X_all[i]

        for t in np.where(np.abs(np.diff(wp)) >= args.swing_threshold)[0]:
            swings.append(
                {
                    "match": r["match"],
                    "map": r["map"],
                    "round": r["roundNum"],
                    "t_sec": float(times[t + 1] / 1000),
                    "before": float(wp[t]),
                    "after": float(wp[t + 1]),
                    "delta": float(wp[t + 1] - wp[t]),
                }
            )

        roster = r["roster"]
        for k in r["kills"]:
            before = (times <= k["t"]) & (times > k["t"] - TICK_MS)
            after = (times >= k["t"]) & (times < k["t"] + TICK_MS)
            if not before.any() or not after.any():
                continue
            delta = float(wp[after].mean() - wp[before].mean())
            sign = 1.0 if k["killer_team"] == "A" else -1.0
            kn, vn = player_name(roster, k["killer"]), player_name(roster, k["victim"])
            wpa[kn]["wpa"] += delta * sign
            wpa[kn]["kills"] += 1
            wpa[vn]["wpa"] -= delta * sign
            wpa[vn]["deaths"] += 1

            imp = delta * sign
            for nm, own in ((kn, imp), (vn, -imp)):
                kind = "K" if nm == kn else "D"
                if nm not in sig or abs(own) > abs(sig[nm]["d"]):
                    sig[nm] = {
                        "match": r["match"],
                        "round": r["roundNum"],
                        "t_ms": k["t"] - r["startMs"],
                        "d": own,
                        "type": kind,
                    }

            j = int(np.argmin(np.abs(times - k["t"])))
            j = min(j, len(wp) - 1)
            alive_a = float(Xi[j, :, 4][Xi[j, :, -1] > 0.5].sum())
            alive_b = float(Xi[j, :, 4][Xi[j, :, -1] <= 0.5].sum())
            diff = int(round(alive_a - alive_b))
            if k["killer_team"] != "A":
                diff = -diff
            diff = max(-5, min(5, diff))
            t_sec = times[j] / 1000.0
            tb = min(int(t_sec // 25), 3)
            kill_records.append(
                {"name": kn, "abs_delta": abs(delta), "situation": (diff, tb)}
            )

    league_L = {}
    by_sit = defaultdict(list)
    for rec in kill_records:
        by_sit[rec["situation"]].append(rec["abs_delta"])
    global_median = float(np.median([rec["abs_delta"] for rec in kill_records]))
    for sit, vals in by_sit.items():
        league_L[sit] = float(np.median(vals)) if len(vals) >= 30 else global_median

    timing_rows = []
    player_kills = defaultdict(list)
    for rec in kill_records:
        player_kills[rec["name"]].append(rec)
    for name, recs in player_kills.items():
        if len(recs) < 50:
            continue
        ale = sum(league_L[r["situation"]] for r in recs) / len(recs)
        actual = sum(r["abs_delta"] for r in recs) / len(recs)
        ph = {"E": [0.0, 0], "M": [0.0, 0], "L": [0.0, 0]}
        for r in recs:
            p = "E" if r["situation"][1] == 0 else ("L" if r["situation"][1] == 3 else "M")
            ph[p][0] += league_L[r["situation"]]
            ph[p][1] += 1
        timing_rows.append(
            {
                "name": name,
                "kills": len(recs),
                "ale": ale,
                "actual": actual,
                "exec": actual / ale if ale else 0.0,
                "ale_e": ph["E"][0] / ph["E"][1] if ph["E"][1] else None,
                "ale_m": ph["M"][0] / ph["M"][1] if ph["M"][1] else None,
                "ale_l": ph["L"][0] / ph["L"][1] if ph["L"][1] else None,
            }
        )
    timing_rows.sort(key=lambda x: x["ale"], reverse=True)

    ranked = sorted(wpa.items(), key=lambda kv: kv[1]["wpa"], reverse=True)
    swings.sort(key=lambda s: abs(s["delta"]), reverse=True)
    mi_path = DATA / "match_info.json"
    match_info = json.loads(mi_path.read_text()) if mi_path.exists() else {}
    build_html(ranked, swings, metas, W, d, args, len(metas), match_info, timing_rows, sig)


if __name__ == "__main__":
    main()
