#!/usr/bin/env python3
import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch

from train_transformer import WinPredictor, predict
from wpa_analysis import clock, svg_curve, fmt_date, match_label, TICK_MS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DATA = ROOT / "data"

CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { background:#0f1923; color:#ece8e1; font-family:'Segoe UI',system-ui,sans-serif; margin:0; padding:24px; }
h1 { font-size:24px; letter-spacing:.04em; margin:0 0 6px; }
h1 span { color:#ff4655; }
h2 { font-size:16px; color:#8fa3ad; border-bottom:1px solid #233; padding-bottom:6px; margin-top:28px;}
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(520px,1fr)); gap:16px; margin-top:14px; }
.card { background:#141f29; border:1px solid #233240; border-radius:12px; padding:14px 16px; }
.card-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.card-title { font-weight:700; }
.badge { font-weight:800; font-size:13px; padding:2px 8px; border-radius:6px; }
.badge-A { background:#052e16; color:#4ade80; border:1px solid #14532d; }
.badge-B { background:#450a0a; color:#f87171; border:1px solid #7f1d1d; }
.curve-wrap { position:relative; }
.curve { width:100%; height:auto; background:#0b1319; border-radius:8px; display:block; }
.curve.hoverable { cursor:crosshair; }
.tip { position:absolute; background:#1b2733; border:1px solid #2a3a48; border-radius:8px; padding:6px 8px; font-size:12px; line-height:1.4; pointer-events:none; transform:translate(-50%,-110%); display:none; white-space:nowrap; z-index:5; box-shadow:0 4px 12px rgba(0,0,0,.4); }
.tip.spike { border-color:#f59e0b; background:#1c1910; }
.tip .t-main { font-family:monospace; font-weight:700; font-size:13px; }
.tip .t-sub { color:#8fa3ad; font-size:11px; }
.vline { position:absolute; top:0; bottom:0; width:1px; background:#ece8e1; opacity:.6; pointer-events:none; display:none; }
.meta { color:#8fa3ad; font-size:11px; margin-top:6px; }
.legend { display:flex; gap:16px; font-size:12px; color:#8fa3ad; margin:8px 0; }
.swatch { display:inline-block; width:20px; height:9px; border-radius:3px; margin-right:5px; vertical-align:middle; }
a { color:#38bdf8; text-decoration:none; }
a:hover { text-decoration:underline; }
"""

def load_model():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = WinPredictor().to(dev)
    m.load_state_dict(torch.load(OUT / "transformer_pooled.pt", map_location=dev, weights_only=True))
    m.eval()
    return m

def predict_all(model, X, at):
    wps=[]
    for i in range(0,len(X),512):
        logits = predict(model, X[i:i+512], at[i:i+512])
        wps.append(1.0/(1.0+np.exp(-logits)))
    return np.concatenate(wps)

def build_one(match_id, out_path):
    d=np.load(DATA / "sequences.npz", allow_pickle=True)
    metas=[json.loads(l) for l in open(DATA / "rounds_meta.jsonl")]
    match_info=json.loads((DATA / "match_info.json").read_text()) if (DATA/"match_info.json").exists() else {}
    # filter indices for this match
    idxs=[i for i,r in enumerate(metas) if r["match"]==str(match_id)]
    if not idxs:
        print(f"no rounds for match {match_id}")
        return
    idxs.sort(key=lambda i: metas[i]["roundNum"])
    model=load_model()
    X_sub=d["X"][idxs]
    at_sub=d["attacker"][idxs]
    W=predict_all(model, X_sub, at_sub)
    # also need per-round mask
    masks=d["mask"][idxs]
    y_sub=d["y"][idxs]

    title, sub_id = match_label(str(match_id), match_info)
    # collect cards
    cards=[]
    for j, idx in enumerate(idxs):
        r=metas[idx]
        valid=masks[j].astype(bool)
        wp=W[j][valid]
        n=len(wp)
        if n<2:
            continue
        start_ms=r["startMs"]
        kill_ts=[(k["t"]-start_ms)/TICK_MS for k in r["kills"] if k["t"]>start_ms]
        # spike/defuse
        spike_ts=[]; defuse_ts=[]; spike_label=""
        for pf in DATA.glob(f"replays/{match_id}-m*.json"):
            try:
                prd=json.loads(pf.read_text())["replayData"]
            except: continue
            if prd.get("map")!=r["map"]:
                continue
            rnd=next((rr for rr in prd["rounds"] if rr["roundNum"]==r["roundNum"]),None)
            if rnd:
                for e in rnd["events"]:
                    if e["t"]<=start_ms: continue
                    tu=(e["t"]-start_ms)/TICK_MS
                    if e["type"]=="plant":
                        spike_ts.append(tu); spike_label=f" · spike @ {clock((e['t']-start_ms)/1000)}"
                    elif e["type"]=="defuse":
                        defuse_ts.append(tu)
                break
        winner="A" if y_sub[j]==1 else "B"
        # swing dots
        swing_idx=list(np.where(np.abs(np.diff(wp))>=0.12)[0]+1)
        # hover data: elapsed seconds for each tick, plant elapsed if any
        elapsed = [(i*TICK_MS)/1000 for i in range(n)]
        plant_elapsed = spike_ts[0]*TICK_MS/1000 if spike_ts else None
        # embed wp/times/plant as JSON for JS
        wp_json = json.dumps([round(float(v),4) for v in wp])
        times_json = json.dumps(elapsed)
        plant_json = str(plant_elapsed) if plant_elapsed is not None else "null"
        curve_svg=svg_curve(wp, kill_times=(kill_ts,len(wp)) if kill_ts else None,
                        spike_times=(spike_ts,len(wp)) if spike_ts else None,
                        defuse_times=(defuse_ts,len(wp)) if defuse_ts else None,
                        swing_idx=swing_idx)
        # wrap with hover layer
        curve = f'<div class="curve-wrap" data-wp=\'{wp_json}\' data-times=\'{times_json}\' data-plant=\'{plant_json}\'>{curve_svg}<div class="vline"></div><div class="tip"></div></div>'
        before_after=""
        if len(swing_idx):
            diffs=np.abs(np.diff(wp))
            bi=np.argmax(diffs)
            before_after=f"P(A) {wp[bi]:.2f} → {wp[bi+1]:.2f} @ {clock((start_ms+ (bi+1)*TICK_MS)/1000 - start_ms/1000)}"
        cards.append(f"<div class='card'><div class='card-head'><span class='card-title'>R{r['roundNum']} · {html.escape(r['map'])} · winner={winner}{spike_label}</span><span class='badge badge-{winner}'>{winner}</span></div>{curve}<div class='meta'>{before_after} · red=kills green=plant blue=defuse yellow=swing (≥0.12) · hover for M:SS</div></div>")

    hover_js = r"""
<script>
function fmt(sec){ sec=Math.max(0,Math.round(sec)); return Math.floor(sec/60)+':'+String(sec%60).padStart(2,'0'); }
document.querySelectorAll('.curve-wrap').forEach(wrap=>{
  const wp=JSON.parse(wrap.dataset.wp);
  const times=JSON.parse(wrap.dataset.times);
  const plant=wrap.dataset.plant==='null'?null:parseFloat(wrap.dataset.plant);
  const tip=wrap.querySelector('.tip');
  const vline=wrap.querySelector('.vline');
  const svg=wrap.querySelector('svg');
  const n=wp.length;
  const pad=10, W=760;
  function showAt(clientX){
    const rect=svg.getBoundingClientRect();
    const x=Math.max(pad, Math.min(W-pad, (clientX - rect.left)/rect.width * W));
    const idx=Math.max(0,Math.min(n-1, Math.round((x-pad)/(W-2*pad)*(n-1))));
    const p=wp[idx];
    const elapsed=times[idx];
    let line1, line2, spikeCls='';
    if(plant!==null && elapsed>=plant){
      const spikeRem=Math.max(0,45-(elapsed-plant));
      const spikeElapsed=elapsed-plant;
      line1=`<span class="t-main" style="color:#fbbf24">SPIKE ${fmt(spikeRem)}</span> <span class="t-sub">(+${fmt(spikeElapsed)} elapsed)</span>`;
      spikeCls=' spike';
    } else {
      const remain=Math.max(0,100 - elapsed);
      line1=`<span class="t-main">${fmt(remain)}</span> <span class="t-sub">remaining · ${fmt(elapsed)} elapsed</span>`;
    }
    const delta = idx>0 ? (p - wp[idx-1]) : 0;
    const dstr = idx>0 ? (delta>=0?` <span style="color:#4ade80">▲${(delta*100).toFixed(1)}%</span>`:` <span style="color:#f87171">▼${(Math.abs(delta)*100).toFixed(1)}%</span>`) : '';
    line2=`P(A) ${(p*100).toFixed(1)}%${dstr}`;
    tip.innerHTML=line1+`<br>`+line2;
    tip.className='tip'+spikeCls;
    tip.style.left=x+'px';
    tip.style.top='8px';
    tip.style.display='block';
    vline.style.left=x+'px';
    vline.style.display='block';
  }
  wrap.addEventListener('mousemove', e=>showAt(e.clientX));
  wrap.addEventListener('mouseleave', ()=>{ tip.style.display='none'; vline.style.display='none'; });
  wrap.addEventListener('touchmove', e=>{ if(e.touches[0]) showAt(e.touches[0].clientX); }, {passive:true});
});
</script>
"""
    html_doc=f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><title>{html.escape(title)} — rib-eval</title><style>{CSS}</style></head><body>
<h1>{html.escape(title)} <span style="color:#8fa3ad;font-size:14px">· {sub_id} · {len(cards)} rounds</span></h1>
<p><a href=\"/\">← Back to report</a> · <a href=\"https://rib.gg/matches/{match_id}\" target=\"_blank\">rib.gg #{match_id}</a></p>
<div class="legend"><span><span class="swatch" style="background:#38bdf8"></span>P(A) curve</span><span><span class="swatch" style="background:#ff4655;opacity:.35"></span>kills</span><span><span class="swatch" style="background:#22c55e;opacity:.5"></span>plant</span><span><span class="swatch" style="background:#60a5fa;opacity:.5"></span>defuse</span><span><span class="swatch" style="background:#fbbf24"></span>swing</span></div>
<div class="grid">{''.join(cards)}</div>
<p class="meta" style="margin-top:20px">Model: pooled Transformer (holdout AUC≈0.87, F=18) · TICK=5s · Generated at tick level. Timeout rounds (freezetime>90s) excluded.</p>
{hover_js}
</body></html>"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html_doc, encoding="utf-8")
    print(f"match {match_id}: {len(cards)} rounds -> {out_path}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("match_id", nargs="?", default="577")
    ap.add_argument("-o","--out", default=None)
    args=ap.parse_args()
    out=args.out or str(OUT / f"match_{args.match_id}.html")
    build_one(args.match_id, out)
