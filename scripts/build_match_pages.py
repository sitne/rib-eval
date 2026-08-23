#!/usr/bin/env python3
import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch

from train_transformer import WinPredictor, predict
from wpa_analysis import clock, svg_curve, fmt_date, match_label, player_name, TICK_MS

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
    # team names for human-readable winner (A = first team in title)
    def _team_names(mid):
        t = (match_info.get(str(mid), {}) or {}).get("title") or ""
        if " vs " in t:
            a, b = t.split(" vs ", 1)
            for sep in [" – ", " — ", " - ", " | "]:
                if sep in b:
                    b = b.split(sep, 1)[0]
                    break
            return a.strip(), b.strip()
        return "Team A", "Team B"
    teamA, teamB = _team_names(match_id)
    # natural map order: m1, m2, m3 (haven → summit …)
    map_order = {}
    for i, pf in enumerate(sorted(DATA.glob(f"replays/{match_id}-m*.json"))):
        try:
            m = json.loads(pf.read_text())["replayData"]["map"]
        except Exception:
            continue
        if m not in map_order:
            map_order[m] = i
    # all rounds from raw (including timeout-excluded ones) for complete display
    all_raw=[]
    for pf in sorted(DATA.glob(f"replays/{match_id}-m*.json")):
        try:
            prd=json.loads(pf.read_text())["replayData"]
        except Exception:
            continue
        for rnd in prd["rounds"]:
            all_raw.append((prd["map"], rnd["roundNum"], rnd))
    all_raw.sort(key=lambda x: (map_order.get(x[0],99), x[1]))
    # per-map score for headers (include all rounds, even timeout)
    from collections import Counter
    map_scores = Counter()
    for m, rn, rr in all_raw:
        w = rr.get("winner")
        if w in ("A","B"):
            map_scores[m + ":" + w] += 1
    idxs.sort(key=lambda i: (map_order.get(metas[i]["map"], 99), metas[i]["roundNum"]))
    lookup={(metas[idx]["map"], metas[idx]["roundNum"]): j for j, idx in enumerate(idxs)}
    # collect cards grouped by map
    cards=[]
    cur_map = None
    round_cnt = 0
    for map_name, round_num, rnd_raw in all_raw:
        # map header
        if map_name != cur_map:
            cur_map = map_name
            wA = map_scores.get(cur_map+":A",0)
            wB = map_scores.get(cur_map+":B",0)
            oi = map_order.get(cur_map,0)+1
            # totals include timeout rounds for header score
            totA = sum(1 for m,r,_ in all_raw if m==cur_map and r==rnd_raw["roundNum"] and False)
            cards.append(f"<h2 style='grid-column:1/-1;margin:18px 0 2px;color:#8fa3ad'>Map {oi} · {html.escape(cur_map)} — {html.escape(teamA)} {wA}–{wB} {html.escape(teamB)}</h2>")
        j = lookup.get((map_name, round_num))
        if j is not None:
            r=metas[idxs[j]]
            valid=masks[j].astype(bool)
            wp=W[j][valid]
            n=len(wp)
            if n<2:
                continue
            start_ms=r["startMs"]
            kill_ts=[(k["t"]-start_ms)/TICK_MS for k in r["kills"] if k["t"]>start_ms]
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
            attacker="A" if at_sub[j]==1 else "B"
            winnerTeam = teamA if winner=="A" else teamB
            winnerSide = "ATK" if winner==attacker else "DEF"
            swing_idx=list(np.where(np.abs(np.diff(wp))>=0.12)[0]+1)
            elapsed = [(i*TICK_MS)/1000 for i in range(n)]
            plant_elapsed = spike_ts[0]*TICK_MS/1000 if spike_ts else None
            wp_json = json.dumps([round(float(v),4) for v in wp])
            times_json = json.dumps(elapsed)
            plant_json = str(plant_elapsed) if plant_elapsed is not None else "null"
            # kill details for hover near kill
            kill_details=[]
            for k in r["kills"]:
                if k["t"] <= start_ms: continue
                tu=(k["t"]-start_ms)/TICK_MS
                xk=10 + tu/len(wp)*(760-20)
                kill_details.append({"x": round(xk,1), "killer": player_name(r["roster"], k["killer"]), "victim": player_name(r["roster"], k["victim"]), "t": round((k["t"]-start_ms)/1000,1)})
            kills_detail_json = json.dumps(kill_details, ensure_ascii=False).replace("'", "&#39;")
            curve_svg=svg_curve(wp, kill_times=(kill_ts,len(wp)) if kill_ts else None,
                            spike_times=(spike_ts,len(wp)) if spike_ts else None,
                            defuse_times=(defuse_ts,len(wp)) if defuse_ts else None,
                            swing_idx=swing_idx)
            curve = f'<div class="curve-wrap" data-wp=\'{wp_json}\' data-times=\'{times_json}\' data-plant=\'{plant_json}\' data-kills-detail=\'{kills_detail_json}\'>{curve_svg}<div class="vline"></div><div class="tip"></div></div>'
            before_after=""
            if len(swing_idx):
                diffs=np.abs(np.diff(wp))
                bi=np.argmax(diffs)
                before_after=f"P(A) {wp[bi]:.2f} → {wp[bi+1]:.2f} @ {clock((start_ms+ (bi+1)*TICK_MS)/1000 - start_ms/1000)}"
            cards.append(f"<div class='card'><div class='card-head'><span class='card-title'>R{round_num} · {html.escape(winnerTeam)} won <span style='color:#8fa3ad;font-weight:400'>({winnerSide})</span> · {html.escape(map_name)}{spike_label}</span><span class='badge badge-{winner}'>{winnerSide}</span></div>{curve}<div class='meta'>{before_after} · red=kills green=plant blue=defuse yellow=swing (≥0.12) · hover for M:SS</div></div>")
            round_cnt += 1
        else:
            ft = rnd_raw.get("freezetimeEndT") or 0
            dur = rnd_raw.get("durationMs") or 0
            win = rnd_raw.get("winner") or "?"
            winTeam = teamA if win=="A" else teamB if win=="B" else "?"
            reason = "timeout" if ft and ft>90000 else "filtered"
            cards.append(f"<div class='card' style='opacity:0.55'><div class='card-head'><span class='card-title'>R{round_num} · {html.escape(map_name)} · {html.escape(winTeam)} won</span><span class='badge badge-{win}'>{win}</span></div><div style='height:150px;display:flex;align-items:center;justify-content:center;background:#0b1319;border-radius:8px;color:#8fa3ad;font-size:12px;text-align:center'>Timeout — win prob not computed<br><span style='font-size:11px'>freezetime {ft/1000:.0f}s · dur {dur/1000:.0f}s · {reason}</span></div><div class='meta'>winType {rnd_raw.get('winType')} · raw events {len(rnd_raw.get('events',[]))}</div></div>")
            round_cnt += 1

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
    const xRaw=Math.max(pad, Math.min(W-pad, (clientX - rect.left)/rect.width * W));
    let idx=Math.max(0,Math.min(n-1, Math.round((xRaw-pad)/(W-2*pad)*(n-1))));
    let x = pad + idx/(n>1?n-1:1)*(W-2*pad);
    // snap to nearest kill if close
    try{
      const kills=JSON.parse(wrap.dataset.killsDetail||"[]");
      let best=null, bestDist=30;
      for(const k of kills){
        const d=Math.abs(k.x - xRaw);
        if(d<bestDist){ bestDist=d; best=k; }
      }
      if(best){
        x=best.x;
        idx=Math.max(0,Math.min(n-1, Math.round((x-pad)/(W-2*pad)*(n-1))));
      }
    }catch(e){}
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
    let killInfo='';
    try{
      const kills=JSON.parse(wrap.dataset.killsDetail||"[]");
      let bestK=null, bestD=30;
      const curX = pad + idx/(n>1?n-1:1)*(W-2*pad);
      for(const k of kills){
        const d=Math.abs(k.x - curX);
        if(d<bestD){ bestD=d; bestK=k; }
      }
      if(bestK) killInfo=` · <span style="color:#f87171">${bestK.killer} → ${bestK.victim}</span> @${bestK.t}s`;
    }catch(e){}
    line2=`P(A) ${(p*100).toFixed(1)}%${dstr}${killInfo}`;
    tip.innerHTML=line1+`<br>`+line2;
    tip.className='tip'+spikeCls;
    tip.style.left=(x/W*100)+'%';
    tip.style.top='8px';
    tip.style.display='block';
    vline.style.left=(x/W*100)+'%';
    vline.style.display='block';
  }
  wrap.addEventListener('mousemove', e=>showAt(e.clientX));
  wrap.addEventListener('mouseleave', ()=>{ tip.style.display='none'; vline.style.display='none'; });
  wrap.addEventListener('touchmove', e=>{ if(e.touches[0]) showAt(e.touches[0].clientX); }, {passive:true});
});
</script>
"""
    html_doc=f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><title>{html.escape(title)} — rib-eval</title><style>{CSS}</style></head><body>
<h1>{html.escape(title)} <span style="color:#8fa3ad;font-size:14px">· {sub_id} · {round_cnt} rounds</span></h1>
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
