import modal

app = modal.App("rib-eval-all-skills")

# Base image with Python deps + match list
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy==2.4.6",
        "scikit-learn==1.9.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_file("data/match_info.json", "/root/match_info.json")
)

# Volume for data (replays, sequences, outputs)
volume = modal.Volume.from_name("rib-eval-data", create_if_missing=True)
VOL_MOUNT = "/data"
# Bundle match_info to avoid scanning 1..900 blindly
import pathlib as _pl
_match_info_path = _pl.Path(__file__).parent / "data" / "match_info.json"

@app.function(
    image=image,
    gpu="A10G",
    memory=32768,
    timeout=3600,
    volumes={VOL_MOUNT: volume},
)
def train_all_skills(epochs: int = 20, batch: int = 256):
    import json, sys
    from pathlib import Path

    # Mounted data is at /data, but repo is not mounted – we need to build from scratch?
    # Instead, assume replays are already in volume at /data/replays
    # For now, we rebuild sequences with per-ability independent (240 dims) inside cloud
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    # --- Build sequences with per-ability independent (120*2) ---
    # Load ability list
    import pathlib, json as js
    # Volume path for replays - use existing volume (already has 1133 files), no re-fetch needed
    CACHE = Path(f"{VOL_MOUNT}/replays")
    print(f"replays in volume: {len(list(CACHE.glob('*.json')))} files", flush=True)

    # Load ability mapping (per-ability independent)
    # Use same logic as local: each rib ability is its own feature
    from collections import Counter
    all_abs = set()
    for p in CACHE.glob("*.json"):
        try:
            rd = js.loads(p.read_text()).get("replayData", js.loads(p.read_text()))
            for r in rd.get("rounds", []):
                for e in r.get("events", []):
                    if e.get("type") == "ability":
                        all_abs.add(e["ability"])
        except:
            pass
    ABILITIES = sorted(all_abs)
    print(f"ABILITIES {len(ABILITIES)} -> F {18+len(ABILITIES)*2+2}")

    # Import build logic (replicate build_sequence)
    import math
    from pathlib import Path as P

    TICK_SEC = 5.0
    TMAX = 44
    NPLAYERS = 10
    WEAPON_TIER = {"vandal":0,"phantom":0,"bulldog":0,"guardian":0,"bandit":0,"odin":0,"operator":1,"marshal":1,"outlaw":1,"stinger":2,"spectre":2,"judge":3,"bucky":3,"classic":4,"ghost":4,"sheriff":4,"frenzy":4,"shorty":4}
    N_WEAPON_TIERS = 6
    FREEZE_TIMEOUT_CUTOFF_MS = 90000
    F = 18 + len(ABILITIES)*2 + 2

    def view_angle_sin_cos(vv):
        n = math.hypot(vv["x"], vv["y"]) or 1.0
        return vv["x"]/n, vv["y"]/n
    def state_at(states, t_ms):
        cur=None
        for s in states:
            if s["t"] <= t_ms: cur=s
            else: break
        return cur
    def snap_at(ev_by_actor, actor, t_ms, max_age=6000):
        snaps=ev_by_actor.get(actor)
        if not snaps: return None
        lo,hi,best=0,len(snaps)-1,None
        while lo<=hi:
            mid=(lo+hi)//2
            if snaps[mid]["t"] <= t_ms:
                best=snaps[mid]; lo=mid+1
            else: hi=mid-1
        if best is None or t_ms - best["t"] > max_age: return None
        return best

    # Build sequences (simplified, same as build_sequence.py but with per-ability)
    MAPS = ["ascent","split","haven","sunset","summit","lotus","breeze","pearl","fracture","bind","abyss"]
    abil_to_idx = {a:i for i,a in enumerate(ABILITIES)}
    pit_idxs = [abil_to_idx.get("vipers-pit"), abil_to_idx.get("viper's-pit")]
    pit_idxs = [i for i in pit_idxs if i is not None]

    X, M, Y, AT, maps, matches = [], [], [], [], [], []
    for f in sorted(CACHE.glob("*.json")):
        try:
            rd = js.loads(f.read_text())["replayData"]
        except: continue
        if rd["map"] not in MAPS: continue
        bminx, bminy = rd["bounds"]["min"]["x"], rd["bounds"]["min"]["y"]
        span_x = (rd["bounds"]["max"]["x"] - bminx) or 1.0
        span_y = (rd["bounds"]["max"]["y"] - bminy) or 1.0
        roster = rd["roster"]
        actors = sorted(roster.keys(), key=int)
        if len(actors)!=NPLAYERS: continue
        match_id = f.stem.split("-")[0]
        for rnd in rd["rounds"]:
            winner = rnd["winner"]
            if winner not in ("A","B"): continue
            duration = rnd["durationMs"]
            freeze_end = rnd.get("freezetimeEndT") or 0
            if freeze_end > FREEZE_TIMEOUT_CUTOFF_MS: continue
            plant_ms = next((e["t"] for e in rnd["events"] if e["type"]=="plant"), None)
            ev_by_actor={}
            ability_events=[]
            for e in rnd["events"]:
                if e["type"]=="snapshot":
                    ev_by_actor.setdefault(e["actorId"], []).append(e)
                elif e["type"]=="ability":
                    ability_events.append(e)
            for lst in ev_by_actor.values(): lst.sort(key=lambda e:e["t"])
            ability_events.sort(key=lambda e:e["t"])
            states={a: rnd["playerStates"].get(a,[]) for a in actors}
            seq=np.zeros((TMAX,NPLAYERS,F), dtype=np.float32)
            mask=np.zeros(TMAX, dtype=np.float32)
            t_sec=max(freeze_end/1000.0+TICK_SEC, TICK_SEC)
            k=0
            while t_sec*1000.0 < duration and k<TMAX:
                t_ms=t_sec*1000.0
                is_post=1.0 if (plant_ms is not None and t_ms>=plant_ms) else 0.0
                alive_flags=[]
                for actor in actors:
                    st=state_at(states[actor], t_ms)
                    alive_flags.append(bool(st and st.get("alive")))
                alive_a=sum(1 for f,a in zip(alive_flags, actors) if f and roster[a]["team"]=="A")
                alive_b=sum(1 for f,a in zip(alive_flags, actors) if f and roster[a]["team"]=="B")
                adv=(alive_a-alive_b)/5.0
                cnt_a=[0.0]*len(ABILITIES)
                cnt_b=[0.0]*len(ABILITIES)
                for ab in ability_events:
                    if ab["t"]>t_ms: break
                    idx=abil_to_idx.get(ab.get("ability"))
                    if idx is None: continue
                    team=roster.get(str(ab.get("actorId")),{}).get("team")
                    if team=="A": cnt_a[idx]+=1.0
                    elif team=="B": cnt_b[idx]+=1.0
                cnt_a=[c/5.0 for c in cnt_a]
                cnt_b=[c/5.0 for c in cnt_b]
                pit_a=sum(cnt_a[i] for i in pit_idxs) if pit_idxs else 0.0
                pit_b=sum(cnt_b[i] for i in pit_idxs) if pit_idxs else 0.0
                ability_feat=cnt_a+cnt_b+[is_post*pit_a, is_post*pit_b]
                for pi, actor in enumerate(actors):
                    st=state_at(states[actor], t_ms)
                    alive=alive_flags[pi]
                    hp=(st or {}).get("health",0)/100.0
                    armor=(st or {}).get("armor",0)/50.0
                    eco=(st or {}).get("loadoutValue",0)/20000.0
                    money=(st or {}).get("money",0)/9000.0
                    wname=(st or {}).get("weapon")
                    tier=WEAPON_TIER.get(wname,5) if wname else 5
                    w_onehot=[0.0]*N_WEAPON_TIERS
                    w_onehot[tier]=1.0
                    sn=snap_at(ev_by_actor, actor, t_ms)
                    if sn and alive:
                        xn=(sn["pos"]["x"]-bminx)/span_x
                        yn=(sn["pos"]["y"]-bminy)/span_y
                        sx,sy=view_angle_sin_cos(sn["viewVector"])
                    else:
                        xn=yn=sx=sy=0.0
                    base=(xn,yn,sx,sy,1.0 if alive else 0.0,hp,armor,eco,1.0 if roster[actor]["team"]=="A" else 0.0,money,adv,is_post)
                    feat=base+tuple(w_onehot)+tuple(ability_feat)
                    seq[k,pi]=feat
                mask[k]=1.0
                k+=1
                t_sec+=TICK_SEC
            if k<2: continue
            X.append(seq); M.append(mask); Y.append(1 if winner=="A" else 0); AT.append(1 if rnd["attackerTeam"]=="A" else 0); maps.append(rd["map"]); matches.append(match_id)
    Xa=np.stack(X); Ma=np.stack(M)
    print(f"built {Xa.shape} F={F}")
    # Train (pooled)
    TMAX_ = 44
    class WinPredictor(nn.Module):
        def __init__(self, f, d=128, layers=3, heads=4, dropout=0.1):
            super().__init__()
            self.proj=nn.Linear(f,d)
            self.pos=nn.Embedding(TMAX_,d)
            self.atk=nn.Embedding(2,d)
            layer=nn.TransformerEncoderLayer(d,heads,dim_feedforward=d*4,dropout=dropout,batch_first=True)
            self.spatial=nn.TransformerEncoder(layer,layers)
            self.gru=nn.GRU(d,d,batch_first=True)
            self.head=nn.Linear(d,1)
        def forward(self,x,at):
            B,T,P,_=x.shape
            h=self.proj(x)+self.pos.weight[:T].view(1,T,1,-1)
            atk_e=self.atk(at)[:,None,None,:].expand(-1,T,1,-1)
            h=h.view(B*T,P,-1)+atk_e.reshape(B*T,1,-1)
            h=self.spatial(h).view(B,T,P,-1)
            alive=x[...,4:5]
            pooled=(h*alive).sum(2)/alive.sum(2).clamp(min=1.0)
            out,_=self.gru(pooled)
            return self.head(out).squeeze(-1)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev} F {F}")
    # Split
    import numpy as _np
    uniq=sorted(set(matches))
    import random
    random.seed(42); random.shuffle(uniq)
    split=int(len(uniq)*0.8)
    tr_m, va_m=set(uniq[:split]), set(uniq[split:])
    tr=_np.array([m in tr_m for m in matches])
    va=~tr
    Xs,Ms,ys,ats=Xa[tr],Ma[tr],_np.array(Y)[tr],_np.array(AT)[tr]
    Xv,Mv,yv,av=Xa[va],Ma[va],_np.array(Y)[va],_np.array(AT)[va]
    xt=torch.tensor(Xs,device=dev); mt=torch.tensor(Ms,device=dev); yt=torch.tensor(ys,dtype=torch.float32,device=dev); att=torch.tensor(ats.astype(_np.int64),device=dev)
    xv=torch.tensor(Xv,device=dev); mv=torch.tensor(Mv,device=dev)
    model=WinPredictor(f=F).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(reduction="none")
    best_auc=0; best_state=None
    n=len(xt)
    for ep in range(epochs):
        model.train()
        perm=torch.randperm(n,device=dev)
        tot=0.0
        for i in range(0,n,512):
            idx=perm[i:i+512]
            logits=model(xt[idx],att[idx])
            l=lossf(logits, yt[idx][:,None].expand_as(logits))
            l=(l*mt[idx]).sum()/mt[idx].sum()
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tot+=float(l)*len(idx)
        model.eval()
        with torch.no_grad():
            outs=[]
            for i in range(0,len(Xv),512):
                outs.append(model(xv[i:i+512], torch.tensor(av[i:i+512].astype(_np.int64),device=dev)).cpu().numpy())
            vl=np.concatenate(outs)
        from sklearn.metrics import roc_auc_score
        vmask=mv.cpu().numpy().astype(bool)
        auc=roc_auc_score(np.repeat(yv[:,None],TMAX_,1)[vmask], vl[vmask])
        print(f"epoch {ep}: loss {tot/n:.4f} val_auc {auc:.4f}")
        if auc>best_auc:
            best_auc=auc; best_state={k:v.clone() for k,v in model.state_dict().items()}
    print(f"BEST {best_auc:.4f}")
    # Save to volume
    out_path=Path(f"{VOL_MOUNT}/outputs/transformer_pooled_240.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)
    print(f"saved to {out_path}")
    return float(best_auc)

@app.local_entrypoint()
def main(epochs: int = 20):
    # Upload local replays to volume if needed (one-time)
    # For now, assume volume already has replays; if not, user should run: modal volume put rib-eval-data data/replays /replays
    train_all_skills.remote(epochs=epochs)
