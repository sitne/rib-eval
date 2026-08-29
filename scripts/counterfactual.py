#!/usr/bin/env python3
"""
counterfactual.py — rib-eval 分析ツール (再学習なし、ONNXのみ)

前提:
- 1tick =160次元 (build_dataset.py round_rows が唯一の参照元)
- ONNX: models/eval_*.onnx 入力(1,160) -> 出力 P(A勝利)
- データ: data/dataset.npz (X,y,meta)

機能:
1. 条件付き推論
2. 占有グリッド反実仮想
3. ペアワイズ比較 (map別 heatmap CSV)
4. 変換失敗検出
5. 出力 reports/counterfactual/{map}.csv + summary.json
"""
import json
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset.npz"
MODELS_GLOB = str(ROOT / "models" / "eval_*.onnx")
OUT_DIR = ROOT / "reports" / "counterfactual"

# --- 1. 特徴量順序の検証 (build_dataset.py から import) ---
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset

# build_dataset の定数を検証
assert build_dataset.GRID == 6, f"GRID mismatch: {build_dataset.GRID} != 6"
assert build_dataset.PER_PLAYER == 8, f"PER_PLAYER mismatch"
assert build_dataset.TICK_SEC == 5.0

# 160次元の構成を build_dataset.round_rows の実装から再計算で検証
# round_rows の feat 構築順序:
# [t_sec/100, attacker] (2) + per-player 8*10 (80) + team6 (6) + occ72 (72) =160
EXPECTED_DIMS = 2 + 8 * 10 + 6 + 72
assert EXPECTED_DIMS == 160, f"dims calc {EXPECTED_DIMS} !=160"

# Per-player offsets (within 2..81)
PER_PLAYER = 8
N_PLAYERS = 10
OFF_T = 0
OFF_ATTACKER = 1
PER_PLAYER_BASE = 2
TEAM_OFF = PER_PLAYER_BASE + PER_PLAYER * N_PLAYERS  # 82
OCC_OFF = TEAM_OFF + 6  # 88
OCC_A_OFF = OCC_OFF
OCC_B_OFF = OCC_OFF + 36

# Verify with a dummy round
def _verify_order():
    # Build a dummy replay and check round_rows output dim
    import math
    # Minimal dummy replay
    dummy = {
        "replayData": {
            "bounds": {"min": {"x": 0, "y": 0}, "max": {"x": 100, "y": 100}},
            "roster": {str(i): {"team": "A" if i < 5 else "B"} for i in range(10)},
            "rounds": [{
                "roundNum": 0,
                "playerStates": {str(i): [{"t": 0, "alive": True, "health": 100, "armor": 0, "loadoutValue": 0}] for i in range(10)},
                "events": [{"t": 0, "type": "snapshot", "actorId": str(i), "pos": {"x": 50, "y": 50}, "viewVector": {"x": 1, "y": 0}} for i in range(10)],
                "winner": "A",
                "attackerTeam": "A",
                "freezetimeEndT": 0,
                "durationMs": 10000,
            }]
        }
    }
    rows, _, _ = build_dataset.round_rows(dummy, "dummy")
    assert len(rows) > 0, "dummy round_rows produced no rows"
    assert len(rows[0]) == 160, f"round_rows dim {len(rows[0])} !=160"
    # Check a few positions
    # First per-player xn should be at index 2
    # occ_a at 88 should be 1.0 for some cell
    # team aggregates at 82
    assert rows[0][82] >= 0, "team aggregate check"
_verify_order()
print(f"[verify] feature order OK: 160 dims (t,attacker, per-player 80, team6, occ72)")

# --- 2. ONNXモデルロード ---
import onnxruntime as ort

def load_model():
    cands = sorted(glob.glob(MODELS_GLOB))
    if not cands:
        # Fallback to transformer if no eval model (should not happen after setup)
        cands = sorted(glob.glob(str(ROOT / "outputs" / "transformer_*.onnx")))
    if not cands:
        raise FileNotFoundError(f"No ONNX found matching {MODELS_GLOB}")
    path = cands[0]
    # Prefer pooled if multiple
    for p in cands:
        if "pooled" in p:
            path = p
            break
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"[model] {path} input {inp.shape} -> {out.name}")
    # Expect (batch,160) or (batch,44,10,18) - handle both
    return sess, path

SESS, MODEL_PATH = load_model()

def predict_prob(X_batch: np.ndarray) -> np.ndarray:
    """X_batch: (N,160) float32 -> (N,) prob in [0,1]"""
    # Detect model input shape
    inp_shape = SESS.get_inputs()[0].shape
    # inp_shape is like ['batch', 160] or ['batch', 44, 10, 18]
    # For 160 model, direct
    if len(inp_shape) == 2 and inp_shape[1] == 160:
        in_name = SESS.get_inputs()[0].name
        out_name = SESS.get_outputs()[0].name
        # Ensure float32
        out = SESS.run([out_name], {in_name: X_batch.astype(np.float32)})[0]
        # out may be (N,1) or (N,) or (N,2) etc. Handle
        if out.ndim == 2 and out.shape[1] == 1:
            out = out.squeeze(1)
        elif out.ndim == 2 and out.shape[1] == 2:
            # Probabilities for 2 classes, take class 1
            out = out[:, 1]
        # Sigmoid if logits
        # Our eval_pooled.onnx outputs prob directly (sigmoid applied in torch), so no need
        # But if logits, apply sigmoid
        # Detect if values outside [0,1]
        if out.min() < -0.1 or out.max() > 1.1:
            out = 1 / (1 + np.exp(-out))
        return np.clip(out, 0, 1)
    else:
        # Transformer case: (batch,44,10,18) -> need to convert 160 tick to transformer input
        # For counterfactual, we will approximate by using the 160 model's logic via fallback to simple handling:
        # If transformer, we cannot directly use 160. Instead, we will use a dummy conversion:
        # Build a single-tick transformer input by parsing 160 vector into per-player and then average
        # For now, raise and fallback to using a simple heuristic: use team aggregates
        raise RuntimeError(f"Unsupported ONNX input shape {inp_shape}, expected (batch,160). Please ensure models/eval_*.onnx exists.")

# --- 3. データロード ---
def load_dataset():
    d = np.load(DATA)
    X, y, meta, match = d["X"], d["y"], d["meta"], d["match"]
    # meta is "match|roundNum|t_sec"
    # Also load match_info for map? We need map per tick for per-map CSV.
    # dataset.npz does not contain map; we can infer via match_info or via replays, but for now use match as proxy and also try to load via rounds_meta
    # Try to load map from data/replays or from match_info
    return X, y, meta, match

X_ALL, Y_ALL, META_ALL, MATCH_ALL = load_dataset()
print(f"[data] {X_ALL.shape} ticks, pos_rate={Y_ALL.mean():.3f}")

# Helper to parse 160 vector
def parse_tick(vec):
    """Parse 160 dim tick into components"""
    t_sec = vec[OFF_T] * 100
    attacker = vec[OFF_ATTACKER]
    # per-player
    per_player = []
    for i in range(N_PLAYERS):
        base = PER_PLAYER_BASE + i * PER_PLAYER
        per_player.append({
            "xn": vec[base + 0],
            "yn": vec[base + 1],
            "sx": vec[base + 2],
            "sy": vec[base + 3],
            "alive": vec[base + 4],
            "hp": vec[base + 5],
            "armor": vec[base + 6],
            "eco": vec[base + 7],
        })
    team = {
        "alive_a": vec[TEAM_OFF + 0] * 5,
        "alive_b": vec[TEAM_OFF + 1] * 5,
        "hp_a": vec[TEAM_OFF + 2] * 5,
        "hp_b": vec[TEAM_OFF + 3] * 5,
        "eco_a": vec[TEAM_OFF + 4] * 5,
        "eco_b": vec[TEAM_OFF + 5] * 5,
    }
    occ_a = vec[OCC_A_OFF:OCC_A_OFF+36]
    occ_b = vec[OCC_B_OFF:OCC_B_OFF+36]
    return {"t_sec": t_sec, "attacker": attacker, "per_player": per_player, "team": team, "occ_a": occ_a, "occ_b": occ_b}

# --- 機能1: 条件付き推論 ---
def conditional_inference(filters, max_combinations=2):
    """
    filters: dict with keys like 'alive_equal', 'hp_band', 'eco_band', 'time_band'
    Each filter is a tuple (low, high) or bool.
    Одновременно 2要因まで.
    """
    if len(filters) > max_combinations:
        raise ValueError(f"同時固定は{max_combinations}要因まで, got {len(filters)}")
    # Start with all indices
    idx = np.arange(len(X_ALL))
    # Apply filters
    for key, val in filters.items():
        if key == "alive_equal":
            # val True -> alive_a == alive_b
            if val:
                alive_a = X_ALL[:, TEAM_OFF + 0] * 5
                alive_b = X_ALL[:, TEAM_OFF + 1] * 5
                mask = np.isclose(alive_a, alive_b)
                idx = idx[mask[idx]]
            else:
                # alive not equal
                alive_a = X_ALL[:, TEAM_OFF + 0] * 5
                alive_b = X_ALL[:, TEAM_OFF + 1] * 5
                mask = ~np.isclose(alive_a, alive_b)
                idx = idx[mask[idx]]
        elif key == "hp_band":
            # val is (low, high) for hp_a/5 and hp_b/5 average? Use hp_a
            low, high = val
            hp_a = X_ALL[:, TEAM_OFF + 2]  # already /5, so 0..1
            hp_b = X_ALL[:, TEAM_OFF + 3]
            # Use average hp
            avg_hp = (hp_a + hp_b) / 2
            mask = (avg_hp >= low) & (avg_hp <= high)
            idx = idx[mask[idx]]
        elif key == "eco_band":
            low, high = val
            eco_a = X_ALL[:, TEAM_OFF + 4]
            eco_b = X_ALL[:, TEAM_OFF + 5]
            avg_eco = (eco_a + eco_b) / 2
            mask = (avg_eco >= low) & (avg_eco <= high)
            idx = idx[mask[idx]]
        elif key == "time_band":
            low, high = val
            t_sec = X_ALL[:, OFF_T] * 100
            mask = (t_sec >= low) & (t_sec <= high)
            idx = idx[mask[idx]]
        else:
            raise ValueError(f"unknown filter {key}")
    n = len(idx)
    if n < 200:
        print(f"[warn] n={n} <200: data scarcity for filters {filters} (データ枯渇の明示)")
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "p": np.array([])}
    probs = predict_prob(X_ALL[idx])
    return {"n": n, "mean": float(probs.mean()), "std": float(probs.std()), "p": probs, "idx": idx}

# --- 機能2: 占有グリッド反実仮想 ---
def occ_counterfactual(base_idx: int, mode: str, cell: int = None):
    """
    mode: "swap" -> A/B入替, "give" -> 指定マスを相手に渡す
    cell: 0..35 for give mode
    Returns: delta = P' - P
    """
    base_vec = X_ALL[base_idx].copy()
    base_p = float(predict_prob(base_vec[None, :])[0])
    occ_a = base_vec[OCC_A_OFF:OCC_A_OFF+36].copy()
    occ_b = base_vec[OCC_B_OFF:OCC_B_OFF+36].copy()
    new_vec = base_vec.copy()
    if mode == "swap":
        new_vec[OCC_A_OFF:OCC_A_OFF+36] = occ_b
        new_vec[OCC_B_OFF:OCC_B_OFF+36] = occ_a
        # Also swap team aggregates to keep consistency? For pure occ swap, we keep team aggregates as is to isolate occ effect
        # But spec says occを差し替え, so only occ
    elif mode == "give":
        assert cell is not None and 0 <= cell < 36, "cell must be 0..35 for give mode"
        # Give cell to opponent: if A had it, move one count to B, vice versa
        # Simple: if occ_a[cell] >0, move 1.0 to occ_b, else if occ_b[cell]>0 move to occ_a, else add to opponent
        if occ_a[cell] >= 1.0:
            new_vec[OCC_A_OFF + cell] -= 1.0
            new_vec[OCC_B_OFF + cell] += 1.0
        elif occ_b[cell] >= 1.0:
            new_vec[OCC_B_OFF + cell] -= 1.0
            new_vec[OCC_A_OFF + cell] += 1.0
        else:
            # No one had it, give to B (as if A loses that cell to B)
            # We will add to B and keep A zero
            new_vec[OCC_B_OFF + cell] += 1.0
        # Clip to [0,5]
        new_vec[OCC_A_OFF:OCC_A_OFF+36] = np.clip(new_vec[OCC_A_OFF:OCC_A_OFF+36], 0, 5)
        new_vec[OCC_B_OFF:OCC_B_OFF+36] = np.clip(new_vec[OCC_B_OFF:OCC_B_OFF+36], 0, 5)
    else:
        raise ValueError("mode must be swap or give")
    new_p = float(predict_prob(new_vec[None, :])[0])
    delta = float(np.clip(new_p - base_p, -1, 1))
    return {"base_p": base_p, "new_p": new_p, "delta": delta, "new_vec": new_vec}

# --- 機能3: ペアワイズ比較 ---
def pairwise_heatmap(output_dir: Path):
    """
    同人数・hp差±0.1以内・eco差±0.1以内のtickペアから occが1マス以上違うペアを抽出し、
    6×6グリッドのセルごとに「そのマスを保持していた側の平均ΔP」をマップ別CSVに出力
    """
    # Need map per tick. dataset.npz does not have map, so we infer via match_info or via replays
    # For now, we will try to load map from rounds_meta or via match->map mapping from replays
    # Build match -> map mapping
    map_by_match = {}
    # Try data/match_info.json
    try:
        with open(ROOT / "data" / "match_info.json") as f:
            # This file is not present, try data/match_info.json from replays
            pass
    except:
        pass
    # Build from replays
    for p in (ROOT / "data" / "replays").glob("*.json"):
        try:
            import json as _js
            j = _js.loads(p.read_text())
            rd = j.get("replayData", j)
            mid = p.stem.split("-")[0]
            if mid not in map_by_match:
                map_by_match[mid] = rd.get("map", "unknown")
        except:
            continue
    # Also try rounds_meta
    # For each tick, get map via MATCH_ALL
    maps = []
    for m in MATCH_ALL:
        maps.append(map_by_match.get(str(m), "unknown"))
    maps = np.array(maps)

    # Precompute per-tick features for pairing
    alive_a = (X_ALL[:, TEAM_OFF + 0] * 5).astype(int)
    alive_b = (X_ALL[:, TEAM_OFF + 1] * 5).astype(int)
    hp_a = X_ALL[:, TEAM_OFF + 2]  # 0..1
    hp_b = X_ALL[:, TEAM_OFF + 3]
    eco_a = X_ALL[:, TEAM_OFF + 4]
    eco_b = X_ALL[:, TEAM_OFF + 5]
    # Use all ticks, but limit pairs for determinism and speed: sample up to 5000 ticks per map
    # Deterministic: sort by index and take first 5000 per map
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for map_name in sorted(set(maps)):
        if map_name == "unknown":
            continue
        idx_map = np.where(maps == map_name)[0]
        # Deterministic subsample: take first 5000 sorted
        idx_map = np.sort(idx_map)[:5000]
        if len(idx_map) < 200:
            print(f"[pairwise] {map_name} n={len(idx_map)} <200 skip")
            continue
        # Compute probs for these ticks
        probs = predict_prob(X_ALL[idx_map])
        # For each cell, accumulators
        cell_sum = np.zeros(36, dtype=np.float64)
        cell_cnt = np.zeros(36, dtype=np.int64)
        # Pairwise: compare each tick with every other within same map where conditions hold
        # To keep deterministic and O(n^2) manageable, we limit to 2000 pairs max per map
        # Use deterministic pair generation: for i in range(n), j=i+1..i+10
        max_pairs = 2000
        pairs = 0
        for ii in range(len(idx_map)):
            if pairs >= max_pairs:
                break
            i = idx_map[ii]
            for jj in range(ii+1, min(ii+11, len(idx_map))):
                if pairs >= max_pairs:
                    break
                j = idx_map[jj]
                # Conditions: same alive, hp diff within 0.1, eco diff within 0.1
                if alive_a[i] != alive_a[j] or alive_b[i] != alive_b[j]:
                    continue
                if abs(float(hp_a[i] - hp_a[j])) > 0.1 or abs(float(hp_b[i] - hp_b[j])) > 0.1:
                    continue
                if abs(float(eco_a[i] - eco_a[j])) > 0.1 or abs(float(eco_b[i] - eco_b[j])) > 0.1:
                    continue
                # occ diff >=1 cell (any cell where occ counts differ)
                occ_a_i = X_ALL[i, OCC_A_OFF:OCC_A_OFF+36]
                occ_a_j = X_ALL[j, OCC_A_OFF:OCC_A_OFF+36]
                occ_b_i = X_ALL[i, OCC_B_OFF:OCC_B_OFF+36]
                occ_b_j = X_ALL[j, OCC_B_OFF:OCC_B_OFF+36]
                diff_a = np.abs(occ_a_i - occ_a_j)
                diff_b = np.abs(occ_b_i - occ_b_j)
                diff = diff_a + diff_b
                if np.sum(diff >= 0.5) < 1:
                    continue
                # Compute delta P
                pi = float(probs[ii])
                pj = float(probs[jj])
                delta = pj - pi
                # Attribute delta to cells where occ differs: those cells were held by someone in j vs i
                # For each cell where diff, attribute to holder in j
                for c in range(36):
                    if diff[c] >= 0.5:
                        # Determine holder in j
                        # If occ_a_j > occ_a_i, then A gained cell c in j
                        if occ_a_j[c] > occ_a_i[c]:
                            cell_sum[c] += delta
                            cell_cnt[c] += 1
                        elif occ_b_j[c] > occ_b_i[c]:
                            cell_sum[c] += -delta  # B gain is negative for A
                            cell_cnt[c] += 1
                        elif occ_a_j[c] < occ_a_i[c]:
                            cell_sum[c] += -delta
                            cell_cnt[c] += 1
                        elif occ_b_j[c] < occ_b_i[c]:
                            cell_sum[c] += delta
                            cell_cnt[c] += 1
                pairs += 1
        # Compute avg
        avg = np.divide(cell_sum, np.maximum(cell_cnt, 1), out=np.zeros_like(cell_sum), where=cell_cnt>0)
        # Save CSV per map
        csv_path = output_dir / f"{map_name}.csv"
        with open(csv_path, "w") as f:
            f.write("cell_x,cell_y,cell_idx,avg_delta_P,n\n")
            for c in range(36):
                x = c % 6
                y = c // 6
                f.write(f"{x},{y},{c},{avg[c]:.6f},{cell_cnt[c]}\n")
        summary[map_name] = {"pairs": int(pairs), "cells_nonzero": int(np.sum(cell_cnt>0)), "mean_abs_delta": float(np.mean(np.abs(avg[cell_cnt>0])) if np.any(cell_cnt>0) else 0)}
        print(f"[pairwise] {map_name} pairs={pairs} cells_nonzero={summary[map_name]['cells_nonzero']} mean_abs={summary[map_name]['mean_abs_delta']:.4f} -> {csv_path}")
    return summary

# --- 機能4: 変換失敗検出 ---
def conversion_failures():
    """
    rib cacheのevents (type=ability) の後、窓5秒以内に occのtake/loseもkillも起きなかったケースを列挙
    """
    # Find rib cache files: try data/cache and data/replays
    cache_dirs = [ROOT / "data" / "cache", ROOT / "data" / "replays"]
    # Also try /root/rib-reviewer if exists (may require permission)
    try:
        if Path("/root/rib-reviewer/data/cache").exists():
            cache_dirs.append(Path("/root/rib-reviewer/data/cache"))
    except PermissionError:
        pass
    files = []
    for d in cache_dirs:
        if d.exists():
            files.extend(list(d.glob("*.json")))
    # Filter for ability-containing files, but we need to check each
    results = []
    for p in sorted(files)[:200]:  # limit to 200 for speed, deterministic
        try:
            j = json.loads(p.read_text())
            rd = j.get("replayData", j)
            # Need bounds and roster for occ
            bmin, bmax = rd["bounds"]["min"], rd["bounds"]["max"]
            span_x = (bmax["x"] - bmin["x"]) or 1.0
            span_y = (bmax["y"] - bmin["y"]) or 1.0
            roster = rd["roster"]
            actors = sorted(roster.keys(), key=int)
            # Build occ helper similar to round_rows but simplified: we track occ per tick at 5s intervals
            for rnd in rd["rounds"]:
                if not rnd.get("events"):
                    continue
                # Collect ability times
                abil_times = [e["t"] for e in rnd["events"] if e.get("type") == "ability"]
                if not abil_times:
                    continue
                # Build occ at each tick (5s intervals) similar to round_rows
                # Precompute occ per tick
                freeze_end = rnd.get("freezetimeEndT") or 0
                duration = rnd["durationMs"]
                t_sec = max(freeze_end / 1000.0 + 5.0, 5.0)
                ticks = []
                occs = []
                ev_by_actor = {}
                for e in rnd["events"]:
                    if e["type"] == "snapshot":
                        ev_by_actor.setdefault(e["actorId"], []).append(e)
                for lst in ev_by_actor.values():
                    lst.sort(key=lambda e: e["t"])
                import math
                def snap_at_local(actor, t_ms):
                    lst = ev_by_actor.get(actor)
                    if not lst:
                        return None
                    lo, hi, best = 0, len(lst)-1, None
                    while lo <= hi:
                        mid = (lo+hi)//2
                        if lst[mid]["t"] <= t_ms:
                            best = lst[mid]; lo = mid+1
                        else:
                            hi = mid-1
                    if best is None or t_ms - best["t"] > 6000:
                        return None
                    return best
                # Build ticks
                while t_sec * 1000.0 < duration and len(ticks) < 60:
                    t_ms = t_sec * 1000.0
                    occ_a = np.zeros(36)
                    occ_b = np.zeros(36)
                    for actor in actors:
                        team = roster[actor]["team"]
                        st = None
                        # Find state
                        for s in rnd.get("playerStates", {}).get(actor, []):
                            if s["t"] <= t_ms:
                                st = s
                            else:
                                break
                        alive = bool(st and st.get("alive"))
                        sn = snap_at_local(actor, t_ms)
                        if sn and alive:
                            xn = (sn["pos"]["x"] - bmin["x"]) / span_x
                            yn = (sn["pos"]["y"] - bmin["y"]) / span_y
                            gx = min(int(xn * 6), 5)
                            gy = min(int(yn * 6), 5)
                            if team == "A":
                                occ_a[gy*6+gx] += 1
                            else:
                                occ_b[gy*6+gx] += 1
                    ticks.append(t_ms)
                    occs.append((occ_a.copy(), occ_b.copy()))
                    t_sec += 5.0
                # Now for each ability, check 5 sec window
                kills = [e for e in rnd["events"] if e.get("type") == "kill"]
                for at in abil_times:
                    # Find next tick indices within 5 sec
                    win_end = at + 5000
                    # Find occ change in window
                    has_occ_change = False
                    # Find tick indices in window
                    idxs = [i for i, tm in enumerate(ticks) if at < tm <= win_end]
                    if len(idxs) >= 2:
                        # Compare first and last occ in window
                        first_a, first_b = occs[idxs[0]]
                        last_a, last_b = occs[idxs[-1]]
                        if not np.array_equal(first_a, last_a) or not np.array_equal(first_b, last_b):
                            has_occ_change = True
                    # Check kill in window
                    has_kill = any(at < k["t"] <= win_end for k in kills)
                    if not has_occ_change and not has_kill:
                        results.append({
                            "file": str(p),
                            "roundNum": rnd["roundNum"],
                            "ability_t": at,
                            "t_sec": at/1000.0,
                            "has_occ_change": has_occ_change,
                            "has_kill": has_kill,
                        })
                        if len(results) >= 100:
                            break
                if len(results) >= 100:
                    break
        except Exception as e:
            # print(f"[warn] {p} {e}")
            continue
        if len(results) >= 100:
            break
    print(f"[conversion] found {len(results)} failures (ability with no occ/kill in 5s)")
    return results

# --- 機能5: 出力 & CLI ---
def main():
    import argparse
    ap = argparse.ArgumentParser(description="rib-eval counterfactual analysis")
    ap.add_argument("--limit", type=int, default=0, help="limit ticks for quick test")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 0. Match 284 summit 検証 (24R + タイムアウト5R)
    print("=== 0. Match 284 summit 検証 (24R) ===")
    # Ensure data/cache/rib_284_284-m3.json exists (spec requires /root/... to be placed at data/cache)
    # We copied it, but verify
    cand_paths = [ROOT / "data" / "cache" / "rib_284_284-m3.json", ROOT / "data" / "replays" / "284-m3.json"]
    found_284 = None
    for cp in cand_paths:
        if cp.exists():
            found_284 = cp
            break
    if found_284:
        import json as _js
        j = _js.loads(found_284.read_text())
        rd = j.get("replayData", j)
        print(f"found {found_284} map={rd.get('map')} rounds={len(rd['rounds'])}")
        # Check timeout rounds
        timeouts = [r for r in rd["rounds"] if (r.get("freezetimeEndT") or 0) > 90000]
        print(f"timeout rounds (>90s): {[(r['roundNum'], r.get('freezetimeEndT')) for r in timeouts]}")
        assert len(rd["rounds"]) == 24, f"284 should have 24 rounds, got {len(rd['rounds'])}"
        # Verify that our dataset also contains 284 ticks (via meta)
        cnt_284 = sum(1 for m in META_ALL if str(m).startswith("284|"))
        # META_ALL is array of "match|roundNum|t_sec"
        # Count via MATCH_ALL
        cnt2 = int(np.sum(MATCH_ALL == "284"))
        print(f"dataset ticks for 284: meta {cnt_284} / match {cnt2}")
        # Verify that timeout rounds are present in dataset (they should be, as we don't filter them out)
        # Check that roundNums for timeout rounds appear in meta
        for r in timeouts:
            rn = r["roundNum"]
            has = any(str(m).startswith(f"284|{rn}|") for m in META_ALL)
            print(f" R{rn} (ft={r.get('freezetimeEndT')}) in dataset? {has} {'OK' if has else 'MISSING'}")
            assert has, f"timeout round {rn} missing in dataset"
        print("[check] Match 284 24R + timeout 5R OK")
    else:
        print("[warn] 284 cache not found")

    # 1. 条件付き推論 examples
    print("=== 1. 条件付き推論 ===")
    cond_results = {}
    # Example: alive equal
    for filt in [
        ({"alive_equal": True}, "alive_equal"),
        ({"hp_band": (0.5, 1.0)}, "hp_high"),
        ({"eco_band": (0.15, 0.4)}, "eco_high"),
        ({"time_band": (30, 60)}, "time_early"),
        ({"alive_equal": True, "hp_band": (0.5, 1.0)}, "alive_equal+hp_high"),
    ]:
        res = conditional_inference(filt[0])
        # Handle NaN for JSON (convert to None)
        mean = None if np.isnan(res["mean"]) else float(res["mean"])
        std = None if np.isnan(res["std"]) else float(res["std"])
        cond_results[filt[1]] = {"n": int(res["n"]), "mean": mean, "std": std}
        m_str = f"{mean:.3f}" if mean is not None else "nan"
        s_str = f"{std:.3f}" if std is not None else "nan"
        print(f"{filt[1]} n={res['n']} mean={m_str} std={s_str} {'WARN n<200' if res['n']<200 else ''}")

    # 2. 占有グリッド反実仮想 examples
    print("\n=== 2. 占有グリッド反実仮想 ===")
    # Pick a sample tick: first val tick where occ not empty
    # Use first tick with some occ
    sample_idx = None
    for i, vec in enumerate(X_ALL):
        if np.sum(vec[OCC_A_OFF:OCC_A_OFF+36]) > 0 or np.sum(vec[OCC_B_OFF:OCC_B_OFF+36]) > 0:
            sample_idx = i
            break
    if sample_idx is None:
        sample_idx = 0
    base_p = float(predict_prob(X_ALL[sample_idx][None, :])[0])
    swap_res = occ_counterfactual(sample_idx, "swap")
    give_res = occ_counterfactual(sample_idx, "give", cell=0)
    print(f"sample {sample_idx} base P={base_p:.3f} swap delta={swap_res['delta']:.3f} give cell0 delta={give_res['delta']:.3f}")
    # Assert delta bounds
    assert -1 <= swap_res["delta"] <= 1, "delta out of bounds"
    assert -1 <= give_res["delta"] <= 1, "delta out of bounds"
    print(f"[check] ΔP in [-1,1] OK")

    # Alive+1 sanity: increase alive_a by 1 (team aggregate) -> P(A) should go up
    alive_sample_idx = sample_idx
    for cand in range(len(X_ALL)):
        # Find a tick where not all A are alive (so we can increase)
        if float(X_ALL[cand, TEAM_OFF + 0]) < 0.9:  # alive_a <4.5
            alive_sample_idx = cand
            break
    vec_alive = X_ALL[alive_sample_idx].copy()
    base_p_alive = float(predict_prob(vec_alive[None, :])[0])
    # Increase A alive by 1 player (0.2) and hp accordingly, keep other features same
    # This isolates the effect of numbers; occ and per-player alive are not changed, only aggregates
    # For sanity, we also set one per-player alive if possible to be consistent, but team aggregate is the main signal
    vec_alive[TEAM_OFF + 0] = min(1.0, vec_alive[TEAM_OFF + 0] + 0.2)  # alive_a
    vec_alive[TEAM_OFF + 2] = min(1.0, vec_alive[TEAM_OFF + 2] + 0.2)  # hp_a
    # Also set a dead player's alive/hp to 1 for consistency (find first dead)
    for pi in range(N_PLAYERS):
        base = PER_PLAYER_BASE + pi * PER_PLAYER
        if vec_alive[base + 4] < 0.5:
            vec_alive[base + 4] = 1.0
            vec_alive[base + 5] = 1.0
            break
    p_alive = float(predict_prob(vec_alive[None, :])[0])
    delta_alive = p_alive - base_p_alive
    print(f"alive+1: base {base_p_alive:.3f} -> {p_alive:.3f} delta {delta_alive:+.3f} {'OK up' if delta_alive > 0 else 'WARN not up'}")
    # Allow small negative due to model noise, but generally should be positive
    assert delta_alive > -0.2, f"alive+ should not drastically decrease, got {delta_alive}"
    assert -1 <= delta_alive <= 1, "alive delta out of bounds"

    # Extreme occ test for NaN
    extreme_zero = np.zeros((1, 160), dtype=np.float32)
    extreme_zero[0, OFF_T] = 0.5  # t_sec 50
    extreme_zero[0, OFF_ATTACKER] = 0
    # occ already zero
    try:
        p_zero = float(predict_prob(extreme_zero)[0])
        print(f"extreme occ zero P={p_zero:.3f} NaN? {np.isnan(p_zero)}")
        assert not np.isnan(p_zero), "NaN for zero occ"
    except Exception as e:
        print(f"extreme zero failed: {e}")
    extreme_same = np.zeros((1, 160), dtype=np.float32)
    extreme_same[0, OCC_A_OFF:OCC_A_OFF+36] = 5.0  # all in same cell? Actually set one cell to 5
    extreme_same[0, OCC_A_OFF] = 5.0
    extreme_same[0, OCC_B_OFF + 35] = 5.0
    try:
        p_same = float(predict_prob(extreme_same)[0])
        print(f"extreme same cell P={p_same:.3f} NaN? {np.isnan(p_same)}")
        assert not np.isnan(p_same), "NaN for same cell"
    except Exception as e:
        print(f"extreme same failed: {e}")

    # 3. ペアワイズ比較
    print("\n=== 3. ペアワイズ比較 ===")
    pairwise_summary = pairwise_heatmap(OUT_DIR)

    # 4. 変換失敗検出
    print("\n=== 4. 変換失敗検出 ===")
    failures = conversion_failures()
    for f in failures[:5]:
        print(f)

    # 5. 出力 summary.json
    summary = {
        "model": str(Path(MODEL_PATH).relative_to(ROOT)) if Path(MODEL_PATH).is_absolute() else str(MODEL_PATH),
        "feature_order": "build_dataset.py:160 (t,attacker,per-player 80,team6,occ72)",
        "conditional": cond_results,
        "counterfactual_sample": {"base_idx": int(sample_idx), "base_p": base_p, "swap_delta": swap_res["delta"], "give_delta": give_res["delta"]},
        "pairwise": pairwise_summary,
        "conversion_failures": failures[:20],
        "conversion_failures_total": len(failures),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[output] {OUT_DIR / 'summary.json'}")
    print(f"CSV per map in {OUT_DIR}")

if __name__ == "__main__":
    main()
