#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "replays"
OUT = ROOT / "data" / "dataset.npz"

TICK_SEC = 5.0
GRID = 6
MAX_ROUNDS_TICKS = 60

PER_PLAYER = 8


def view_angle_sin_cos(vv):
    n = math.hypot(vv["x"], vv["y"]) or 1.0
    return vv["x"] / n, vv["y"] / n


def state_at(states, t_ms):
    cur = None
    for s in states:
        if s["t"] <= t_ms:
            cur = s
        else:
            break
    return cur


def snap_at(events_by_actor, actor, t_ms, max_age=6000):
    snaps = events_by_actor.get(actor)
    if not snaps:
        return None
    lo, hi, best = 0, len(snaps) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if snaps[mid]["t"] <= t_ms:
            best = snaps[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None or t_ms - best["t"] > max_age:
        return None
    return best


def round_rows(replay, match_id):
    rd = replay["replayData"]
    bmin, bmax = rd["bounds"]["min"], rd["bounds"]["max"]
    span_x = (bmax["x"] - bmin["x"]) or 1.0
    span_y = (bmax["y"] - bmin["y"]) or 1.0
    roster = rd["roster"]
    actors = sorted(roster.keys(), key=int)
    rows, labels, meta = [], [], []
    if len(actors) != 10:
        return rows, labels, meta

    for rnd in rd["rounds"]:
        if not rnd.get("playerStates") or not rnd.get("events"):
            continue
        winner = rnd["winner"]
        attacker = rnd["attackerTeam"]
        freeze_end = rnd.get("freezetimeEndT") or 0
        duration = rnd["durationMs"]
        ev_by_actor = {}
        for e in rnd["events"]:
            if e["type"] == "snapshot":
                ev_by_actor.setdefault(e["actorId"], []).append(e)
        for lst in ev_by_actor.values():
            lst.sort(key=lambda e: e["t"])

        t_sec = max(freeze_end / 1000.0 + TICK_SEC, TICK_SEC)
        n_ticks = 0
        while t_sec * 1000.0 < duration and n_ticks < MAX_ROUNDS_TICKS:
            t_ms = t_sec * 1000.0
            feat = [t_sec / 100.0, 1.0 if attacker == "A" else 0.0]
            occ_a = np.zeros(GRID * GRID)
            occ_b = np.zeros(GRID * GRID)
            alive_a = alive_b = hp_a = hp_b = eco_a = eco_b = 0
            for actor in actors:
                team = roster[actor]["team"]
                st = state_at(rnd["playerStates"].get(actor, []), t_ms)
                alive = bool(st and st.get("alive"))
                hp = (st or {}).get("health", 0) / 100.0
                armor = (st or {}).get("armor", 0) / 50.0
                eco = (st or {}).get("loadoutValue", 0) / 20000.0
                sn = snap_at(ev_by_actor, actor, t_ms)
                if sn and alive:
                    xn = (sn["pos"]["x"] - bmin["x"]) / span_x
                    yn = (sn["pos"]["y"] - bmin["y"]) / span_y
                    sx, sy = view_angle_sin_cos(sn["viewVector"])
                    gx = min(int(xn * GRID), GRID - 1)
                    gy = min(int(yn * GRID), GRID - 1)
                    (occ_a if team == "A" else occ_b)[gy * GRID + gx] += 1.0
                else:
                    xn = yn = sx = sy = 0.0
                feat += [xn, yn, sx, sy, 1.0 if alive else 0.0, hp, armor, eco]
                if team == "A":
                    alive_a += alive
                    hp_a += hp
                    eco_a += eco
                else:
                    alive_b += alive
                    hp_b += hp
                    eco_b += eco
            feat += [
                alive_a / 5.0,
                alive_b / 5.0,
                hp_a / 5.0,
                hp_b / 5.0,
                eco_a / 5.0,
                eco_b / 5.0,
            ]
            feat += occ_a.tolist() + occ_b.tolist()
            rows.append(feat)
            labels.append(1 if winner == "A" else 0)
            meta.append(f"{match_id}|{rnd['roundNum']}|{t_sec:.0f}")
            n_ticks += 1
            t_sec += TICK_SEC
    return rows, labels, meta


def main():
    X, y, meta, match_ids = [], [], [], []
    files = sorted(CACHE.glob("*.json"))
    for f in files:
        match_id = f.stem.split("-")[0]
        try:
            replay = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"[skip] {f.name}: invalid json")
            continue
        r, l, m = round_rows(replay, match_id)
        X += r
        y += l
        meta += m
        match_ids += [match_id] * len(r)
        print(f"{f.stem}: {len(r)} tick samples")
    Xa = np.array(X, dtype=np.float32)
    ya = np.array(y, dtype=np.int8)
    np.savez_compressed(OUT, X=Xa, y=ya, meta=np.array(meta), match=np.array(match_ids))
    print(f"dataset: {Xa.shape}, pos_rate={ya.mean():.3f} -> {OUT}")


if __name__ == "__main__":
    main()
