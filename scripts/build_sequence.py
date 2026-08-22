#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np

from build_dataset import snap_at, state_at, view_angle_sin_cos

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "replays"
OUT = ROOT / "data" / "sequences.npz"

TICK_SEC = 5.0
TMAX = 44
NPLAYERS = 10
F = 18
MAPS = ["ascent", "split", "haven", "sunset", "summit", "lotus"]
FREEZE_FALLBACK_MS = 8000
FREEZE_TIMEOUT_CUTOFF_MS = 90000

WEAPON_TIER = {
    "vandal": 0, "phantom": 0, "bulldog": 0, "guardian": 0, "bandit": 0, "odin": 0,
    "operator": 1, "marshal": 1, "outlaw": 1,
    "stinger": 2, "spectre": 2,
    "judge": 3, "bucky": 3,
    "classic": 4, "ghost": 4, "sheriff": 4, "frenzy": 4, "shorty": 4,
}
N_WEAPON_TIERS = 6


def build_round(rd, rnd, bx, by, span_x, span_y, actors, roster):
    winner = rnd["winner"]
    if winner not in ("A", "B"):
        return None
    duration = rnd["durationMs"]
    freeze_end = rnd.get("freezetimeEndT") or 0
    if freeze_end > FREEZE_TIMEOUT_CUTOFF_MS:
        return None
    kills = [
        {
            "t": e["t"],
            "killer": e["actorId"],
            "victim": e["targetId"],
            "killer_team": roster[e["actorId"]]["team"] if e["actorId"] in roster else None,
        }
        for e in rnd["events"]
        if e["type"] == "kill"
    ]
    plant_ms = next((e["t"] for e in rnd["events"] if e["type"] == "plant"), None)
    meta = {"roundNum": rnd["roundNum"], "startMs": max(freeze_end / 1000.0 + TICK_SEC, TICK_SEC) * 1000.0, "kills": kills, "plantMs": plant_ms}
    ev_by_actor = {}
    for e in rnd["events"]:
        if e["type"] == "snapshot":
            ev_by_actor.setdefault(e["actorId"], []).append(e)
    for lst in ev_by_actor.values():
        lst.sort(key=lambda e: e["t"])
    states = {a: rnd["playerStates"].get(a, []) for a in actors}

    seq = np.zeros((TMAX, NPLAYERS, F), dtype=np.float32)
    mask = np.zeros(TMAX, dtype=np.float32)
    t_sec = max(freeze_end / 1000.0 + TICK_SEC, TICK_SEC)
    k = 0
    while t_sec * 1000.0 < duration and k < TMAX:
        t_ms = t_sec * 1000.0
        is_post = 1.0 if (plant_ms is not None and t_ms >= plant_ms) else 0.0
        alive_flags = []
        for actor in actors:
            st = state_at(states[actor], t_ms)
            alive_flags.append(bool(st and st.get("alive")))
        alive_a = sum(1 for f, a in zip(alive_flags, actors) if f and roster[a]["team"] == "A")
        alive_b = sum(1 for f, a in zip(alive_flags, actors) if f and roster[a]["team"] == "B")
        adv = (alive_a - alive_b) / 5.0
        for pi, actor in enumerate(actors):
            st = state_at(states[actor], t_ms)
            alive = alive_flags[pi]
            hp = (st or {}).get("health", 0) / 100.0
            armor = (st or {}).get("armor", 0) / 50.0
            eco = (st or {}).get("loadoutValue", 0) / 20000.0
            money = (st or {}).get("money", 0) / 9000.0
            wname = (st or {}).get("weapon")
            tier = WEAPON_TIER.get(wname, 5) if wname else 5
            w_onehot = [0.0] * N_WEAPON_TIERS
            w_onehot[tier] = 1.0
            sn = snap_at(ev_by_actor, actor, t_ms)
            if sn and alive:
                xn = (sn["pos"]["x"] - bx) / span_x
                yn = (sn["pos"]["y"] - by) / span_y
                sx, sy = view_angle_sin_cos(sn["viewVector"])
            else:
                xn = yn = sx = sy = 0.0
            base = (xn, yn, sx, sy, 1.0 if alive else 0.0, hp, armor, eco, 1.0 if roster[actor]["team"] == "A" else 0.0, money, adv, is_post)
            seq[k, pi] = base + tuple(w_onehot)
        mask[k] = 1.0
        k += 1
        t_sec += TICK_SEC
    if k < 2:
        return None
    return seq, mask, (1 if winner == "A" else 0), (1 if rnd["attackerTeam"] == "A" else 0), meta


def main():
    X, M, Y, AT, maps, matches = [], [], [], [], [], []
    meta_path = ROOT / "data" / "rounds_meta.jsonl"
    meta_f = open(meta_path, "w")
    for f in sorted(CACHE.glob("*.json")):
        try:
            rd = json.loads(f.read_text())["replayData"]
        except (json.JSONDecodeError, KeyError):
            continue
        if rd["map"] not in MAPS:
            continue
        bminx, bminy = rd["bounds"]["min"]["x"], rd["bounds"]["min"]["y"]
        span_x = (rd["bounds"]["max"]["x"] - bminx) or 1.0
        span_y = (rd["bounds"]["max"]["y"] - bminy) or 1.0
        roster = rd["roster"]
        actors = sorted(roster.keys(), key=int)
        if len(actors) != NPLAYERS:
            continue
        match_id = f.stem.split("-")[0]
        n_r = 0
        for rnd in rd["rounds"]:
            out = build_round(rd, rnd, bminx, bminy, span_x, span_y, actors, roster)
            if out is None:
                continue
            seq, mask, y, at, meta = out
            X.append(seq)
            M.append(mask)
            Y.append(y)
            AT.append(at)
            maps.append(rd["map"])
            matches.append(match_id)
            meta_f.write(
                json.dumps(
                    {
                        "match": match_id,
                        "map": rd["map"],
                        "roster": roster,
                        **meta,
                    }
                )
                + "\n"
            )
            n_r += 1
        print(f"{f.stem}: {n_r} rounds", flush=True)
    meta_f.close()

    Xa = np.stack(X)
    Ma = np.stack(M)
    np.savez_compressed(
        OUT,
        X=Xa,
        mask=Ma,
        y=np.array(Y, dtype=np.int8),
        attacker=np.array(AT, dtype=np.int8),
        map=np.array(maps),
        match=np.array(matches),
    )
    print(f"sequences: {Xa.shape}, pos_rate={np.mean(Y):.3f} -> {OUT}")


if __name__ == "__main__":
    main()
