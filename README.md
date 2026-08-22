# rib-eval

Win-probability ("eval graph") models for **professional VALORANT**, trained on public 2D replay data from [rib.gg](https://rib.gg).

Think of the eval bar in chess, but for VALORANT rounds: for every 5-second tick of every round, a model estimates **P(team A wins)** from the board state — player positions, view directions, alive counts, HP/armor, and economy. The result is a shogi-style evaluation graph per round, plus player impact rankings based on how much each kill moved the win probability.

**Live report: https://sitne.github.io/rib-eval/** (self-contained single HTML file — open anywhere)

## What coaches can use this for

- **Auto-highlighted VOD review**: instead of rewatching full rounds, jump straight to the ticks where win probability actually swung (yellow dots on the curves).
- **Player impact beyond K/D**: WPA (win probability added) measures *when* kills happened, not just how many. A 1v3 clutch entry is worth far more than a spawn-peek exit frag.
- **Opponent scouting**: aggregate swings and WPA patterns across an opponent's public match history.
- **Decision risk/reward analysis**: quantify force-buys vs. saves, retakes vs. round concessions (NFL "4th down analytics" style).

## Results

| model | holdout AUC | tick accuracy | last-tick accuracy |
|---|---|---|---|
| XGBoost (single tick) | 0.871 | ~70% avg | — |
| Transformer (pooled, 6 maps) | 0.866 | 76.2% | 97.6% |
| Transformer (per map) | 0.853–0.867 | 75–77% | 97–98% |

Per-map specialist models did **not** beat the pooled model — shared knowledge across maps wins at this data scale (~17.5k rounds). Accuracy climbs from ~64% right after the freeze period to 83–88% mid-round as information accumulates.

## How to read the report (`outputs/wpa_report.html`)

- **Chips**: dataset size (rounds / kills attributed / players / detected swings)
- **WPA tables**: green = players whose kills added win probability, red = cost it. `avg/K` is the most interesting column: win-probability points moved per kill (high-leverage clutch players score high here even with modest K/D)
- **Swing cards**: blue curve = P(A wins) over round time; red bands = kill moments; yellow dots = ±12% probability jumps in 5 seconds; chip in the corner = biggest swing magnitude
- Cards link to the match page on rib.gg

## Pipeline

```
fetch_replays.py     scrape rib.gg replay API (463 matches × m1–m3, Vercel-checkpoint aware)
                     + fetch_match_info.py (team/event names + dates for human-readable labels)
build_dataset.py     flat per-tick features → dataset.npz          (XGBoost baseline)
build_sequence.py    [44 ticks × 10 players × 9 features] → sequences.npz + rounds_meta.jsonl
train_eval.py        XGBoost single-tick baseline
train_transformer.py spatial transformer + GRU, pooled or per-map
wpa_analysis.py      swing detection + per-player WPA → self-contained HTML report
```

## Quickstart

```bash
uv sync                                        # installs torch (CUDA), xgboost, sklearn, playwright deps
uv run python scripts/fetch_replays.py         # collect replays (~1–2h for the full archive)
uv run python scripts/fetch_match_info.py      # optional: human-readable team/event labels
uv run python scripts/build_dataset.py         # XGBoost features
uv run python scripts/train_eval.py            # baseline training
uv run python scripts/build_sequence.py        # sequence tensors
uv run python scripts/train_transformer.py --map ALL   # per-map + pooled models
uv run python scripts/wpa_analysis.py          # HTML report -> outputs/
```

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). GPU optional (CPU works, ~15× slower).

## Caveats

- Probabilities are **uncalibrated** — treat absolute percentages as directional; relative comparisons (swings, WPA rankings) are more trustworthy.
- WPA attribution covers kills only. Plants, defuses and utility value are not yet credited.
- rib.gg data is post-match only — no live/in-game use.
- Trained on the current competitive map pool (Ascent, Split, Haven, Sunset, Summit, Lotus). Abyss has almost no replay coverage.

## License

MIT. Data courtesy of rib.gg.
