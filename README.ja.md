# rib-eval

[English](README.md) | 日本語 | [中文](README.zh.md)

**プロVALORANT**向けの勝率（評価値グラフ）モデル。[rib.gg](https://rib.gg)の公開2Dリプレイデータから学習。

将棋の評価値グラフをVALORANTに持ち込んだものです：ラウンド内の5秒ごとのティックで、盤面状態（プレイヤー座標・視線方向・生存数・HP/アーマー・経済）から **P(Aチームがラウンドに勝つ)** を推定します。ラウンド毎の評価値グラフと、各キルが勝率をどれだけ動かしたかによる選手影響度ランキングを出力します。

**レポートはこちら: https://sitne.github.io/rib-eval/** （自己完結する単一HTMLファイル — どこでも開けます）

## コーチのユースケース

- **VODレビューの自動ハイライト**: ラウンド全体を見返す代わりに、勝率が実際に動いたティック（曲線上の黄色い点）だけを確認
- **K/Dを超える選手評価**: WPA（Win Probability Added）は「キル数」ではなく「いつキルしたか」を測ります。1v3クラッチのエントリーはスポーンピークの額面キルより遥かに価値が高い
- **相手スカウティング**: 相手チームの公式戦履歴全体からスイングパターンとWPA傾向を集計
- **コールのリスクリワード分析**: フォースバイ vs セーブ、リテイク vs 諦めを定量化（NFLの「4thダウン分析」方式）

## 結果

| モデル | 検証AUC | ティック精度 | ラウンド末精度 |
|---|---|---|---|
| XGBoost（単一ティック） | 0.871 | 平均~70% | — |
| Transformer（プール型、6マップ共有） | 0.866 | 76.2% | 97.6% |
| Transformer（マップ別） | 0.853–0.867 | 75–77% | 97–98% |

マップ別専門モデルはプール型を**上回りませんでした** — このデータ規模（約17,500ラウンド）ではマップ横断の知識共有が勝ちます。精度はフリーズ明け直後の約64%から、情報が蓄積するラウンド中盤には83–88%まで上昇します。

## レポートの見方（`outputs/wpa_report.html`）

- **上部チップ**: データセット規模（ラウンド数 / 帰属キル数 / 選手数 / 検出スイング数）
- **WPAテーブル**: 緑 = キルで勝率を押し上げた選手、赤 = 損なった選手。`avg/K` 列が最重要：1キルあたり何%pt動かしたか（K/Dが控えめでも高レバレッジなクラッチプレイヤーはここで高得点）
- **スイングカード**: 青い曲線 = ラウンド時間軸でのP(A勝利)。赤帯 = キル瞬間。黄色い点 = 5秒で±12%以上の勝率変動。隅のチップ = 最大スイング幅
- カードは rib.gg の試合ページにリンク

## パイプライン

```
fetch_replays.py     rib.ggリプレイAPIのスクレイピング（463試合 × m1–m3、Vercelチェックポイント対応）
                     + fetch_match_info.py（チーム名/大会名/日付の人間用ラベル）
build_dataset.py     ティック単位の平坦特徴量 → dataset.npz          （XGBoostベースライン）
build_sequence.py    [44ティック × 10選手 × 9特徴量] → sequences.npz + rounds_meta.jsonl
train_eval.py        XGBoost単一ティックベースライン
train_transformer.py 空間Transformer + GRU（プール型 / マップ別）
wpa_analysis.py      スイング検出 + 選手別WPA → 自己完結HTMLレポート
```

## Quickstart

```bash
uv sync                                        # torch(CUDA), xgboost, sklearn, playwright等をインストール
uv run python scripts/fetch_replays.py         # リプレイ収集（全アーカイブで約1〜2時間）
uv run python scripts/fetch_match_info.py      # 任意：人間用ラベル（チーム名・大会名）の取得
uv run python scripts/build_dataset.py         # XGBoost用特徴量
uv run python scripts/train_eval.py            # ベースライン学習
uv run python scripts/build_sequence.py        # 系列テンソル
uv run python scripts/train_transformer.py --map ALL   # マップ別 + プール型モデル
uv run python scripts/wpa_analysis.py          # HTMLレポート -> outputs/
```

Python ≥ 3.11 と [uv](https://docs.astral.sh/uv/) が必要です。GPUは任意（CPUでも動作、約15倍遅い）。

## 既知の限界

- 確率は**未較正**です — 絶対値は方向的な目安として扱い、信頼できるのは相対比較（スイング、WPAランキング）
- WPA帰属はキルのみ。プラント・デフューズ・ユーティリティ価値は未計上
- rib.ggデータは試合後公開のみ — ライブ用途には使えません
- 学習は現行競技プールのマップ（Ascent / Split / Haven / Sunset / Summit / Lotus）。Abyssはリプレイカバレッジがほぼ皆無

## ライセンス

MIT。データは rib.gg 提供。
