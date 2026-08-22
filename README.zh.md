# rib-eval

[English](README.md) | [日本語](README.ja.md) | 中文

面向**职业VALORANT**的胜率（评估图）模型，基于 [rib.gg](https://rib.gg) 的公开2D回放数据训练。

相当于把将棋/国际象棋的评估函数搬进VALORANT：对每个回合内每5秒的时间片，模型根据盘面状态（选手坐标、朝向、存活数、护甲、经济）估算 **A队赢得该回合的概率**。输出每个回合的评估曲线，以及基于"每次击杀使胜率变动多少"的选手影响力排行。

**报告预览: https://sitne.github.io/rib-eval/** （单个自包含HTML文件 — 随时随地打开）

## 教练能用它做什么

- **VOD复盘自动高亮**: 不必重看整个回合，直接跳到胜率真正发生波动的时刻（曲线上的黄点）
- **超越K/D的选手评估**: WPA（Win Probability Added）衡量的是"击杀发生的时机"，而非数量。残局1v3的突破价值远超出生点的刷数据击杀
- **对手侦察**: 汇总对手公开比赛历史中的波动模式与WPA倾向
- **决策风险收益分析**: 量化强起 vs 保枪、回防 vs 放弃回合（类似NFL的"四档分析"）

## 结果

| 模型 | 验证集AUC | 时间片准确率 | 回合末准确率 |
|---|---|---|---|
| XGBoost（单时间片） | 0.871 | 平均~70% | — |
| Transformer（混合模型，6张地图共享） | 0.866 | 76.2% | 97.6% |
| Transformer（分地图模型） | 0.853–0.867 | 75–77% | 97–98% |

分地图的专用模型**并未超过**混合模型——在这个数据规模（约17,500个回合）下，跨地图共享知识更有效。准确率从冻结期结束后的约64%，随着信息积累在中盘升至83–88%。

## 报告阅读指南（`outputs/wpa_report.html`）

- **顶部标签**: 数据规模（回合数 / 归因击杀数 / 选手数 / 检测到的波动次数）
- **WPA表格**: 绿色 = 击杀推高胜率的选手，红色 = 造成损失的选手。`avg/K` 列最有参考价值：平均每次击杀推动多少个百分点（K/D平平但擅长关键局的选手在这里得分很高）
- **波动卡片**: 蓝色曲线 = 回合时间轴上的P(A获胜)；红色竖带 = 击杀瞬间；黄色圆点 = 5秒内±12%以上的胜率跳变；角落标签 = 最大波动幅度
- 卡片可点击跳转到 rib.gg 的比赛页面

## 流水线

```
fetch_replays.py     抓取 rib.gg 回放API（463场比赛 × m1–m3，含Vercel验证处理）
                     + fetch_match_info.py（队伍名/赛事名/日期等人类可读标签）
build_dataset.py     时间片级扁平特征 → dataset.npz          （XGBoost基线）
build_sequence.py    [44时间片 × 10选手 × 9特征] → sequences.npz + rounds_meta.jsonl
train_eval.py        XGBoost单时间片基线
train_transformer.py 空间Transformer + GRU（混合 / 分地图）
wpa_analysis.py      波动检测 + 选手WPA → 自包含HTML报告
```

## 快速开始

```bash
uv sync                                        # 安装 torch(CUDA)、xgboost、sklearn、playwright 等
uv run python scripts/fetch_replays.py         # 收集回放（全量约1–2小时）
uv run python scripts/fetch_match_info.py      # 可选：获取人类可读标签（队名、赛事名）
uv run python scripts/build_dataset.py         # XGBoost特征
uv run python scripts/train_eval.py            # 基线训练
uv run python scripts/build_sequence.py        # 序列张量
uv run python scripts/train_transformer.py --map ALL   # 分地图 + 混合模型
uv run python scripts/wpa_analysis.py          # HTML报告 -> outputs/
```

需要 Python ≥ 3.11 和 [uv](https://docs.astral.sh/uv/)。GPU可选（CPU可用，约慢15倍）。

## 已知局限

- 概率**未经校准** —— 绝对值仅作方向性参考；相对比较（波动幅度、WPA排名）更可靠
- WPA归因目前仅覆盖击杀；下包、拆包与道具价值尚未计入
- rib.gg 数据为赛后发布 —— 无法用于实时场景
- 训练数据限于当前竞技地图池（Ascent / Split / Haven / Sunset / Summit / Lotus）。Abyss几乎没有回放覆盖

## 许可证

MIT。数据由 rib.gg 提供。
