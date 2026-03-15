# Anchor 匹配优化执行步骤（Step-by-Step）

本文档基于 `anchor_match.md` 的策略整理为可执行步骤，目标是分阶段降低 `ignore`、提升 side/短线匹配质量，并避免一次改动过大导致不可定位问题。

## Step 1. 冻结基线配置

- 复制当前可运行配置为实验配置，例如：
  - `configs/tusimple_res18_fpn_matchv2.yaml`
- 后续全部在该实验配置上改，避免影响主线。

验收：
- 能用新配置正常运行 `vis_dataset.py`。

## Step 2. 先做可观测性（不改分配逻辑）

在 `tools/vis_dataset.py` 和训练前统计中增加以下输出：

1. per-GT `max_iou` 分布：`mean / p50 / p90 / min`
2. per-GT 正样本数分布：`mean / min / max`
3. `ignore` 来源计数（按失败原因拆分）：
   - common 约束失败
   - top 约束失败
   - angle 门控失败
   - x_bottom 门控失败
   - 阈值灰区

验收：
- 一次 `--max 20` 运行能输出上述统计。

## Step 3. 接入参数骨架（默认等价旧行为）

在 `label_assigner.py` + YAML 中先新增参数，不改变默认行为：

- `match.topk_per_gt`
- `match.min_force_pos_iou`
- `match.use_soft_gating`
- `match.penalty_common_ratio`
- `match.penalty_top_err`
- `match.penalty_angle`
- `match.pos_thr_bottom`
- `match.pos_thr_side`
- `match.neg_thr`

验收：
- 不开新开关时，结果与旧版本近似一致。

## Step 4. 实现 Dynamic Top-K（可开关）

在 `assign(...)` 中实现 per-GT Top-K 强制正样本逻辑：

1. 构建 IoU 矩阵 `[num_gt, num_anchors]`
2. 每条 GT 取 `top_k` 候选
3. 满足 `iou >= min_force_pos_iou` 时标为 `force_pos`
4. 初版先不开 fallback（避免过度放宽）

建议初值：
- `topk_per_gt = 3`
- `min_force_pos_iou = 0.10`

验收：
- `pos` 有可控上升，误配不明显恶化。

## Step 5. 实现 Soft Gating（可开关）

将部分“硬作废（IoU=-1）”改为“软惩罚（IoU*=penalty）”：

- common_ratio 不足：`* penalty_common_ratio`
- top_err 超阈值：`* penalty_top_err`
- angle 超阈值：`* penalty_angle`

仅保留极端硬作废：
- `common_count == 0`（或其它你明确认定的不可匹配场景）

验收：
- `ignore` 降低，`pos/neg` 分布更均衡。

## Step 6. 缩小灰区（参数小步扫描）

按小步网格调整阈值，每轮只改一组：

- `pos_thr_bottom: 0.38 -> 0.34 -> 0.30`
- `pos_thr_side: 0.27 -> 0.25 -> 0.24 -> 0.22`
- `neg_thr: 0.10 -> 0.15 -> 0.18`

每次固定命令比较：

```powershell
python tools/vis_dataset.py --cfg configs/tusimple_res18_fpn_matchv2.yaml --out outputs/visualizations/m2_stepX --show_match --max 20 --topk 8
```

验收：
- `ignore` 下降明显；
- side 正样本提升；
- 视觉误配不明显变差。

## Step 7. M2 阶段验收

M2 通过标准建议：

1. side 正样本达到目标区间（按每图统计）
2. `ignore` 显著下降（相对基线）
3. 中远端候选覆盖改善
4. 可视化误配可控

## Step 8. 小规模训练验证（3~5 epoch）

先短训验证方向，不直接全量训练：

关注：
- `cls_loss` 是否不再快速塌到接近 0
- `reg_loss` 是否稳定下降
- 推理可视化中边缘线/短线是否改善

## Step 9. 全量训练与评估

在最优参数组上跑全量训练，统一对比：

- `Accuracy / FP / FN`
- 可视化质量（起始/末端稳定性、边缘线召回）

## Step 10. 固化与复盘

将最终版本记录回文档：

1. 最终参数
2. 与基线对比结果
3. 失败参数组合与原因

---

## 执行原则（强烈建议）

1. 一次只改一个策略或一组小参数。
2. 每次改动保留实验配置快照。
3. 同时看统计指标与可视化，不单看单一指标。
4. 优先保留“可解释且稳定”的提升，而非偶然高分配置。
