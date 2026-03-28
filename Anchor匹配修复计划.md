# Anchor 匹配修复计划

## 1. 背景

当前 Anchor-based 路线的主要风险不在 Backbone 或 Head，而在 `lane_det/anchors/label_assigner.py` 的匹配语义已经与配置和文档脱节，直接影响训练监督质量。已确认的核心问题如下：

1. 当 `match.topk_per_gt = 0` 时，当前实现不会产生任何正样本。
2. `neg_thr` 与 `line_iou_pos_thr / line_iou_pos_thr_bottom / line_iou_pos_thr_side` 在当前实现中没有真正参与最终的正负样本判定。
3. fallback 逻辑会绕过角度门控和 `x_bottom` 门控，把原本应判无效的 anchor 强行翻成正样本。

这些问题会导致训练出现以下后果：

- 旧主配置可能出现“有 GT 但无正样本”的硬错误。
- 大量高 IoU 但非 top-k 的 anchor 被错误标成负样本，污染分类监督。
- 几何门控失去约束力，错误方向或错误交点的 anchor 被反向强化。

## 2. 修改目标

本轮修改只修复 Anchor 匹配语义，不做网络结构重构。

目标如下：

1. 恢复基于阈值的 `pos / ignore / neg` 主判定逻辑。
2. 保留 `topk_per_gt`，但仅作为“补充正样本”策略，而不是唯一正样本来源。
3. 保证 fallback 不再绕过硬门控。
4. 让配置字段与代码语义重新一致。
5. 为关键分支补最小单元测试，避免后续回归。

## 3. 改动范围

主要涉及以下文件：

- `lane_det/anchors/label_assigner.py`
- `configs/tusimple_res18_fpn.yaml`
- `configs/tusimple_res18_fpn_matchv2.yaml`
- `tests/` 下新增或补充针对 `LabelAssigner` 的测试文件
- 如有必要，补充一份说明文档，更新当前匹配规则

## 4. 实施方案

### 4.1 恢复阈值驱动的主匹配逻辑

在 `lane_det/anchors/label_assigner.py` 中恢复以下语义：

1. 先完成共享点约束、Top 区约束、角度约束、`x_bottom` 约束，得到最终有效的 `best_iou`。
2. 对每个 anchor 按阈值分配：
   - `best_iou >= pos_thr` -> 正样本
   - `best_iou < neg_thr` 或无有效匹配 -> 负样本
   - `neg_thr <= best_iou < pos_thr` -> ignore
3. `pos_thr` 需要区分 bottom anchor 与 side anchor，重新接回：
   - `line_iou_pos_thr_bottom`
   - `line_iou_pos_thr_side`
   - 若未单独配置，则回退到 `line_iou_pos_thr`

这样可以让旧配置在不依赖 `topk_per_gt` 的情况下正常工作。

### 4.2 调整 `topk_per_gt` 的职责

`topk_per_gt` 不再决定“谁是唯一正样本”，而改为：

1. 在阈值主逻辑跑完之后，检查每条 GT 是否缺少正样本。
2. 若缺少，则从“已经通过硬门控”的有效候选中，额外补充若干个高 IoU anchor 为正样本。
3. 补充时仍要求 `iou >= min_force_pos_iou`。

目标是把 `topk_per_gt` 从“主判定逻辑”降级为“召回兜底策略”。

### 4.3 修复 fallback 绕过门控的问题

当前 fallback 会重新计算 raw IoU，并忽略之前的硬门控结果。修复原则如下：

1. fallback 只能在“通过硬门控”的候选中选择。
2. 不允许重新引入已经被角度门控或 `x_bottom` 门控淘汰的 anchor。
3. 若某条 GT 在硬门控后确实没有合法候选，则保持“该 GT 无正样本”，并通过统计信息暴露，而不是强行制造伪正样本。

这一步的目标是恢复几何门控的真实性。

### 4.4 统一配置字段与代码语义

需要检查并对齐以下配置项：

- `line_iou_pos_thr`
- `line_iou_pos_thr_bottom`
- `line_iou_pos_thr_side`
- `line_iou_neg_thr`
- `topk_per_gt`
- `min_force_pos_iou`

具体要求：

1. `tusimple_res18_fpn.yaml` 在不写 `match.topk_per_gt` 时，也能正常产出正样本。
2. `tusimple_res18_fpn_matchv2.yaml` 的 `topk_per_gt` 保留，但语义改为“补正样本”。
3. 如旧字段不再需要，必须同步清理文档；如继续保留，则必须落实到代码。

## 5. 测试计划

需要补最小化单元测试，至少覆盖以下场景：

1. `topk_per_gt = 0` 时，只要 `best_iou >= pos_thr`，就必须产生正样本。
2. 高 IoU 但未进入 top-k 的 anchor，不能被错误打成负样本；应按阈值落到正样本或 ignore。
3. 被角度门控淘汰的 anchor，不能被 fallback 再翻成正样本。
4. side anchor 和 bottom anchor 使用不同 `pos_thr` 时，分配结果符合预期。
5. 无 GT、空 GT、全无效 GT 等边界情况保持现有稳定行为。

建议新增独立测试文件，例如：

- `tests/test_label_assigner.py`

## 6. 验收标准

完成后应满足以下验收条件：

1. `tusimple_res18_fpn.yaml` 下不再出现“全样本无正 anchor”的情况。
2. `matchv2` 配置下，正负样本分布比当前版本更符合阈值逻辑，不再出现大量高 IoU 负样本。
3. 几何门控失败的 anchor 不会被 fallback 翻正。
4. 新增单元测试全部通过。
5. 训练前匹配统计日志能清楚反映：
   - 每 GT 正样本数
   - ignore 来源
   - 无合法候选的 GT 数量

## 7. 实施顺序

建议按以下顺序执行：

1. 修改 `label_assigner.py`，先恢复阈值主逻辑。
2. 重写 `topk_per_gt` 与 fallback 逻辑，保证不绕过硬门控。
3. 补 `LabelAssigner` 单元测试，锁住 3 个核心问题。
4. 校准两个主配置文件的语义。
5. 运行最小训练前统计检查，确认正负样本分布恢复正常。

## 8. 风险与注意事项

1. 恢复阈值逻辑后，正样本数量可能明显变化，分类损失和回归损失的相对量级会波动。
2. 若当前 `matchv2` 依赖 top-k 强控正样本数量，修复后可能需要重新微调 `cls_weight`、`reg_weight`、`rank_weight`。
3. 若门控过严，修复 fallback 后可能暴露出“某些 GT 完全无合法候选”的真实问题，这属于数据覆盖或 anchor 覆盖问题，不应再用伪正样本掩盖。

## 9. 预期结果

修复完成后，Anchor-based 路线的匹配部分应满足：

- 配置可解释
- 代码语义一致
- 几何门控有效
- 正负样本分布可信
- 训练监督不再被伪负样本和伪正样本污染
