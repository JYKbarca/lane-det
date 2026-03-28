# Anchor匹配修复计划

## 1. 背景

当前 Anchor-based 路线的主要风险不在 Backbone 或 Head，而在 `lane_det/anchors/label_assigner.py` 的匹配语义与配置、训练目标和日志解释不完全一致，直接影响训练监督质量。

目前已经确认的核心问题有四类：

1. 历史版本中，当 `match.topk_per_gt = 0` 时，可能出现“有 GT 但无正样本”的硬问题。
2. 历史版本中，`line_iou_pos_thr*` 与 `line_iou_neg_thr` 没有真正参与最终的 `pos / ignore / neg` 判定。
3. 历史版本中，fallback 曾绕过硬门控，重新用 raw IoU 补正样本。
4. 当前训练配置里，`rank loss` 已经和 Step 1 修复后的标签语义发生冲突，导致整轮训练中几乎恒为 0。

这些问题会带来以下后果：

- 正负样本语义不稳定，分类监督会被污染。
- 几何门控无法真正约束错误 anchor。
- 训练日志中的部分统计值不再完全可信。
- 后续调参会失去明确解释性。

## 2. 修改目标

本轮修改只修复 Anchor 匹配与相关训练语义，不做网络结构重构。

目标如下：

1. 恢复并稳定基于阈值的 `pos / ignore / neg` 主判定逻辑。
2. 将 `topk_per_gt` 保留为补充正样本策略，而不是唯一正样本来源。
3. 保证 fallback 不再绕过角度门控和 `x_bottom` 门控。
4. 让匹配统计和训练损失语义与当前标签定义保持一致。
5. 为关键分支补最小单元测试，避免回归。

## 3. 当前进展

### 3.1 Step 1 已完成

已完成内容：

1. 在 `lane_det/anchors/label_assigner.py` 中接回 `line_iou_pos_thr / line_iou_pos_thr_bottom / line_iou_pos_thr_side` 的配置解析。
2. 已恢复阈值驱动的主匹配逻辑：
   - `best_iou >= pos_thr` -> 正样本
   - `best_iou < neg_thr` -> 负样本
   - 中间区间 -> ignore
3. `tusimple_res18_fpn_matchv2.yaml` 已显式补齐正样本阈值字段，避免依赖默认值。
4. 已做最小脚本验证，确认：
   - `topk_per_gt = 0` 时可以正常产出正样本
   - 中间 IoU 区间可以回到 ignore，不再被一律压成负样本

### 3.2 Step 1 后的训练结果

基于 `outputs/checkpoints/serve/train10.log`，本轮训练表现为：

- 训练过程稳定收敛，无明显异常。
- 最优 epoch 出现在第 7 轮，而不是最后一轮。
- 最优验证结果：
  - `Accuracy = 0.879842`
  - `FP = 0.414365`
  - `FN = 0.249586`
- 最后一轮验证结果：
  - `Accuracy = 0.875119`
  - `FP = 0.417680`
  - `FN = 0.256169`

当前结论：

- Step 1 修复后，训练链路已经能稳定工作。
- 但从第 7 轮开始就进入平台期，后续 loss 继续下降，验证指标没有继续提升。
- 说明当前瓶颈已经不主要在回归拟合，而在匹配质量和分类监督语义。

### 3.3 Step 2 已完成

已完成内容：

1. 已重写 `label_assigner.py` 中的 fallback，不再重新计算绕过门控的 raw IoU。
2. fallback 现在只能从当前 `iou_mat` 中仍为合法候选的 anchor 里选择，也就是只能从已经通过现有门控链路的候选中补正样本。
3. fallback 补正样本时额外要求 `iou >= min_force_pos_iou`，避免出现“被 force 为正，但 `cls_target` 仍映射为 0”的语义冲突。
4. fallback 判断某条 GT 是否已经拥有正样本时，已同时考虑：
   - 阈值主逻辑得到的正样本
   - `force_pos` 但不被阈值正样本覆盖的补充正样本

已做最小验证：

1. 错向 anchor 在开启 `topk_per_gt` 的情况下，不会再被 fallback 翻成正样本。
2. 当某条 GT 丢掉 top-k 共享候选后，fallback 仍可以从未被占用且通过门控的合法候选中补出正样本。

## 4. 当前剩余问题

### 4.1 rank loss 当前实际上失效

当前训练配置：

- `line_iou_neg_thr = 0.10`
- `hard_neg_iou_min = 0.20`

而 `compute_local_rank_loss()` 选择 hard negative 的条件是：

- 样本被标为负样本
- 且 `best_iou >= hard_neg_iou_min`

但 Step 1 之后，负样本的定义已经变成：

- `best_iou < neg_thr`
- 即 `best_iou < 0.10`

因此当前 hard negative 条件与负样本定义直接冲突，导致：

- rank loss 整轮训练中几乎恒为 0
- 当前 `rank_weight = 1.0` 实际没有提供训练信号

这不是调参问题，而是语义冲突问题。

### 4.2 匹配统计仍有失真风险

当前 `per_gt_pos_count` 的统计仍基于：

- `np.argmax(iou_mat[a])`

而不是：

- `matched_gt_idx[a]`

对于 fallback 或 force positive 产生的正样本，这两者可能不一致。

后果是：

- 日志中的 `per-GT positive-count min=0` 可以看趋势
- 但不应直接视为严格准确的“真实 GT 正样本计数”

## 5. 剩余实施方案

### 5.1 Step 2 已完成，下一步转向 rank loss

Step 2 已经完成，当前不再建议在 fallback 语义上继续追加新改动，除非下一轮训练日志再次暴露新的匹配异常。

### 5.2 Step 2.5：处理 rank loss 语义冲突

在开始下一轮完整训练前，必须处理 rank loss，否则会继续带着“配置启用、训练失效”的损失项做实验。

当前状态：

1. 已做过一轮 `Step 2 + rank_weight=0` 的短对照训练。
2. 截至第 7 个 epoch，结果为：
   - `Accuracy = 0.874367`
   - `FP = 0.417127`
   - `FN = 0.254098`
3. 该结果低于上一轮最佳基线：
   - `Accuracy = 0.879842`
   - `FP = 0.414365`
   - `FN = 0.249586`
4. 说明仅修 fallback 语义而不恢复 rank mining，暂时没有带来指标提升。

已完成内容：

1. 已将 `compute_local_rank_loss()` 的 hard negative 选择逻辑改为：
   - 从与正样本属于同一 `best_gt`
   - 且 `best_iou >= hard_neg_iou_min`
   - 且当前为 ignore 的“灰区样本”中选择
2. 不再依赖“严格负样本 + 高 IoU”这一互相冲突的旧条件。

当前结论：

- `rank_weight=0` 只适合做隔离实验，不应作为最终训练配置长期保留。
- 后续正式训练应恢复启用 rank loss，并观察日志中 `Rank` 是否出现稳定非零信号。

### 5.3 Step 3：补测试并修正统计

已完成内容：

1. `per_gt_pos_count` 已改为基于 `matched_gt_idx` 统计，而不是 `argmax(iou_mat)`。
2. 已新增 `tests/test_label_assigner.py`，覆盖以下最小场景：
   - `topk_per_gt = 0` 时仍能产出阈值正样本
   - 被角度门控淘汰的 anchor 不会被 fallback 再翻正
   - `per_gt_pos_count` 按真实 `matched_gt_idx` 计数

## 6. 配置与训练建议

### 6.1 当前训练结果的正确使用方式

当前实验应优先使用：

- `best.pth`

而不是：

- `last.pth`

因为本轮最优结果在第 7 轮，后续已进入平台期。

### 6.2 下一轮实验不建议直接继续拉长 epoch

原因：

1. 当前主要问题不是模型还没学够，而是标签质量和损失语义仍不干净。
2. 继续拉长 epoch，大概率只会继续降低训练 loss，但不会稳定提升验证 Accuracy。

### 6.3 推荐下一轮实验顺序

建议按以下顺序执行：

1. 临时关闭 rank loss，或同步修 rank mining。
2. 修正匹配统计逻辑。
3. 再跑一轮较短训练作为对照：
   - 建议先跑 `8~10` 个 epoch
   - 继续以 `best checkpoint` 为比较对象

## 7. 验收标准

完成后应满足以下条件：

1. `tusimple_res18_fpn.yaml` 和 `tusimple_res18_fpn_matchv2.yaml` 下都能稳定产出正样本。
2. 高 IoU 但未进 top-k 的 anchor，不会被错误打成负样本。
3. 几何门控失败的 anchor 不会被 fallback 翻正。
4. rank loss 若启用，必须在训练日志中出现非零有效信号；若暂不修，则应明确关闭。
5. 匹配统计日志能真实反映：
   - 每 GT 正样本数
   - ignore 来源
   - 无合法候选 GT 数量

## 8. 当前最优先下一步

如果只做一个动作，优先级如下：

1. 用当前代码重新开启一轮短训练，并恢复服务器配置中的 `rank_weight`。
2. 重点观察训练日志中的 `Rank` 是否从 `0.0000` 变为稳定非零。
3. 若 `Rank` 仍为 0，再回头检查 `hard_neg_iou_min` 是否仍高于实际灰区 IoU 分布。

原因：

- fallback 语义、rank mining 语义和匹配统计口径现在已经基本对齐。
- 当前最需要的是通过一轮短训练确认 `rank loss` 是否真正恢复为有效训练信号。

## 9. 预期结果

修复完成后，Anchor-based 路线的匹配部分应满足：

- 配置可解释
- 代码语义一致
- 几何门控有效
- 匹配统计可信
- 分类与回归监督不再被伪正样本、伪负样本和失效损失项共同污染
