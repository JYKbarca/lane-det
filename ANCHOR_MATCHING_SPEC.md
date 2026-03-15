# Anchor 匹配逻辑与判定标准（当前版本）

本文档描述当前代码中 Anchor 生成、匹配、标签分配、训练使用方式的真实实现（以当前仓库代码为准）。

## 1. 流程总览

1. 数据集读取样本（`lane_det/datasets/tusimple.py`）。
2. 根据当前图像尺寸和 `h_samples` 生成 Anchor（`lane_det/anchors/anchor_generator.py`）。
3. 使用 `LabelAssigner.assign(...)` 做匹配与标签分配（`lane_det/anchors/label_assigner.py`）。
4. 训练时使用：
   - `cls_label` 计算分类损失（Focal Loss）
   - `offset_label` + `offset_valid_mask` 计算回归损失（Smooth L1）
   - 实现见 `tools/train.py`

## 2. Anchor 生成规则

当前配置文件：`configs/tusimple_res18_fpn.yaml`

- 图像尺寸：`1280 x 720`
- `num_y`: `56`

### 2.1 Bottom Anchors

- 由 `x_positions × angles` 组合生成
- 当前配置：
  - `x_positions: start=0, end=1280, step=10`
  - `angles: [-75, -70, ..., 70, 75]`
- 数学形式（线性）：
  - `x(y) = x_bottom + tan(angle) * (y_bottom - y)`

### 2.2 Side Anchors

- 从左右边界出发，按多个 `y_start` 和角度生成
- 当前配置：
  - `side_y_step: 20`
  - `side_y_start_ratio: 0.5`
  - `side_angle_min: 10.0`
- 左侧使用正角，右侧使用负角

### 2.3 Anchor 有效掩码

- `anchor_valid_mask[a, y] = 1` 当且仅当该 Anchor 点位于图像横向范围 `[0, W-1]`

## 3. 匹配评分（Line IoU）与前置筛选

文件：`lane_det/anchors/label_assigner.py`

## 3.1 Line IoU

对共享有效点（anchor 和 GT 在该 y 都有效）计算：

- 每个 y 点把线表示为横向线段 `[x-w, x+w]`
- `inter = max(0, 2w - |dx|)`
- `union = 2w + |dx|`
- `IoU = sum(inter) / sum(union)`

其中 `w` 使用分支参数：

- bottom: `line_iou_width_bottom = 25.0`
- side: `line_iou_width_side = 35.0`

## 3.2 共享点与比例约束

若不满足以下条件，当前 anchor-GT 对直接作废（IoU 置为 `-1`）：

- `common_count >= min_common_points_*`
  - bottom: `3`
  - side: `2`
- `common_ratio >= min_common_ratio_*`
  - bottom: `0.15`
  - side: `0.06`

## 3.3 Top 区域一致性约束

- Top 区域定义：`y <= y_min + top_region_ratio * (y_max - y_min)`
- 当前参数：
  - `top_region_ratio: 0.4`
  - `top_min_points_bottom: 2`
  - `top_min_points_side: 0`
  - `top_max_mean_err_bottom: 18.0`
  - `top_max_mean_err_side: 24.0`
- 若 Top 区平均误差超过阈值，anchor-GT 对作废（IoU 置 `-1`）

## 3.4 几何门控

### 角度门控（所有 anchor）

- 估计 GT 方向后，要求：
  - `|anchor_angle - gt_angle| <= geo_angle_thr`
- 当前：`geo_angle_thr = 60.0`

### 底部交点门控（仅 bottom anchor）

- side anchor 不应用该门控
- 对 bottom anchor 要求：
  - `|anchor_x_bottom - gt_x_bottom| <= geo_x_bottom_thr`
- 当前：`geo_x_bottom_thr = 200.0`

## 4. 标签判定标准（pos/neg/ignore）

每个 anchor 先在所有 GT 中取最佳匹配 `best_iou`。

- `pos`：
  - bottom: `best_iou >= line_iou_pos_thr_bottom (0.38)`
  - side: `best_iou >= line_iou_pos_thr_side (0.27)`
- `neg`：
  - `0 <= best_iou <= line_iou_neg_thr (0.10)`
- `ignore`：
  - 其余全部（包括 `best_iou = -1` 或处于正负阈值中间灰区）

### 4.1 为什么 ignore 常偏高

当前实现中：

- 大量候选会在“前置筛选”阶段被置 `best_iou=-1`
- 且 `neg` 区间仅 `<= 0.10`，灰区较宽
- 因此 ignore 高是该判定策略的结构性结果，不是程序错误

## 5. 训练如何使用这些标签

文件：`tools/train.py`

- 分类：
  - 使用 `cls_label` 计算 Focal Loss
  - `cls_label = -1` 的样本会被忽略（不参与分类损失）
- 回归：
  - 使用 `offset_label` 与 `offset_valid_mask`
  - 仅 `offset_valid_mask=1` 的点参与回归损失

这意味着：

- 一个 anchor 即使是正样本，也不是“整条线每个点都参与回归”
- 仅与匹配 GT 的共享有效点会用于回归监督

## 6. 当前行为总结（面向排查）

1. 当前匹配是 Line IoU，不是纯 L1 距离。
2. side/bottom 使用分开阈值与分开约束。
3. side anchor 已关闭 `x_bottom` 门控，仅保留角度门控。
4. ignore 高主要来自前置筛选 + 宽灰区，不代表代码逻辑异常。
5. 若要提升可学正样本，优先检查前置筛选阈值，而不是只降 `pos_thr`。

---

如后续修改了 `configs/tusimple_res18_fpn.yaml` 或 `label_assigner.py`，本文档需要同步更新。
