# CULane 接入执行计划

## 目标

在现有 TuSimple Anchor-based 车道线检测工程基础上，最小改动接入 CULane，跑通以下完整链路：

```text
CULane 数据 -> Dataset -> Anchor 匹配 -> 训练 -> 推理 -> 导出 -> 评估
```

最终目标：

- 能训练 CULane 基线模型
- 能推理并可视化结果
- 能输出 CULane 评估指标
- 不破坏现有 TuSimple 流程

## 全局约束

1. 不修改主模型结构：`ResNet/FPN/Head` 保持不变。
2. 不重写当前 Anchor 体系，优先复用现有表示与匹配逻辑。
3. 不破坏现有 TuSimple 训练、推理、评估链路。
4. 所有新增逻辑通过 `dataset.name` 分支控制。
5. 优先保证“能跑通”，再做细化优化。

## 当前工程主链

```text
TuSimpleDataset
-> 统一 lane 表示（x + valid_mask）
-> Anchor 生成与匹配
-> 模型训练
-> LaneDecoder
-> TuSimpleConverter
-> Evaluate
```

CULane 接入的原则是：不改模型主体，只扩展数据集层、导出层和评估层。

## 阶段 1：实现 CULaneDataset

### 目标

新增一个 `CULaneDataset`，并让它尽量与 `TuSimpleDataset` 的输出接口保持一致，减少对训练和推理脚本的改动。

### 新增文件

```text
lane_det/datasets/culane.py
```

### 必须对齐的输出字段

`__getitem__()` 应尽量返回与当前 `TuSimpleDataset` 一致的字段：

```python
{
    "image": tensor_like,
    "lanes": [N_lane, num_y],
    "valid_mask": [N_lane, num_y],
    "meta": {...},
    "cls_target": ...,
    "offset_label": ...,
    "offset_valid_mask": ...,
    "anchor_xs": ...,
    "anchor_valid_mask": ...,
    "anchor_y_samples": ...,
}
```

### 核心任务

把 CULane 原始标注转换为当前工程的统一 lane 表示：

```text
固定 y_samples 上的 x 序列 + valid_mask
```

### 实现步骤

1. 读取 CULane 图像路径与对应标注文件。
2. 从标注中解析每条车道线点集 `[(x, y), ...]`。
3. 按 `y` 对点排序。
4. 在统一的 `y_samples` 上做插值，得到 `x(y_samples)`。
5. 对插值失败、超出范围、无有效点的位置标记 `valid_mask = 0`。
6. 复用当前 `AnchorGenerator` 和 `LabelAssigner` 生成监督信号。

### 注意事项

- 不要一开始就强行做车道线延长。
- 插值失败的位置必须保持 invalid，不要伪造监督。
- 坐标系统必须与 resize 前后变换保持一致。
- `meta` 中要保留原图路径、原图尺寸、必要的评估导出信息。

## 阶段 2：可视化验证

### 修改文件

```text
tools/vis_dataset.py
lane_det/datasets/__init__.py
```

### 目标

让可视化脚本支持：

```python
if cfg["dataset"]["name"] == "culane":
    dataset = CULaneDataset(...)
```

### 验证标准

至少确认以下几点：

1. 原始 lane 绘制位置正确。
2. 固定 y 采样后的 lane 没有明显形变。
3. `valid_mask` 符合预期。
4. Anchor 匹配后存在合理数量的正样本。

如果这一步不稳定，不进入训练阶段。

## 阶段 3：匹配统计与阈值校准

### 修改文件

```text
lane_det/anchors/label_assigner.py
tools/train.py
tools/vis_dataset.py
```

### 说明

当前 `LabelAssigner` 已经是基于 shared valid points 的 mask-aware 匹配，不需要重写这一套。这里的重点不是“新增 mask-aware”，而是确认当前匹配策略在 CULane 上是否仍然有效。

### 核心检查点

1. 正样本数量是否过少。
2. 短车道线是否大量匹配失败。
3. 侧向车道线在 `common_points/top_region/geo gating` 下是否被过度过滤。
4. `line_iou_width`、`min_common_points`、`top_max_mean_err` 是否需要单独为 CULane 调参。

### 原则

- 不改 Anchor 结构。
- 不引入全新 IoU 定义。
- 先做统计和轻量阈值调整，再考虑结构性修改。

## 阶段 4：新增 CULane 配置

### 新增文件

```text
configs/culane_baseline.yaml
```

### 配置字段必须与当前工程一致

不要新造字段名，直接复用现有格式：

```yaml
dataset:
  name: culane
  root: ...
  list_file: ...
  img_size: [1640, 590]
  y_samples: 72
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

anchor:
  x_positions: ...
  angles: ...
  num_y: ...
```

注意：

- 使用 `dataset.list_file`，不要写成 `dataset.list`。
- 使用 `dataset.img_size`，不要放在顶层。
- 使用 `dataset.y_samples`，同时保持与 `anchor.num_y` 一致。

## 阶段 5：数据集工厂

### 修改文件

```text
lane_det/datasets/__init__.py
```

### 目标

新增统一入口：

```python
def build_dataset(cfg, split):
    if cfg["dataset"]["name"] == "tusimple":
        return TuSimpleDataset(cfg, split=split)
    if cfg["dataset"]["name"] == "culane":
        return CULaneDataset(cfg, split=split)
    raise ValueError(...)
```

### 原则

- 后续所有训练、推理、可视化入口都尽量走这个工厂。
- 不在脚本中继续直接硬编码 `TuSimpleDataset(...)`。

## 阶段 6：训练入口适配

### 修改文件

```text
tools/train.py
```

### 必改内容

1. 用 `build_dataset(cfg, split)` 替代 `TuSimpleDataset(...)`。
2. 将验证集构造逻辑从 TuSimple 专用的 `val.json` 推导中抽离。
3. 将验证阶段的 converter / evaluator 选择改为按 `dataset.name` 分支。

### 重点

当前 `train.py` 不仅绑定了 `TuSimpleDataset`，还绑定了 `TuSimpleConverter` 和 `LaneEval`。这部分必须一起拆开，否则 CULane 训练阶段跑不到有效验证。

## 阶段 7：推理入口适配

### 修改文件

```text
tools/infer.py
```

### 必改内容

1. 使用数据集工厂构造 `dataset`。
2. 将 `train/val/test` 的 list file 解析逻辑改成按数据集类型分支。
3. 将导出逻辑改成 `dataset.name` 分支，TuSimple 和 CULane 各走各的 converter。

### 原则

- 不要把 CULane 导出逻辑硬塞进 `TuSimpleConverter`。
- 不要继续依赖 `test_label.json` 这种 TuSimple 专用命名规则。

## 阶段 8：CULane 导出器

### 新增文件

```text
lane_det/metrics/culane_converter.py
```

### 目标

将模型解码后的 lane 转换为 CULane 官方评估所需格式。

### 必须明确的接口

在实现前先写清楚：

1. 单张图预测结果保存成什么格式。
2. 输出文件路径如何与原图路径对应。
3. 坐标使用原图尺度还是 resize 后尺度。
4. 是否需要按官方目录结构落盘。

没有这四点，不要开始写 converter。

## 阶段 9：CULane 评估脚本

### 新增文件

```text
tools/evaluate_culane.py
```

### 输出目标

```text
Precision
Recall
F1-score
```

### 必须补充的协议定义

在真正实现前先确认：

1. 使用官方评估程序还是 Python 复现。
2. IoU 阈值是多少。
3. 是否只输出总指标，还是同时输出场景分类指标。
4. 输入是预测目录还是预测列表文件。

## 阶段 10：Smoke Test

### Smoke Infer

```bash
python tools/infer.py --cfg configs/culane_baseline.yaml --ckpt <checkpoint>
```

检查项：

- 能运行
- 有输出
- 输出文件结构正确
- 可视化结果基本合理

### Smoke Train

```bash
python tools/train.py --cfg configs/culane_baseline.yaml
```

检查项：

- loss 正常下降
- 没有 NaN
- 有正样本
- 验证流程不报错

## 阶段 11：Full Train

### 目标

- 完整训练
- 正常保存 checkpoint
- 能稳定推理
- 能完成 CULane 评估

## 风险提示

最大风险不是模型，而是这一步：

```text
原始标注 -> 固定 y 采样表示
```

如果这一步有系统性偏差，会直接导致：

- 训练监督失真
- Anchor 匹配统计失真
- 指标结果失真

因此必须优先把数据读取与可视化链路做对。

## 验收标准

满足以下条件视为完成第一版接入：

- [ ] `CULaneDataset` 可正常读取样本
- [ ] `tools/vis_dataset.py` 可正常可视化 CULane 样本
- [ ] `tools/train.py` 可进行 CULane smoke train
- [ ] `tools/infer.py` 可进行 CULane smoke infer
- [ ] 能导出符合预期的预测结果
- [ ] `tools/evaluate_culane.py` 可输出基础指标

## 执行优先级

严格按以下顺序推进：

1. `CULaneDataset`
2. `vis_dataset.py`
3. 匹配统计与阈值校准
4. 数据集工厂
5. `train.py`
6. `infer.py`
7. `culane_converter.py`
8. `evaluate_culane.py`

## 一句话总结

在不改变现有模型主体的前提下，把 CULane 接入现有 Anchor-based 流水线，并优先保证数据表示、匹配和基础训练链路可用。
