# 基于 Anchor 的 CNN 车道线检测（ResNet18 + FPN）

本项目目标是从 0 到 1 跑通 **TuSimple** 车道线检测流程，采用 **固定 Y 采样 + Anchor 表示**，模型为 **ResNet18 + FPN + Anchor Head**，损失使用 **Focal Loss + Smooth L1**，匹配策略基于 **平均距离**。本 README 是工程落地文档，按阶段执行即可完成毕业设计的主线任务。

## 概览
- 表达方式：固定 Y 采样点上的 x 列表 + valid mask
- Anchor：参数化（x_bottom + angle / slope）
- 模型：ResNet18 + FPN + Anchor Head（分类 + offset 回归）
- Loss：FocalLoss（cls）+ Smooth L1（reg）
- 评估：先完成 TuSimple 全流程（CULane 作为后续扩展）

## 目标（本阶段必须做）
- TuSimple 数据解析 + 可视化检查
- Anchor 生成与标签分配（cls_label + offset_label + valid_mask）
- ResNet18 + FPN + Anchor Head 训练跑通
- 推理 + 可视化保存
- TuSimple 评估格式导出

## 里程碑（M1~M4）

### M1：数据与可视化基线（只做 TuSimple）
目标：
- 解析 TuSimple 标注，统一到固定 Y 采样格式
- 完成数据检查与可视化
产出物：
- `lane_det/datasets/tusimple.py`
- `tools/vis_dataset.py`
- 样例可视化图（GT 车道线）
验收标准：
- 能在 TuSimple 上加载数据
- `tools/vis_dataset.py` 能保存 GT 叠加图（>=10 张）

### M2：Anchor 与标签分配
目标：
- Anchor 参数化 + 生成
- 标签分配：cls_label、offset_label、valid_mask
产出物：
- `lane_det/anchors/anchor_generator.py`
- `lane_det/anchors/label_assigner.py`
- Anchor/匹配可视化
验收标准：
- 可视化 anchor 与 GT 的匹配结果
- 正负样本数量可控（非全负/全正）

### M3：模型与训练跑通
目标：
- ResNet18 + FPN + Anchor Head 前向
- FocalLoss + Smooth L1 训练
产出物：
- `lane_det/models/*`
- `lane_det/losses/*`
- `tools/train.py`
- loss 曲线/日志
验收标准：
- 单卡能跑通训练
- loss 曲线正常下降

### M4：推理、导出与评估
目标：
- 推理解码 + 可视化
- 生成 TuSimple 评估格式
产出物：
- `tools/infer.py`
- `tools/visualize.py`
- `tools/evaluate.py`
- TuSimple 评估输出文件
验收标准：
- 推理可视化图像保存成功
- 评估文件可被官方脚本读取

> 说明：CULane 放入“后续扩展”，本阶段不做。

## 当前进度更新（仅记录，不改变路线）
- M2 已完成并跑通：
  - 已实现 `lane_det/anchors/anchor_generator.py`（支持图像外部 Anchor 生成）
  - 已实现 `lane_det/anchors/label_assigner.py`（集成方向一致性与底部交点合理性门控）
  - 已优化 `lane_det/datasets/tusimple.py`（实现虚拟全长 GT 延伸，解决短线段漏检）
  - 已验证可视化结果，消除了交叉误配，覆盖了侧向车道线
- M3 已完成并跑通：
  - 已实现 `lane_det/models/` (ResNet18 + FPN + Anchor Head)
  - 已实现 `lane_det/losses/` (Focal Loss + Smooth L1)
  - 已实现 `tools/train.py` (数据加载 + 训练循环 + 日志保存)
  - 已修复 Focal Loss 数值稳定性与忽略样本问题 (Loss 正常下降)
  - 已通过 `tools/plot_loss.py` 实现 Loss 曲线可视化
- M4 已完成并跑通：
  - 已实现 `lane_det/postprocess/decoder.py` (Anchor 解码与阈值筛选)
  - 已实现 `lane_det/metrics/tusimple_converter.py` (TuSimple 格式转换与插值)
  - 已实现 `tools/infer.py` (端到端推理脚本，支持验证集评估)
  - 已实现 `tools/visualize.py` (预测结果可视化)
  - 已实现 `tools/evaluate.py` (TuSimple 官方评估脚本 Python 实现)
  - 已验证推理流程，可生成 `pred.json` 并输出可视化图片
- 当前已验证可输出 GT 可视化到 `outputs/visualizations/gt`，训练 Checkpoints 保存到 `outputs/checkpoints`，推理可视化到 `outputs/visualizations/pred`

## Quickstart（占位命令，按后续实现调整）

### 1) 安装依赖
```bash
pip install -r requirements.txt
```

### 2) 准备数据
```
data/
  tusimple/    # 放置 TuSimple 数据与标注
```

### 3) 数据可视化检查（M1 必做）
```bash
python tools/vis_dataset.py --cfg configs/tusimple_res18_fpn.yaml --out outputs/visualizations/gt
```

### 4) 训练（M3）
```bash
python tools/train.py --cfg configs/tusimple_res18_fpn.yaml
```

### 5) 推理与可视化（M4）
```bash
python tools/infer.py --cfg configs/tusimple_res18_fpn.yaml --ckpt outputs/checkpoints/last.pth
python tools/visualize.py --cfg configs/tusimple_res18_fpn.yaml --out outputs/visualizations/pred
```

### 6) 官方评估（M4）
```bash
# 假设 GT 文件为 data/tusimple/test_label.json
python tools/evaluate.py --pred pred.json --gt data/tusimple/test_label.json
```

## 数据集与标注格式

### TuSimple 标注 -> 统一 lane 表示
- TuSimple 标注通常是以固定 y 列表对应的 x 值给出。
- 统一表示为：
  - `x_list`: 长度为 `num_y` 的 x 列表
  - `valid_mask`: 长度为 `num_y` 的 0/1 有效掩码（无效点标 0）
- 若原始标注中该 y 处无有效 x，则 `valid_mask=0`。

### Transforms 必须同步
所有 transforms 必须同时作用于 image 与 lane 标注。

推荐最小 transforms（M1 必做）：
- resize 到固定大小（与 config 保持一致）
- normalize

后续可选（现在不做）：
- random horizontal flip
- random affine
- color jitter

### 数据检查与可视化（M1 必做）
- 提供 `tools/vis_dataset.py`，用于检查标注与统一表示。
- 输出 GT 叠加图，验证解析正确。

## Anchor 设计与标签分配

### Anchor 参数化建议
推荐：`anchor = (x_bottom, angle)` 或 `anchor = (x_bottom, slope)`
- 原因：车辆视角中车道线近似直线，底部位置与斜率能覆盖大多数情况。
- 可通过离散的 `x_bottom` 与 `angle/slope` 组合生成 anchor 集合。

### Y 采样与图像尺寸
- TuSimple 推荐 `num_y = 56`。
- 图像尺寸 `W, H` 通过 config 固定，anchor 与标签分配都以该尺寸为准。

### 匹配策略（可执行步骤）
1) 对每条 GT 与每个 anchor，在共同有效的 y 点上计算平均距离：
   - `dist = mean(|x_gt - x_anchor|)`
2) 若 `dist < pos_thr` -> 正样本
3) 若 `dist > neg_thr` -> 负样本
4) 介于阈值之间 -> ignore

### 标签与形状约定
- `cls_label`: shape `[num_anchors]`，值为 {0,1}
- `offset_label`: shape `[num_anchors, num_y]`，为 x 偏移量
- `valid_mask`: shape `[num_anchors, num_y]`，用于 mask 无效点

### 单元测试与可视化验证（M2 必做）
- 最小要求：
  - 可视化 matched anchors 与 GT
  - 检查正负样本比例
- 实现建议：在 `tools/vis_dataset.py` 增加 anchor/匹配叠加显示

## 模型结构
- Backbone：ResNet18
- Neck：FPN
- Head：Anchor 分类 + Offset 回归
- 输出：
  - `cls_logits`：每个 anchor 的分类分数
  - `offsets`：每个 anchor 在每个 y 处的 x 偏移

## 训练流程

### Loss 组成
- 分类：Focal Loss
- 回归：Smooth L1
- 对回归项应用 `valid_mask`

### 建议默认权重（写入 config）
- `cls_weight = 1.0`
- `reg_weight = 1.0`
- `focal_alpha = 0.25`
- `focal_gamma = 2.0`
- `smooth_l1_beta = 1.0`

## 推理流程
- 解码：`anchor + offsets -> lanes points`
- 输出格式：
  - 统一内部格式（x_list + valid_mask）
  - TuSimple 导出格式（M4 完成）

## 评估流程（TuSimple）
- 生成符合 TuSimple 官方评估脚本的预测文件。
- 使用 `tools/evaluate.py` 进行评估：
  1) 推理生成 lane 结果 (`pred.json`)
  2) 运行评估脚本对比预测结果与 GT (`test_label.json`)
  3) 输出 Accuracy, FP, FN 指标

> CULane 评估留到“后续扩展”，现在不做。

## 目录结构与职责
```text
configs/               # 配置文件（训练/推理）
data/                  # 数据集根目录
lane_det/
  datasets/            # 数据加载与 transforms
  anchors/             # Anchor 生成与标签分配
  models/              # Backbone / FPN / Head / Detector
  losses/              # Focal / Smooth L1
  postprocess/         # 解码与 NMS（可选）
  metrics/             # 评估逻辑
  visualization/       # 绘图与可视化工具
  utils/               # 配置/日志/通用函数
tools/                 # 训练/推理/可视化脚本
outputs/               # 日志/权重/可视化输出
```

## 配置示例（YAML 片段）
```yaml
dataset:
  name: tusimple
  root: data/tusimple
  img_size: [1280, 720]
  y_samples: 56

anchor:
  x_positions: 40
  angles: [-20, -10, 0, 10, 20]
  num_y: 56
  pos_thr: 12.0
  neg_thr: 18.0

model:
  backbone: resnet18
  fpn_out: 128
  head_channels: 128

loss:
  focal_alpha: 0.25
  focal_gamma: 2.0
  smooth_l1_beta: 1.0
  cls_weight: 1.0
  reg_weight: 1.0

train:
  batch_size: 8
  lr: 0.001
  epochs: 50
  weight_decay: 0.0001
```

## 实现要求（必须遵守）
- Anchor 生成可缓存，避免重复计算。
- 标签分配在 dataset 或 collate 中完成，不放在模型 forward。
- transforms 必须同步作用于 image 与 lane 标注。

## 常见问题 / 排错
- 正样本过少：
  - 检查 pos_thr 是否过小
  - Anchor 角度/位置覆盖不足
- loss 不收敛：
  - 学习率过大或过小
  - cls/reg 权重不平衡
- 预测偏移异常：
  - 检查 offset 的归一化方式
  - 确认有效 mask 是否正确
- 推理可视化为空：
  - 检查解码逻辑与阈值
  - 检查 anchors 是否正确生成

## 后续扩展（现在不做）
以下内容不影响当前里程碑推进，先不做：

- 支持 CULane
  - 目的：扩展数据集适配能力
  - 修改位置：`lane_det/datasets/culane.py`、`lane_det/metrics/`
  - 建议：若时间充足可做（中等工作量）

- Lane NMS / lane merging
  - 目的：减少重复/重叠车道线
  - 修改位置：`lane_det/postprocess/nms.py`
  - 建议：可选优化（不影响主流程）

- 曲率模板 anchors
  - 目的：提升对弯道的适配
  - 修改位置：`lane_det/anchors/anchor_generator.py`
  - 建议：若结果不足可尝试（中等工作量）

- 更强 backbone（ResNet34/50）对比
  - 目的：提升精度做对比实验
  - 修改位置：`lane_det/models/backbone_resnet.py`
  - 建议：毕业设计可选对比实验（中等工作量）

- 更丰富数据增强（random affine、color jitter）
  - 目的：提升泛化
  - 修改位置：`lane_det/datasets/transforms.py`
  - 建议：可选（不影响主流程）

- 速度优化（cache、mixed precision）
  - 目的：提升训练/推理效率
  - 修改位置：`tools/train.py`、`lane_det/utils/`
  - 建议：若时间充足再做（非必需）


  代码架构
  /root/project/lane-det/
├── .gitignore
├── README.md
├── requirements.txt
├── command.txt
│
├── configs/
│   ├── tusimple_res18_fpn.yaml
│   ├── tusimple_res18_fpn_matchv2.yaml
│   └── tusimple_test.yaml
│
├── lane_det/                          # 核心 Python 包
│   ├── __init__.py
│   │
│   ├── anchors/
│   │   ├── __init__.py
│   │   ├── anchor_generator.py
│   │   └── label_assigner.py
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── tusimple.py
│   │   └── transforms.py
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── focal_loss.py
│   │   ├── reg_loss.py
│   │   └── soft_line_loss.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── tusimple_converter.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py
│   │   ├── detector.py
│   │   ├── fpn.py
│   │   ├── head.py
│   │   └── anchor_feature_pooler.py
│   │
│   ├── postprocess/
│   │   ├── __init__.py
│   │   └── decoder.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py
│
├── tools/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── visualize.py
│   ├── vis_dataset.py
│   ├── plot_loss.py
│   └── prepare_tusimple_split.py
│
├── tests/
│   ├── test_losses.py
│   └── test_models.py
│
└── *.md (技术文档)
    ├── ANCHOR_MATCHING_SPEC.md
    ├── CURRENT_TECHNICAL_PATH.md
    ├── MATCH_OPTIMIZATION_STEPS.md
    ├── anchor_match.md
    ├── compere_find.md
    ├── final_refine.md
    ├── good_sample.md
    ├── refine_loss.md
    ├── refine_module.md
    └── train_process.md

    /root/autodl-tmp/
├── .autodl/
│   ├── autopanel.monitor.db
│   └── autopanel.security.db
│
├── checkpoints/                       # 模型 checkpoint（空或少量）
│
├── outputs/                           # 训练/推理输出
│
└── datasets/
    │
    ├── tusimple/                      # 当前使用的 TuSimple 数据
    │   ├── train.json                 # 训练集索引
    │   ├── val.json                   # 验证集索引
    │   ├── label_data_0313.json
    │   ├── label_data_0531.json
    │   ├── label_data_0601.json
    │   │
    │   └── clips/
    │       ├── 0313-1/
    │       │   ├── 10000/              # clip_id
    │       │   │   ├── 1.jpg
    │       │   │   ├── 2.jpg
    │       │   │   └── ... (20 帧)
    │       │   ├── 10020/
    │       │   └── ...
    │       ├── 0313-2/
    │       ├── 0531/
    │       └── 0601/
    │
    └── archive/                       # 归档数据
        ├── test_label_new.json
        │
        └── TUSimple/
            ├── test_label.json
            │
            ├── test_set/
            │   ├── readme.md
            │   ├── test_tasks_0627.json
            │   └── clips/
            │       └── 0530/
            │           ├── 1492626047222176976_0/
            │           │   ├── 1.jpg
            │           │   ├── 2.jpg
            │           │   └── ... (20 帧)
            │           ├── 1492626126171818168_0/
            │           └── ... (更多 clip)
            │
            └── train_set/
                ├── readme.md
                ├── label_data_0313.json
                ├── label_data_0531.json
                ├── label_data_0601.json
                └── clips/
                    ├── 0313-1/
                    ├── 0313-2/
                    ├── 0531/
                    └── 0601/
                    └── (结构同 tusimple/clips)
