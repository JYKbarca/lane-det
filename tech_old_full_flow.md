# 车道线项目完整技术流程整理

本文档用于系统整理当前车道线检测项目的完整技术流程。整理方式遵循两个原则：

- 按真实代码实现来梳理，而不是只按理论流程概括
- 按步骤持续补充，从项目启动一直整理到训练、推理、评估与结果输出

当前已完成：

- 步骤 1：项目整体入口与总数据流
- 步骤 2：TuSimple 标注如何变成训练标签
- 步骤 3：模型结构与前向传播
- 步骤 4：损失函数与训练循环
- 步骤 5：验证、推理、后处理与评估
- 步骤 6：输出目录、实验产物与完整闭环
- 步骤 7：两条技术分支的关系与当前 row/grid 主线

后续若项目继续演化，可在本文档中追加：

- 不同实验版本的对比总结
- 两条分支的最终消融对照
- 论文写作中的方法章节收束版本

---

## 一、项目总目标

这个项目是一个面向 TuSimple 数据格式的车道线检测工程，核心任务是对道路图像进行结构化车道线预测，并输出可用于官方评估的结果文件。

从仓库当前真实状态看，项目已经不是单一路线，而是在同一套数据与评估底座上形成了两条技术分支：

- 分支 A：`anchor-based`。这是最早实现完成、工程链路最完整的一条路线，核心是 Ray Anchor 生成、Anchor-GT 匹配、Anchor 特征采样、逐点偏移回归与 lane-level 后处理。
- 分支 B：`row/grid-based`。这是在 anchor 路线基础上继续演化出的新路线，核心是固定最多 `K` 条车道槽位，在固定 `Y` 个 row 上做横向位置网格分类，并通过期望解码恢复连续坐标。

如果按毕业论文的口径来定义，这两条路线的角色应当区分为：

- `anchor-based`：第一条完整跑通的结构化检测基线，也是问题诊断与路线转向的依据
- `row/grid-based`：当前重点推进的主线方案，也是后续实验与论文方法部分更应该着重描述的对象

因此，项目当前可以概括为：

`TuSimple 数据集 -> 固定 y 采样的统一车道线表示 -> 分支 A（Ray Anchor 匹配与偏移回归）/ 分支 B（row/grid 位置分类） -> TuSimple 格式导出 -> Accuracy / FP / FN 评估`

项目当前核心模块分布如下：

- 配置文件：`configs/`
- 数据集与变换：`lane_det/datasets/`
- Ray Anchor 生成与标签分配：`lane_det/anchors/`
- 模型结构：`lane_det/models/`
- 损失函数：`lane_det/losses/`
- 后处理与解码：`lane_det/postprocess/`
- 指标转换与评估：`lane_det/metrics/`、`tools/evaluate.py`
- 训练、推理、可视化脚本：`tools/`

---

## 二、完整流程总览

从端到端角度看，这个项目的完整技术流程应拆成“共享底座 + 分支化检测主链”两部分：

1. 准备 TuSimple 原始标注和图像数据
2. 合并原始标注并划分 `train.json`、`val.json`
3. 通过 `TuSimpleDataset` 读取图像与标注
4. 将原始 `lanes + h_samples` 转为统一内部表示
5. 对图像和车道标注同步做数据增强与缩放
6. 在共享表示之后，进入两条检测路线之一
7. 分支 A：生成 Ray Anchor，完成 Anchor-GT 匹配，构造 `cls_label + offset_label + offset_valid_mask`
8. 分支 A：`LaneDetector` 输出每条 Anchor 的分类分数和逐点偏移，并经过解码、NMS、平滑后导出结果
9. 分支 B：`RowTargetBuilder` 将车道线整理为固定槽位、固定 row 的 `exist + x_coords + coord_mask`
10. 分支 B：`RowLaneDetector` 输出 `exist_logits + row_valid_logits + grid_logits`，并经过 row 解码后直接导出结果
11. 两条分支最终都通过 `TuSimpleConverter` 输出 `pred.json`
12. 两条分支最终都通过 `LaneEval` 计算 `Accuracy`、`FP`、`FN`
13. 通过 `vis_dataset.py`、`visualize.py`、`visualize_row.py`、`plot_loss.py` 回查问题并继续迭代

本文在结构上仍保留原有的步骤 2 到步骤 6，对 `anchor-based` 路线做完整细拆；随后在步骤 7 中集中总结 `row/grid-based` 路线，以及两条路线在论文中的关系。

---

## 步骤 1：项目整体入口与总数据流

这一步解决的问题是：这个项目从哪里开始运行，外层是怎么把整条链路串起来的。

### 1.1 外层入口脚本

当前项目最核心的外层脚本可以分成共享脚本与分支脚本两类：

- 共享脚本：`tools/prepare_tusimple_split.py`、`tools/evaluate.py`
- `anchor-based` 分支：`tools/train.py`、`tools/infer.py`
- `row/grid-based` 分支：`tools/train_row.py`、`tools/infer_row.py`

另外还有几类辅助脚本：

- 数据与匹配可视化：`tools/vis_dataset.py`
- Anchor 分支预测可视化：`tools/visualize.py`
- Row 分支预测可视化：`tools/visualize_row.py`
- 训练曲线绘制：`tools/plot_loss.py`

它们分别对应：

- 数据列表准备
- 分支化训练与验证
- 分支化 checkpoint 推理导出
- TuSimple 指标评估

### 1.2 配置文件是整个流程的总开关

当前最关键的配置文件有 3 类：

- `anchor-based` 训练主配置：`configs/tusimple_res18_fpn_matchv2.yaml`
- `row/grid-based` 训练主配置：`configs/tusimple_row_res18_fpn.yaml`
- 测试/推理配置：`configs/tusimple_test.yaml`

这些配置文件共同定义了整条流程所需的核心参数，包括：

- 数据集根目录
- 训练/验证列表文件
- 输入图像尺寸
- `y_samples` 数量
- 分支相关参数
  - Anchor 的位置、角度和匹配阈值
  - Row/Grid 的 `max_lanes`、`num_y`、`num_grids`
- 模型结构参数
- 损失函数权重
- 训练超参数
- 验证和推理解码参数

也就是说，这个项目的技术流程不是完全硬编码在 Python 里，而是由：

`入口脚本 + YAML 配置`

共同决定。

### 1.3 数据流从哪里开始

整个项目的数据流起点是 TuSimple 原始标注文件，也就是多个：

- `label_data_0313.json`
- `label_data_0531.json`
- `label_data_0601.json`

这些文件和对应图像一起构成原始训练数据。

### 1.4 训练前先生成 train/val 列表

脚本 `tools/prepare_tusimple_split.py` 的职责是：

1. 扫描根目录下所有 `label_data_*.json`
2. 逐行读取每一条 TuSimple 标注记录
3. 合并成一个总样本集
4. 随机打乱
5. 按 `val_ratio` 划分训练集和验证集
6. 输出：
   - `train.json`
   - `val.json`

这里生成的是 jsonl 格式文件，也就是每行一个 JSON 样本。

### 1.5 训练与推理从哪里进入数据集

`anchor-based` 训练入口 `tools/train.py` 会：

1. 读取 YAML 配置
2. 创建训练集 `TuSimpleDataset(cfg, split="train")`
3. 创建验证集 `TuSimpleDataset(val_cfg, split="val")`
4. 通过 `DataLoader + collate_fn` 组成带有 Anchor 标签的 batch
5. 进入 `LaneDetector` 的训练和验证流程

`row/grid-based` 训练入口 `tools/train_row.py` 会：

1. 读取 YAML 配置
2. 创建训练集 `TuSimpleDataset(cfg, split="train")`
3. 创建验证集 `TuSimpleDataset(val_cfg, split="val")`
4. 在 `collate_fn` 中使用 `RowTargetBuilder` 构造 row/grid 监督目标
5. 进入 `RowLaneDetector` 的训练和验证流程

`anchor-based` 推理入口 `tools/infer.py` 会：

1. 读取 YAML 配置
2. 加载指定 checkpoint
3. 创建 `TuSimpleDataset(..., split="val" / "test")`
4. 批量推理
5. 通过 `LaneDecoder` 解码为车道线
6. 转为 TuSimple 输出格式并保存

`row/grid-based` 推理入口 `tools/infer_row.py` 会：

1. 读取 YAML 配置
2. 加载指定 checkpoint
3. 创建 `TuSimpleDataset(..., split="train" / "val" / "test")`
4. 批量推理 `exist_logits + row_valid_logits + grid_logits`
5. 通过 row 解码恢复连续车道坐标
6. 转为 TuSimple 输出格式并保存

评估入口 `tools/evaluate.py` 则负责：

1. 读取预测文件 `pred.json`
2. 读取 GT 文件
3. 逐样本计算 TuSimple 指标
4. 输出 `Accuracy`、`FP`、`FN`

### 1.6 第 1 步的本质

从工程角度讲，第 1 步做的事情不是训练模型，而是搭起端到端运行骨架：

`配置读取 -> 数据列表准备 -> 数据集实例化 -> 分支化训练/推理脚本接管后续流程 -> 统一评估与可视化`

所以第 1 步的作用可以概括为：

“确定实验参数，并把原始 TuSimple 数据正式接入整个工程流水线。”

---

## 步骤 2：TuSimple 标注如何变成训练标签

这一步是整个项目最重要的数据中间层。它解决的问题是：

`原始 JSON 标注 -> 统一车道线表示 -> 数据增强/缩放 -> 分支 A 的 Anchor 标签 / 分支 B 的 row/grid 标签`

相关核心文件：

- `lane_det/datasets/tusimple.py`
- `lane_det/datasets/transforms.py`
- `lane_det/datasets/row_target_builder.py`
- `lane_det/anchors/anchor_generator.py`
- `lane_det/anchors/label_assigner.py`

### 2.1 TuSimple 原始标注长什么样

TuSimple 每条样本的关键字段主要有：

- `raw_file`：图像相对路径
- `lanes`：多条车道线，每条是一个 `x` 坐标序列
- `h_samples`：这些 `x` 坐标对应的固定 `y` 采样位置

它的含义是：

- `h_samples[i]` 表示第 `i` 个纵向采样高度
- `lanes[j][i]` 表示第 `j` 条车道线在该高度的横向坐标
- 若该点不存在，则通常为负值，表示无效

所以原始标注本身已经是一种“固定 y 采样的车道线表达”。

### 2.2 Dataset 如何读取单条样本

`TuSimpleDataset.__getitem__()` 每次读取一条样本时，主要做以下事情：

1. 从 jsonl 记录中取出当前样本
2. 根据 `raw_file` 拼接图像路径
3. 读取图像并转换为 RGB
4. 解析 `lanes` 和 `h_samples`
5. 对图像和标注同步应用变换
6. 若启用 Anchor 配置，则生成 Ray Anchor 并分配监督标签
7. 若走 row/grid 分支，则保留标准化后的 `lanes`、`valid_mask`、`h_samples`，供后续 `RowTargetBuilder` 使用
8. 返回训练或推理所需字段

### 2.3 原始车道线如何转成统一内部表示

`_parse_lanes()` 会把每条车道线解析为两个数组：

- `lanes`：形状约为 `[N_lane, num_y]`
- `valid_mask`：形状同样为 `[N_lane, num_y]`

处理规则是：

- `x >= 0` 的位置视为有效点
- `x < 0` 的位置视为无效点
- 无效点在 `lanes` 里先置为 `0.0`
- 同时在 `valid_mask` 中记为 `0`

这样做之后，每条车道线都被整理成：

- 固定长度的 `x` 序列
- 同长度的有效点掩码

这就是本项目后续所有模块共同使用的内部标准表示。

### 2.4 图像与标注必须同步变换

图像变换位于 `lane_det/datasets/transforms.py`。

训练阶段的变换链为：

- `RandomHorizontalFlip`
- `ColorJitter`
- `ResizeNormalize`

验证和推理阶段则只保留：

- `ResizeNormalize`

同步规则如下。

#### 2.4.1 随机水平翻转

若触发翻转：

- 图像做左右镜像
- 所有有效车道点同步做横坐标映射
- 变换公式为：`x -> (w - 1 - x)`

#### 2.4.2 颜色扰动

颜色扰动只修改图像外观，不修改车道线几何坐标。

#### 2.4.3 缩放与归一化

`ResizeNormalize` 会：

- 将图像 resize 到配置指定大小
- 按宽度比例缩放车道线 `x`
- 按高度比例缩放 `h_samples`
- 对图像做归一化
- 将图像转为 `[C, H, W]`

这意味着后续 Anchor 生成、匹配、回归、推理，都是在“变换后的图像坐标系”中完成的。

### 2.5 为什么还要修正 `h_samples` 数量

项目代码中对异常样本还做了一层长度修正：

- 若当前样本的 `h_samples` 数量与配置中的 `y_samples` 不一致
- 就重新插值到统一长度

这一层的意义是保证整个训练过程中的维度一致，否则以下模块都可能出问题：

- Anchor 生成维度
- Head 回归输出维度
- 损失计算维度

所以不论原始样本是否完全规整，最终都会被对齐到统一的采样长度。

### 2.6 Anchor 是什么

这个项目中的 Anchor 不是目标检测里的矩形框，而是一组“候选车道线模板”。更准确地说，它属于 **Ray Anchor / 射线式 Anchor** 范式：每条 Anchor 都由一个起始位置和一个方向角确定，再投影到固定的 `y_samples` 上，形成一条候选车道线轨迹。

每条 Anchor 内部主要包含：

- `anchor_xs`：Anchor 在各个 `y` 采样点上的横坐标
- `valid_mask`：Anchor 在哪些采样点落在图像范围内
- `x_bottom`：Anchor 在底部的参考横坐标
- `angles`：Anchor 对应的方向角
- `y_samples`：Anchor 所使用的纵向采样位置

如果只看主干参数化形式，可以近似理解为：

`Anchor = (start_x / x_bottom, angle, y_samples)`

这和论文里常说的 Ray Anchor 思路是一致的，只是当前工程在此基础上又补充了从左右边界出发的 side anchors。

### 2.7 Anchor 如何生成

Anchor 生成位于 `lane_det/anchors/anchor_generator.py`，本质上都是“给定起点 + 给定角度，生成一条射线式候选线”，只是当前实现分成两类来源。

#### 2.7.1 Bottom anchors

底部 Anchor 的生成方式是：

- 枚举一组 `x_bottom`
- 枚举一组 `angle`
- 对每个组合生成一条从图像底部出发、按固定角度延伸的参考直线

所以这类 Anchor 本质上就是最典型的 bottom-ray anchors，适合表示那些“从图像底部进入画面”的车道线。

#### 2.7.2 Side anchors

侧边 Anchor 的生成方式是：

- 从左边界或右边界若干 `y_start` 位置出发
- 只使用朝图像内部延伸的角度
- 构成从左右边界进入画面的候选车道线

这类 Anchor 仍然是 Ray Anchor 的扩展形式，只是起点不在底边，而在左右边界。它的作用是增强对以下情况的覆盖：

- 从侧面进入画面的车道线
- 底部不可见但中上部可见的车道线
- 较短或偏侧的车道线

### 2.8 为什么要给 Anchor 做缓存

Anchor 集只依赖这些固定因素：

- 输入图像尺寸
- `y_samples`
- `x_positions`
- `angles`
- side anchor 相关参数

因此只要这些配置不变，生成结果就是固定的。项目会将 Anchor 缓存为 `.npz` 文件，避免重复计算，提高训练和推理效率。

### 2.9 GT 和 Anchor 如何匹配

标签分配逻辑在 `lane_det/anchors/label_assigner.py` 中完成。

它的核心思路不是 box IoU，而是 `Line IoU`。基本思想为：

- 在 GT 与 Anchor 共同有效的采样点上进行比较
- 在每个采样点，把车道线看作一个带宽度的水平线段
- 比较两条线在这些点上的重合程度
- 最终将所有共同点的重合结果聚合成一条线级别的 IoU

因此它本质上衡量的是：

“Anchor 这条候选线，和 GT 这条真实车道线，在整条路径上的形状重合程度。”

### 2.10 匹配时做了哪些约束

这个项目不是只看 IoU，还引入了多层几何约束来减少误配。

主要包括：

- 共同有效点数量约束
- 共同有效点比例约束
- top 区域平均误差约束
- 方向角一致性约束
- 底部交点合理性约束

这些约束的作用分别是：

- 防止只靠极少数点误配
- 防止线条底部靠近但整体方向错误
- 防止顶部严重发散的候选 Anchor 被当成正样本
- 防止 X 型交叉样式的错误匹配

### 2.11 如何划分正样本、负样本和忽略样本

对每个 Anchor，代码都会找到与其最匹配的一条 GT，并得到一个最佳 IoU。

然后根据阈值划分为：

- 正样本：`best_iou >= pos_thr`
- 负样本：`best_iou <= neg_thr`
- 忽略样本：介于两者之间

另外，当前代码还支持一套辅助策略：

- `topk_per_gt`
- `min_force_pos_iou`
- `soft_gating`

这些机制的作用主要是：

- 保证某些 GT 至少能分到足够的正样本
- 对接近阈值但不是完全错误的候选进行柔性惩罚

### 2.12 最终训练监督标签是什么

匹配结束后，Dataset 会为每个样本生成这些核心字段：

- `cls_label`
- `offset_label`
- `offset_valid_mask`
- `matched_gt_idx`
- `match_stats`

其中最核心的是前 3 个。

#### 2.12.1 分类标签 `cls_label`

表示每个 Anchor 的类别：

- `1`：正样本
- `0`：负样本
- `-1`：忽略样本

#### 2.12.2 回归标签 `offset_label`

表示 GT 相对当前 Anchor 的横向偏移量：

`offset = gt_x - anchor_x`

也就是说，模型不是直接预测整条车道线的绝对位置，而是学习：

“当前 Anchor 在每个采样点上应该往左或往右修正多少像素。”

#### 2.12.3 回归有效掩码 `offset_valid_mask`

表示哪些采样点允许参与回归损失计算。

这一步很重要，因为并不是每个 Anchor 的每个位置都应该被监督：

- 有些点超出图像范围
- 有些点在 GT 中原本就是无效的
- 有些点虽然参与了匹配判断，但不适合进入回归监督

所以训练时只会在 `offset_valid_mask == 1` 的位置上计算回归损失。

### 2.13 Dataset 最终输出给训练器的字段

经过 `collate_fn` 组装后，训练阶段拿到的 batch 会随分支不同而变化：

- `anchor-based` 分支主要包含：
  - `images`
  - `cls_targets`
  - `offset_labels`
  - `offset_masks`
  - `anchors`
  - `metas`
- `row/grid-based` 分支主要包含：
  - `images`
  - `exist_targets`
  - `x_targets_norm`
  - `coord_masks`
  - `row_h_samples`
  - `metas`

也就是说，从这一步开始，模型训练已经不再直接面对原始 TuSimple JSON，而是面对按不同技术路线整理完成的监督张量。

### 2.14 第 2 步的本质

第 2 步完成的是整个项目最关键的数据桥接工作：

`原始车道线标注 -> 标准化车道线表示 -> 分支化监督目标构造`

没有这一步，后面的模型训练无法成立。

---

## 步骤 3：模型结构与前向传播

这一步解决的问题是：

`已经完成标签分配的 batch 数据 -> 如何进入神经网络 -> 网络内部如何提取特征 -> 如何沿 Ray Anchor 轨迹取特征 -> 如何输出分类分数和回归偏移`

下面这一步主要详细拆解 `anchor-based` 分支的模型结构。`row/grid-based` 分支的模型结构会在步骤 7 中单独总结。

这一步对应的核心文件为：

- `lane_det/models/detector.py`
- `lane_det/models/backbone.py`
- `lane_det/models/fpn.py`
- `lane_det/models/head.py`
- `lane_det/models/anchor_feature_pooler.py`

如果说步骤 2 是“把数据变成可监督的训练样本”，那么步骤 3 就是：

“把这些样本送进网络，并把图像特征转成每条 Ray Anchor 的分类与几何修正预测。”

### 3.1 模型总结构概览

当前项目的检测器类是 `LaneDetector`，它由 3 个主模块组成：

1. Backbone：`ResNet18`
2. Neck：`LaneFPN`
3. Head：`AnchorHead`

完整链路可以概括成：

`images -> ResNet18 提取多层特征 -> FPN 融合成多尺度金字塔特征 -> 沿 Ray Anchor 轨迹池化特征 -> 分类头输出每条 Anchor 的质量感知分类分数 -> 回归头输出每条 Anchor 在各 y 位置上的偏移量`

如果启用 refinement，则回归部分还会多一层：

`stage1 粗回归 -> stage2 残差细化 -> final offsets`

### 3.2 模型输入是什么

在训练阶段，`tools/train.py` 中的一个 batch 主要包含：

- `images`：`[B, 3, H, W]`
- `anchors`：AnchorSet 对象
- `cls_targets`
- `offset_labels`
- `offset_masks`

在前向传播中，真正输入模型的是：

- 图像张量 `images`
- 当前 batch 共用的 Anchor 集 `anchors`

也就是说，这个模型不是传统的“纯卷积后直接密集预测”的方式，而是：

- 先卷积提特征
- 再根据预先定义好的 Ray Anchor 轨迹去特征图上取样
- 最后对每条 Anchor 单独做分类和回归

### 3.3 总控模块 `LaneDetector`

`lane_det/models/detector.py` 中的 `LaneDetector` 是总装模块，它负责把 backbone、FPN 和 head 串起来。

它的初始化过程主要做了三件事：

1. 构建 `ResNet18(pretrained=True)`
2. 构建 `LaneFPN(in_channels_list=[64, 128, 256, 512], out_channels=fpn_out, use_c2=True)`
3. 构建 `AnchorHead(...)`

其中配置里会进一步决定 head 的行为，例如：

- `use_masked_pooling`
- `debug_shapes`
- `use_refinement`
- `reg_seq_layers`
- `reg_hidden_channels`
- `reg_kernel_size`

`LaneDetector.forward(images, anchors, return_aux=False)` 的逻辑非常清晰：

1. 图像先进入 backbone
2. 得到多层卷积特征
3. 特征送入 FPN
4. 得到融合后的多尺度特征
5. 将 FPN 特征和 Anchor 一起送入 head
6. 输出：
   - `cls_logits`
   - `reg_preds`
   - 若 `return_aux=True`，还返回中间回归阶段信息

因此 `LaneDetector` 自己并不做复杂几何计算，它的职责更接近于：

“负责调度整个特征提取和预测链路。”

### 3.4 Backbone：ResNet18 做了什么

Backbone 位于 `lane_det/models/backbone.py` 中。

当前实现使用的是 torchvision 的标准 `resnet18`，并保留到 `layer4` 为止，不再使用最后的：

- `avgpool`
- `fc`

因为这个任务不是图像分类，而是要保留空间结构信息做密集几何预测。

#### 3.4.1 ResNet18 的输出层级

当前 forward 返回 4 层特征：

- `c2`
- `c3`
- `c4`
- `c5`

对应的通道数分别是：

- `c2`: 64
- `c3`: 128
- `c4`: 256
- `c5`: 512

对应的下采样倍率分别约为：

- `c2`: stride 4
- `c3`: stride 8
- `c4`: stride 16
- `c5`: stride 32

也就是说，ResNet18 在这个项目中的作用不是直接预测，而是把输入图像编码成不同空间尺度、不同语义层次的特征图。

#### 3.4.2 为什么要保留多层特征

车道线检测和一般分类任务不同，它既依赖：

- 高层语义信息
- 也依赖精细空间结构

仅使用最深层特征会导致：

- 分辨率太低
- 细长的车道线轨迹信息损失较多

而只用浅层特征又会导致：

- 语义判别能力不足
- 抗干扰能力较弱

所以该项目选择保留多层特征，并交给 FPN 做融合。

### 3.5 Neck：LaneFPN 如何融合多尺度特征

FPN 实现在 `lane_det/models/fpn.py` 中，实际使用的是 `LaneFPN`，不是普通的 `FPN`。

#### 3.5.1 LaneFPN 的输入与目标

输入是 backbone 的多层特征：

- `[c2, c3, c4, c5]`

目标是：

- 将不同尺度、不同语义强度的特征映射到同一通道维度
- 通过 top-down 融合，让高层语义向高分辨率层传播
- 输出供后续 Anchor 特征池化使用的多尺度特征列表

#### 3.5.2 `use_c2=True` 的含义

当前 `LaneDetector` 初始化 `LaneFPN` 时传入了：

- `use_c2=True`

这意味着 FPN 会从最早的 `c2` 开始参与融合，而不是只用 `c3/c4/c5`。

这样做的直接影响是：

- 能保留更高空间分辨率
- 对细长目标如车道线更友好
- 有利于后续沿轨迹采样时保留局部几何细节

#### 3.5.3 LaneFPN 的内部处理流程

`LaneFPN` 的 forward 逻辑可以拆成 4 步：

1. 选择启用的 backbone 层级
2. 每层先过一个 `1x1 conv`，统一通道数到 `out_channels`
3. 做自顶向下的逐层上采样和相加
4. 每层再过一个 `3x3 conv` 平滑融合结果

这是一套比较标准的 FPN 思路。

更具体地说：

- 最深层特征包含最强语义信息
- 向上采样后与更高分辨率层相加
- 让浅层既保留定位信息，又带有更强语义

#### 3.5.4 当前 FPN 的实际输出

这里有一个非常重要的实现细节：

`LaneFPN` 当前返回的不是单张融合特征图，而是一个多尺度特征列表 `outs`。

在当前配置下，若启用了 `use_c2=True`，则返回：

- `p2`
- `p3`
- `p4`
- `p5`

也就是说，head 看到的不是单层特征，而是一个多尺度金字塔特征集合。

这点很关键，因为后面的 `AnchorFeaturePooler` 会在多个尺度上都沿 Anchor 轨迹采样，再进行融合。

### 3.6 为什么这里不是传统 Dense Head

传统目标检测常见做法是：

- 在特征图每个网格位置上直接预测框参数

但这里的目标是车道线，具有这些特点：

- 细长、连续、强结构化
- 更接近“轨迹”而不是“区域”
- 同一条线跨越较大纵向范围

因此本项目采用了另一种更适合车道线的做法：

- 先定义一组全局 Ray Anchor 轨迹模板
- 然后沿这些轨迹去特征图上取样
- 每条 Anchor 拿到一整条特征序列后再做分类与回归

这使得模型从结构上就对“车道线是一条线而不是一个框”这件事进行了显式建模。

### 3.7 AnchorFeaturePooler：沿 Anchor 轨迹取特征

`AnchorFeaturePooler` 位于 `lane_det/models/anchor_feature_pooler.py`，它是整个模型里最有项目特色的模块之一。

它解决的问题是：

“给定一条 Anchor 轨迹，如何从特征图中准确地取出这条轨迹对应的视觉特征序列。”

#### 3.7.1 为什么需要专门的 Pooler

如果直接把整张特征图 flatten 再交给全连接层，存在几个问题：

- 车道线是细长结构，信息分散在整张图上
- 同一条车道线在不同高度有连续几何关系
- Anchor 已经告诉我们“该去哪里看”

所以更自然的做法是：

- 根据 Anchor 给出的 `(x, y)` 轨迹点
- 在特征图这些位置逐点采样
- 将采样结果组成一条按 `y` 排列的特征序列

这正是 `AnchorFeaturePooler` 在做的事情。

#### 3.7.2 输入与输出

Pooler 的输入包括：

- FPN 输出的单层或多层特征图
- `anchors.anchor_xs`
- `anchors.y_samples`
- 原图尺寸 `img_h, img_w`

输出是：

- `pooled_features: [B, C, NumAnchors, NumY]`

这意味着：

- 对 batch 中每张图
- 对每条 Anchor
- 对 Anchor 的每个 y 采样点
- 都取到了一份长度为 `C` 的局部特征向量

所以这个张量可以理解成：

“每条 Anchor 都有自己的一条特征时间序列，只不过这里的序列轴不是时间，而是沿 y 方向的几何采样位置。”

#### 3.7.3 如何做坐标映射

Anchor 本身是定义在图像坐标系中的：

- `anchor_xs` 是图像尺度上的 x 坐标
- `y_samples` 是图像尺度上的 y 坐标

而 `grid_sample` 要求输入坐标被规范化到 `[-1, 1]` 范围。

因此 Pooler 会先做归一化：

- `x_norm = (x / (img_w - 1)) * 2 - 1`
- `y_norm = (y / (img_h - 1)) * 2 - 1`

然后把每个 Anchor 的 `(x_norm, y_norm)` 组织成采样网格 `grid`。

#### 3.7.4 如何实际采样

Pooler 内部使用的是 `torch.nn.functional.grid_sample`。

其本质是：

- 对每张特征图
- 按给定坐标做双线性插值采样

这样即使 Anchor 点不正好落在离散像素中心，也能获得连续、可导的特征值。

这比手工四舍五入取整更合理，因为：

- 保留了连续几何信息
- 梯度传播更稳定
- 对斜线型车道更友好

#### 3.7.5 当前 Pooler 支持多尺度特征融合

当前实现中，如果输入的是一个特征列表 `[p2, p3, p4, p5]`，Pooler 会：

1. 对每一层都用同一组 Anchor 网格进行采样
2. 得到多个 `[B, C, NumAnchors, NumY]` 张量
3. 在通道维上拼接
4. 再通过一个 `1x1 conv + BN + ReLU` 融合回原通道数

也就是说，当前模型不是只在某一个尺度上取特征，而是：

“每条 Anchor 同时从多层语义、多层分辨率特征图中取证据，再融合成一条更完整的 Anchor 特征序列。”

这对车道线检测是有意义的：

- 浅层提供细节和边缘
- 深层提供结构与语义
- 融合后能增强鲁棒性

### 3.8 AnchorHead：整体职责

`AnchorHead` 位于 `lane_det/models/head.py`，它负责完成：

- Anchor 特征池化
- 分类预测
- 回归预测
- 两阶段 refinement

整体输入是：

- FPN 特征
- Anchors
- 原图大小

整体输出是：

- `cls_logits: [B, NumAnchors]`
- `reg_final: [B, NumAnchors, NumY]`

若开启 `return_aux`，还会返回中间量：

- `reg_stage1`
- `reg_delta_stage2`
- `reg_final`

### 3.9 分类分支是怎么做的

直观上，分类分支在判断：

“当前 Anchor 是否成线、是否值得保留。”

但更准确地说，当前实现不是输出一个纯二值存在概率，而是为每条 Anchor 预测一个质量感知的分类分数，用来衡量它与 GT 的匹配质量；这个分数在推理时会作为筛选和排序依据。

#### 3.9.1 输入特征来自哪里

分类分支并不是先把整条序列压成一个全局向量再分类。当前实现会同时使用两类信息：

- 沿 Anchor 轨迹采样得到的视觉特征 `pooled: [B, C, NumAnchors, NumY]`
- 由当前回归结果构造出的几何上下文 `cls_geo: [B, C, NumAnchors, NumY]`

其中 `cls_geo` 的来源是：

1. 先将 `reg_final.detach()` 变形为 `[B * NumAnchors, 1, NumY]`
2. 经过 `reg_proj`
3. 再经过 `cls_geo_encoder`
4. 最后 reshape 回 `[B, C, NumAnchors, NumY]`

这意味着分类头看到的不只是图像外观特征，还能看到“当前这条 Anchor 被回归头修正成了什么几何形状”。

#### 3.9.2 为什么支持 masked pooling

当前代码支持 `use_masked_pooling=True`。

但 mask 的作用位置不是一开始就把 `pooled` 直接平均成 `pooled_mean`，而是：

- 先把分类输入序列按有效点掩码做逐点屏蔽
- 再在分类头输出的 `NumY` 维上做 masked average

原因是很多 Anchor 尤其 side anchor：

- 并不是在所有 y 位置都落在图像内

如果直接无掩码聚合，就会把越界区域的无效响应混进最终分数。

因此 masked pooling 的实际意义是：

“只在这条 Anchor 真正位于图像内部的那段轨迹上累积分类证据。”

#### 3.9.3 分类头的结构

当前分类头不是一个只看全局均值的简化 MLP，也不是两个 `1x1 Conv1d` 的浅层头。它的结构是：

- `cls_seq_encoder`：两层 `Conv1d + BatchNorm1d + ReLU`
- `cls_pred`：`Conv1d(in_channels, 1, kernel_size=1)`

实际张量流为：

- 先把 `pooled` 与 `cls_geo` 在通道维拼接，得到 `cls_seq_input: [B, 2C, NumAnchors, NumY]`
- 再变形为 `[B * NumAnchors, 2C, NumY]`
- 经过 `cls_seq_encoder + cls_pred`，得到逐点分类响应 `cls_seq_logits: [B * NumAnchors, 1, NumY]`
- 最后沿 `NumY` 维做 masked average 或普通 mean，得到 `cls_logits: [B, NumAnchors]`

#### 3.9.4 输出语义

这里不直接做 sigmoid，是因为训练时会把 logits 交给分类损失函数处理。

配合当前 `QualityFocalLoss`，`cls_logits` 学习的是与 Anchor-GT 匹配质量对齐的分类分数，而不是纯二值存在概率。推理阶段再对它做 sigmoid，作为解码筛选分数使用。

### 3.10 回归分支为什么按“序列建模”

车道线最核心的几何信息，不是一个单独点，而是一整条随 y 变化的轨迹。

所以代码没有把回归做成“每个 y 独立预测”的简单全连接，而是把每条 Anchor 的特征组织成一条序列，再用 `Conv1d` 沿 y 方向建模。

这就是 `LaneSequenceHead` 的用途。

#### 3.10.1 序列输入是如何构造的

池化张量为：

- `pooled: [B, C, NumAnchors, NumY]`

在回归前会重排为：

- `[B * NumAnchors, C, NumY]`

这个变形很关键。它的含义是：

- 把每条 Anchor 都单独拿出来
- 当成一条长度为 `NumY` 的特征序列
- 在通道维上保留卷积表达能力

因此对回归头而言，每个样本单位不再是“整张图”，而是“某张图中的某条 Anchor”。

#### 3.10.2 `LaneSequenceHead` 的结构

`LaneSequenceHead` 是一个沿序列方向做卷积的头部模块，其结构是：

- 若干层 `Conv1d + BatchNorm1d + ReLU`
- 最后接一个 `1x1 Conv1d` 输出单通道结果

输出形状为：

- `[B * NumAnchors, 1, NumY]`

再 reshape 后得到：

- `[B, NumAnchors, NumY]`

这样每条 Anchor 在每个 y 采样位置上都会得到一个回归值。

#### 3.10.3 为什么用 `Conv1d` 很合适

`Conv1d` 沿序列方向建模有几个好处：

- 局部几何变化能被卷积核捕捉
- 邻近 y 位置之间的信息可以共享
- 比完全独立逐点预测更符合车道线连续性
- 参数量比大型 RNN / Transformer 更可控

所以它本质上是在做：

“沿车道线方向的局部几何模式建模。”

### 3.11 回归 stage1：粗偏移预测

第一阶段回归头 `reg_head_stage1` 接收的输入是：

- 每条 Anchor 的特征序列 `[B * NumAnchors, C, NumY]`

它输出的是：

- `reg_stage1: [B, NumAnchors, NumY]`

其语义是：

“对于每条 Anchor，在每个 y 位置上，初步预测应当修正多少横向偏移。”

也就是说，这一阶段的任务是先完成一个粗对齐。

### 3.12 回归 stage2：残差细化

如果配置开启了 `use_refinement=True`，模型还会执行第二阶段回归。

#### 3.12.1 stage2 的输入是什么

第二阶段不是重新从头预测，而是将：

- 原始序列特征
- 第一阶段预测结果

拼接起来，形成新的输入：

- `seq_features_stage2 = cat([seq_features, stage1_pred_detached], dim=1)`

这意味着 stage2 看到了两类信息：

- 图像特征本身
- stage1 已经给出的粗预测

#### 3.12.2 为什么要 `detach`

代码里对 stage1 预测做了 `detach()`，也就是：

- stage2 使用 stage1 的结果作为条件输入
- 但 stage2 的梯度不会反向穿透去直接干扰 stage1 的学习

这样设计的意图是：

- stage1 专注于做粗定位
- stage2 专注于学习残差修正
- 避免两个阶段过度耦合，训练不稳定

#### 3.12.3 stage2 输出什么

stage2 预测的是：

- `reg_delta_stage2`

它不是最终坐标，而是对 stage1 的增量修正。

最后得到：

- `reg_final = reg_stage1 + reg_delta_stage2`

因此两阶段回归的本质可以理解为：

- 第一步把线大致拉到正确位置
- 第二步再修边、细调、补齐局部误差

这和很多 coarse-to-fine 结构的思想是一致的。

### 3.13 一次完整前向传播的张量流动

为了更清晰地理解整个模型，这里按一次 forward 的顺序，把张量流动完整写出来。

#### 3.13.1 输入阶段

输入：

- `images: [B, 3, H, W]`
- `anchors`：包含 `anchor_xs`、`valid_mask`、`x_bottom`、`angles`、`y_samples`

#### 3.13.2 Backbone 阶段

图像进入 `ResNet18`，输出：

- `c2: [B, 64, H/4,  W/4]`
- `c3: [B, 128, H/8, W/8]`
- `c4: [B, 256, H/16, W/16]`
- `c5: [B, 512, H/32, W/32]`

#### 3.13.3 FPN 阶段

`LaneFPN` 对这些特征做 lateral 映射和 top-down 融合，输出：

- `p2`
- `p3`
- `p4`
- `p5`

每一层都具有统一的通道数 `fpn_out`。

#### 3.13.4 Anchor 特征采样阶段

`AnchorFeaturePooler` 根据 Anchor 轨迹在每一层 FPN 特征图上采样，输出多层采样结果，再进行融合，得到：

- `pooled: [B, C, NumAnchors, NumY]`

#### 3.13.5 回归阶段

将 `pooled` 变形为：

- `[B * NumAnchors, C, NumY]`

送入 stage1，得到：

- `reg_stage1: [B, NumAnchors, NumY]`

若启用 refinement，则再拼接 stage1 结果，进入 stage2，得到：

- `reg_delta_stage2: [B, NumAnchors, NumY]`

最终：

- `reg_final: [B, NumAnchors, NumY]`

#### 3.13.6 分类阶段

分类阶段会使用：

- 视觉特征 `pooled`
- 几何上下文 `cls_geo`，它由 `reg_final.detach()` 经过 `reg_proj + cls_geo_encoder` 得到

二者拼接后得到：

- `cls_seq_input: [B, 2C, NumAnchors, NumY]`

再 reshape 为：

- `[B * NumAnchors, 2C, NumY]`

送入 `cls_seq_encoder + cls_pred`，先得到逐点分类响应，再沿 `NumY` 维做 masked average 或 mean，最终输出：

- `cls_logits: [B, NumAnchors]`

#### 3.13.7 输出阶段

模型最终返回：

- `cls_logits`
- `reg_final`

如果训练脚本要求 `return_aux=True`，还会额外返回：

- `reg_stage1`
- `reg_delta_stage2`
- `reg_final`

这些中间量主要用于训练时的多阶段损失监督。

### 3.14 当前模型设计的核心思想

从结构设计上看，这个项目不是把车道线当作普通检测目标，而是显式地把它当作：

- 具有连续轨迹结构的线性目标
- 可以由 Anchor 模板初始化
- 需要沿线方向建模和逐点修正的对象

因此整个模型的设计逻辑是：

1. 用 CNN 提取全图特征
2. 用 FPN 融合语义和空间分辨率
3. 用 Anchor 指定“应当沿哪条轨迹看特征”
4. 用 `grid_sample` 从这些轨迹上提取连续特征
5. 用分类头判断这条 Anchor 是否为真实车道
6. 用序列回归头沿 y 方向输出整条车道的偏移修正
7. 用第二阶段 refinement 对粗预测进行细化

这个结构相比简单的逐像素分割或单点检测，更强调：

- 几何先验
- 轨迹连续性
- 基于候选线模板的结构化回归

### 3.15 步骤 3 的本质

步骤 3 的本质可以概括为一句话：

`模型先把整张图编码成多尺度特征，再沿每条 Anchor 的轨迹提取一整条特征序列，最后分别判断这条 Anchor 是否成线，并回归其整条几何偏移。`

也可以再压缩成更工程化的表达：

`全图卷积编码 + 多尺度融合 + Anchor 轨迹采样 + 序列式分类/回归 + 两阶段细化`

这就是当前项目神经网络主干部分的完整技术实现逻辑。

---

## 步骤 4：损失函数、优化器与训练循环

这一步解决的问题是：

`模型已经能够输出 cls_logits 和 reg_preds 之后，训练阶段如何定义监督目标、如何计算损失、如何反向传播、如何调学习率、如何做验证与保存 checkpoint`

这一步主要对应的文件为：

- `lane_det/losses/focal_loss.py`
- `lane_det/losses/reg_loss.py`
- `lane_det/losses/soft_line_loss.py`
- `tools/train.py`

下面这一步主要详细拆解 `anchor-based` 分支的训练流程。`row/grid-based` 分支对应的 `tools/train_row.py` 会在步骤 7 中单独总结。

如果说步骤 3 解决的是“模型怎么预测”，那么步骤 4 解决的就是：

“模型预测出来之后，怎样告诉它哪里错了，以及如何通过迭代把参数优化到更好的状态。”

### 4.1 训练阶段的整体结构

在 `anchor-based` 分支中，训练流程由 `tools/train.py` 驱动。

从宏观流程看，它的训练逻辑可以概括为：

1. 读取配置文件
2. 构建训练集和验证集
3. 构建模型
4. 构建优化器和学习率调度器
5. 构建损失函数
6. 进入 epoch 循环
7. 每个 epoch 内进行 batch 级训练
8. 每个 epoch 结束后在验证集评估
9. 保存 `epoch_x.pth`、`last.pth`、`best.pth`

所以这一步不是单纯“算一个 loss”，而是整个训练管理过程。

### 4.2 训练脚本先做了哪些准备

#### 4.2.1 读取配置

脚本首先读取 YAML 配置文件，其中最重要的配置块有：

- `dataset`
- `model`
- `loss`
- `train`
- `eval`
- `paths`

这些配置将分别控制：

- 数据来源
- 模型结构
- 损失函数行为
- 训练轮数与学习率
- 验证阶段解码参数
- 输出路径

#### 4.2.2 创建工作目录与日志系统

训练脚本会以当前时间戳创建工作目录，例如：

- `outputs/checkpoints/<timestamp>/`

然后通过 `setup_logger()` 同时建立：

- 控制台日志输出
- 文件日志 `train.log`

这一层的作用是让每次实验都有独立目录，避免不同实验的 checkpoint 和日志互相覆盖。

#### 4.2.3 选择计算设备

训练脚本会自动判断：

- 若 CUDA 可用，则使用 GPU
- 否则退回 CPU

代码中使用：

- `torch.device("cuda" if torch.cuda.is_available() else "cpu")`

### 4.3 训练前的数据准备

训练脚本在正式训练前，会构建：

- `train_dataset`
- `val_dataset`

并用 `DataLoader` 包装。

#### 4.3.1 训练集 DataLoader

训练集加载器的关键配置为：

- `batch_size = cfg["train"]["batch_size"]`
- `shuffle = True`
- `num_workers = 0`
- `drop_last = True`
- `collate_fn = collate_fn`

这里 `drop_last=True` 的意思是：

- 若最后一个 batch 样本数不满，则直接丢弃

这样做通常是为了保持训练阶段 batch 形状稳定。

#### 4.3.2 验证集 DataLoader

验证集加载器使用：

- `shuffle = False`
- `drop_last = False`
- `batch_size = cfg.get("test", {}).get("batch_size", 1)`

验证不打乱顺序、也不丢尾 batch，这更符合评估逻辑。

#### 4.3.3 `collate_fn` 在训练中做了什么

`tools/train.py` 中的 `collate_fn` 负责把多个样本组织成一个 batch。

它主要做了这些事情：

1. 堆叠图像为 `images: [B, 3, H, W]`
2. 堆叠分类标签为 `cls_targets: [B, NumAnchors]`
3. 堆叠回归标签为 `offset_labels: [B, NumAnchors, NumY]`
4. 堆叠回归掩码为 `offset_masks: [B, NumAnchors, NumY]`
5. 从 batch 的第一个样本重建一个共享的 `AnchorSet`

这里有一个工程上的默认前提：

- 同一个 batch 内所有样本使用的是相同 Anchor 配置
- 因此可以共享同一套 AnchorSet

### 4.4 训练前先统计匹配质量

训练脚本在正式训练前还会调用：

- `log_pretrain_match_stats(train_dataset, logger, max_samples=args.pre_stat_max)`

这个函数会遍历前若干个样本，统计：

- 正样本数量
- 负样本数量
- 忽略样本数量
- 每条 GT 的最大 IoU 分布
- 每条 GT 匹配到的正样本数分布
- 忽略样本的来源原因统计

这些统计的作用非常重要，因为这个项目是 Ray-Anchor-based 结构，训练能否稳定，很大程度取决于标签分配是否合理。

换句话说，这一步相当于：

“先检查 Anchor 和 GT 的匹配生态是否正常，再决定是否值得开始训练。”

### 4.5 模型、优化器和调度器是如何构建的

#### 4.5.1 模型构建

训练脚本会实例化：

- `model = LaneDetector(cfg)`

然后将其移动到指定设备。

#### 4.5.2 优化器

当前优化器使用的是：

- `torch.optim.Adam`

核心参数来自配置：

- `lr = cfg["train"]["lr"]`
- `weight_decay = cfg["train"]["weight_decay"]`

也就是说，当前项目并没有采用 SGD，而是使用 Adam 做一阶自适应优化。

从工程上看，这通常意味着：

- 对初期调参更友好
- 对不同分支和不同参数尺度更稳定
- 能更快把项目跑通并进入可用状态

#### 4.5.3 学习率调度器

当前学习率调度器是：

- `MultiStepLR`

相关配置包括：

- `lr_milestones`
- `lr_gamma`

它的行为是：

- 当 epoch 走到若干里程碑时
- 将当前学习率乘以一个衰减系数 `gamma`

这是一种典型的分段式学习率衰减策略。

### 4.6 为什么要使用梯度累计

训练脚本里固定设置了：

- `accumulation_steps = 2`

它的含义是：

- 每次 forward 和 backward 后先不立即更新参数
- 连续积累 2 个 mini-batch 的梯度
- 再执行一次 `optimizer.step()`

这相当于在显存不增加太多的情况下，模拟更大的有效 batch size。

当前这么做通常有两个原因：

- 单卡显存有限
- 车道线任务中较大的 batch 有时更利于统计稳定性

### 4.7 当前训练使用了哪些损失函数

当前项目训练时可能涉及 3 类损失：

1. 分类损失：`QualityFocalLoss`
2. 回归损失：`RegLoss`，内部为 `Smooth L1`
3. 线形相似性损失：`SoftLineOverlapLoss`

其中前两者是主损失，第三项是可选增强项。

### 4.8 分类损失 `QualityFocalLoss`

分类损失定义在 `lane_det/losses/focal_loss.py`。

#### 4.8.1 输入与目标

输入是：

- `inputs`：模型输出的 `cls_logits`
- `targets`：batch 中的 `cls_targets`

其中目标值取值为：

- `(0, 1]`：正样本质量分数，由匹配到的 `Line IoU` 经过 `_map_iou_to_cls_target(...)` 映射得到
- `0`：负样本
- `-1`：忽略样本

这意味着当前项目的分类监督已经不是“纯二值分类”，而是“质量感知分类”：

- 分数越高，表示该 Anchor 与 GT 的匹配质量越好
- 分数越低但仍大于 0，表示它是正样本，但质量较弱
- 负样本仍然使用 `0`

#### 4.8.2 忽略样本如何处理

损失函数开头会先做：

- `valid_mask = (targets >= 0)`

也就是说，所有 `-1` 的忽略样本会直接被过滤掉，不参与分类损失。

这很重要，因为灰区样本本来就是“不确定样本”，不应该强行当正或负来优化。

#### 4.8.3 核心计算逻辑

`QualityFocalLoss` 内部先计算：

- `binary_cross_entropy_with_logits`

然后分别对正负样本做不同的重加权：

- 正样本权重与目标值本身相关，目标越高，正样本损失权重越大
- 负样本使用 `alpha * sigmoid(inputs)^gamma` 的 focal 权重

它的核心思想可以理解为：

- 用 BCEWithLogits 建立基础分类监督
- 让正样本目标值携带匹配质量信息
- 用 focal 方式压低大量简单负样本的影响

#### 4.8.4 当前实现的一个重要细节

当前实现不是简单对全部样本一起求均值，而是：

1. 把正样本损失单独求和后除以正样本数
2. 把负样本损失单独求和后除以负样本数
3. 最后再将二者相加

这一步的实际作用是：

- 防止负样本过多时彻底淹没正样本梯度
- 让稀少的正 Anchor 仍然能有足够训练信号

这对于 Ray-Anchor-based 车道线检测尤其关键，因为候选 Anchor 往往大量为负样本。

### 4.9 回归损失 `RegLoss`

回归损失定义在 `lane_det/losses/reg_loss.py`，内部采用的是：

- `smooth_l1_loss`

#### 4.9.1 输入是什么

回归损失的输入为：

- `inputs`：模型预测偏移
- `targets`：`offset_label`
- `mask`：`offset_valid_mask`

从语义上看，它比较的是：

“模型预测的每个 Anchor 各 y 位置横向偏移”

和

“GT 相对 Anchor 的真实横向偏移”

之间的差距。

#### 4.9.2 为什么必须乘 `mask`

车道线回归并不是所有位置都有效，因此损失函数里会先做：

- `loss = loss * mask`

然后只在有效位置上求平均。

这一步是必须的，因为：

- 有些 y 点 GT 本来就无效
- 有些 Anchor 在某些 y 上超出图像
- 有些匹配虽然成立，但并不是每个点都适合回归监督

所以回归损失本质上是：

“只在合法的、可监督的车道采样点上计算。”

#### 4.9.3 当前回归损失的平均方式

`RegLoss` 在 `reduction='mean'` 时，不是直接按张量总元素数平均，而是：

- 用 `mask.sum()` 作为有效元素个数

这意味着监督强度只由真实有效点决定，而不会被无效补零区域稀释。

### 4.10 线级相似性损失 `SoftLineOverlapLoss`

这个损失定义在 `lane_det/losses/soft_line_loss.py`，它不是必开项，而是由配置控制：

- `use_line_overlap_loss`

#### 4.10.1 这个损失想解决什么问题

普通逐点回归损失虽然能优化每个点的偏移误差，但它更像“点对点监督”。

而车道线本身是一整条连续曲线，因此作者又加了一个线级相似性损失，目的是：

- 从整条线的层面约束预测结果和 GT 的重合程度

#### 4.10.2 计算思路

这个损失会：

1. 计算预测偏移和目标偏移的逐点平方距离
2. 用高斯函数把距离转成软相似度
3. 对每条线在所有有效点上取平均相似度
4. 最终定义损失为 `1 - similarity`

所以它不是一个硬性的几何重叠判定，而是一个连续可导的“越接近越奖励”的软约束。

#### 4.10.3 为什么要设 `min_valid_points`

如果某条线有效点太少，那么从“整条线重合度”角度去评价它是没有意义的。

所以这个损失还要求：

- 每条线至少有一定数量的有效点

否则该样本不参与 line loss 计算。

### 4.11 refinement 开启时，损失是怎样组合的

当配置中：

- `use_refinement=True`

时，模型会返回：

- `reg_stage1`
- `reg_final`

训练脚本会把这两个阶段都纳入监督。

#### 4.11.1 stage1 损失

第一阶段回归损失为：

- `reg1_loss = RegLoss(reg_stage1, offset_labels, offset_masks)`

它监督的是：

- 粗回归是否已经接近 GT

#### 4.11.2 stage2 / final 损失

第二阶段回归损失为：

- `reg2_loss = RegLoss(reg_stage2, offset_labels, offset_masks)`

这里代码里 `reg_stage2` 实际上指的是：

- `reg_final`

也就是 refinement 之后的最终预测。

因此 stage2 的监督目标不是中间残差本身，而是最终修正结果是否逼近 GT。

#### 4.11.3 两阶段回归总损失

训练脚本会根据配置中的权重：

- `reg_loss_stage1_weight`
- `reg_loss_stage2_weight`

将两者合成为：

- `reg_loss = w1 * reg1_loss + w2 * reg2_loss`

这说明当前训练目标不是只关心最终输出，而是希望：

- 第一阶段有基本可用的粗预测
- 第二阶段在此基础上进一步细化

### 4.12 总损失是如何组合的

主损失的组合方式为：

- `loss = cls_weight * cls_loss + reg_weight * reg_loss`

其中：

- `cls_weight` 来自配置
- `reg_weight` 来自配置

也就是说，分类和回归在总目标中并不是默认等权，而是可以通过实验手动调平衡。

### 4.13 line loss 是如何叠加的

如果开启了 `use_line_overlap_loss`，训练脚本还会额外计算：

- `line1_loss`
- 若启用 refinement，还会计算 `line2_loss`

然后按和回归类似的阶段权重进行合成：

- `line_loss = w1 * line1_loss + w2 * line2_loss`

最后再乘上：

- `line_loss_weight`

叠加到总损失：

- `loss += line_loss_weight * line_loss`

因此完整总损失可以写成：

`total_loss = cls_weight * cls_loss + reg_weight * reg_loss + line_loss_weight * line_loss`

其中最后一项是可选的。

### 4.14 为什么 line loss 还有动态 sigma

当前训练脚本里有一段特殊逻辑：

- `line_sigma`
- `line_sigma_refined`
- `line_sigma_step`

其行为是：

- 训练早期使用一个较大的 `sigma`
- 到了指定 epoch 后，将 `sigma` 调整为更小的值

从损失形状上看，这样做意味着：

- 早期允许较宽松的“相似”
- 后期要求预测与 GT 更精确地重合

这是一种典型的 coarse-to-fine 训练思路，和 refinement 的思想是一致的。

### 4.15 一个 batch 内的训练流程

下面按训练脚本中的实际逻辑，梳理一次 batch 训练时发生的全部步骤。

#### 4.15.1 取出 batch 数据

从 DataLoader 取出一个 batch 后，会获得：

- `images`
- `cls_targets`
- `offset_labels`
- `offset_masks`
- `anchors`

然后把张量移动到目标设备。

#### 4.15.2 前向传播

若启用 refinement：

- `cls_logits, reg_preds, aux = model(images, anchors, return_aux=True)`

否则：

- `cls_logits, reg_preds = model(images, anchors)`

其中：

- `reg_stage1 = aux["reg_stage1"]`
- `reg_stage2 = aux["reg_final"]`

若未开启 refinement，则：

- `reg_stage1 = reg_preds`
- `reg_stage2 = reg_preds`

#### 4.15.3 计算分类与回归损失

当前 batch 会先算：

- `cls_loss`
- `reg1_loss`
- 若有 refinement，再算 `reg2_loss`

然后合成：

- `reg_loss`
- `total_loss`

#### 4.15.4 可选计算 line loss

若配置启用 line loss，还会再基于：

- `reg_stage1`
- `reg_stage2`
- `offset_labels`
- `offset_masks`

计算线级重合损失并叠加。

#### 4.15.5 反向传播与梯度累计

脚本中真正执行反向传播的是：

- `(loss / accumulation_steps).backward()`

这里先除以 `accumulation_steps` 是为了保证：

- 累积多个小 batch 梯度之后，总梯度尺度与一个大 batch 近似一致

然后在满足条件时执行：

- `optimizer.step()`
- `optimizer.zero_grad()`

#### 4.15.6 日志记录

每隔 10 个 step，训练脚本会打印一次当前：

- `Loss`
- `Cls`
- `Reg`
- `Reg1`
- `Reg2`
- 若启用还包括 `Line`

这使得训练过程中可以持续观察：

- 分类是否稳定
- 回归是否下降
- 两阶段回归谁更难
- line loss 是否过大或失效

### 4.16 一个 epoch 结束后会做什么

每个 epoch 训练完成后，脚本会进行几件固定工作。

#### 4.16.1 统计平均损失

脚本会汇总本 epoch 中的：

- `avg_loss`
- `avg_cls`
- `avg_reg`
- `avg_reg1`
- `avg_reg2`
- `avg_line`

并写入日志。

这一步提供的是 epoch 级趋势，和 step 级即时日志互补。

#### 4.16.2 验证集评估

训练脚本随后会调用：

- `validate(model, val_loader, val_dataset.samples, cfg, device)`

也就是说，当前项目不是只看训练损失，还会在每个 epoch 之后立即跑一次完整验证。

这一点很重要，因为车道线任务里：

- 训练损失下降

并不总是等价于：

- TuSimple 指标上升

所以必须用验证集上的 `Accuracy / FP / FN` 作为更可靠的判断依据。

#### 4.16.3 学习率调度器更新

每个 epoch 验证结束后会执行：

- `scheduler.step()`

然后记录当前学习率。

这说明学习率调度是按 epoch 级更新的，而不是按 step 级更新。

### 4.17 训练过程里的验证逻辑

虽然“验证、推理、评估”会在后续步骤中单独展开，但在训练上下文里，验证函数的作用需要先说明。

`validate()` 做的事情是：

1. 将模型切到 `eval()`
2. 构建 `LaneDecoder`
3. 对验证集逐 batch 前向推理
4. 把输出解码成车道线
5. 用 `TuSimpleConverter` 转成评估格式
6. 用 `LaneEval.bench()` 逐样本计算：
   - `Accuracy`
   - `FP`
   - `FN`
7. 对整个验证集求平均

也就是说，这里的验证已经不是“只算一个 val loss”，而是直接使用接近最终测试阶段的完整流程来评价模型。

这是一个很强的工程信号，说明作者更重视：

- 真实端到端指标

而不是单纯的中间损失数值。

### 4.18 checkpoint 是如何保存的

训练脚本在每个 epoch 结束后会保存三类权重文件。

#### 4.18.1 每轮快照

每个 epoch 都会保存：

- `epoch_{n}.pth`

这用于保留完整训练过程中的阶段性快照，便于回溯和对比。

#### 4.18.2 最新权重

每个 epoch 还会覆盖保存：

- `last.pth`

它表示当前最新训练进度，最适合训练中断后恢复或继续实验。

#### 4.18.3 最优权重

若当前 epoch 的验证集 `Accuracy` 优于历史最好值，则保存：

- `best.pth`

同时记录：

- `best_acc`

这意味着当前项目的“最佳模型”选择标准是：

- 验证集上的 `Accuracy`

而不是最小训练损失。

### 4.19 resume 机制是怎么工作的

训练脚本支持通过：

- `--resume <checkpoint>`

恢复训练。

恢复时会加载：

- `model.state_dict()`
- `optimizer.state_dict()`
- `epoch`
- `best_acc`

并通过循环调用 `scheduler.step()` 同步学习率调度状态。

这样可以较完整地恢复训练现场，而不只是恢复模型参数。

### 4.20 当前训练策略体现出的设计取向

从整个训练实现可以看出，这个项目的训练设计有几个非常明确的偏向。

#### 4.20.1 不是只优化点误差，而是兼顾线形

因为除了 `SmoothL1` 外，还额外引入了 `SoftLineOverlapLoss`。

这说明作者不满足于“每个点误差变小”，而希望：

- 整条车道线形状也更一致

#### 4.20.2 不是只监督最终输出，而是监督中间阶段

refinement 开启时，stage1 和 final 都会被监督。

这说明当前设计希望：

- coarse 阶段先站稳
- refine 阶段再做精修

#### 4.20.3 不是只看 loss，而是每轮都看最终任务指标

训练结束每个 epoch 都直接跑验证并计算 TuSimple 指标，这说明项目更关注：

- 真实检测效果

而不是只看内部 surrogate loss。

#### 4.20.4 不是只求能跑，而是带有实验管理意识

这一点体现在：

- 独立工作目录
- 日志文件
- 训练前匹配统计
- `epoch/last/best` 多种 checkpoint
- resume 能力

说明当前训练流程已经具备较完整的实验闭环意识。

### 4.21 步骤 4 的本质

步骤 4 的本质可以概括为：

`训练脚本利用分类损失、逐点回归损失和可选的线级相似性损失，对 Ray-Anchor-based 车道线检测模型进行联合优化，并在每个 epoch 后通过真实解码和 TuSimple 指标验证模型效果，再保存阶段性和最优权重。`

如果再压缩成更工程化的一句话，就是：

`多目标联合监督 + Adam 优化 + 分段学习率衰减 + 梯度累计 + 每轮验证 + 多版本 checkpoint 管理`

这就是当前项目训练部分的完整技术逻辑。

---

## 步骤 5：验证、推理、解码、NMS 与评估

这一步解决的问题是：

`模型已经训练好并且能够输出 Anchor 分类分数和偏移量之后，如何把这些原始张量还原成真正的车道线，如何去重，如何平滑，如何导出为 TuSimple 格式，以及如何最终计算 Accuracy / FP / FN`

这一步主要对应的文件有：

- `lane_det/postprocess/decoder.py`
- `lane_det/metrics/tusimple_converter.py`
- `tools/infer.py`
- `tools/evaluate.py`
- `tools/visualize.py`

下面这一步主要详细拆解 `anchor-based` 分支的解码与后处理链路。`row/grid-based` 分支的推理和解码会在步骤 7 中单独总结。

如果说步骤 4 的终点是：

- 获得一个可用的 checkpoint

那么步骤 5 的起点就是：

- 给定图像和 checkpoint，输出最终可评估、可可视化的车道线结果

这一步是整个项目从“训练内部张量”走向“可交付结果文件”的关键桥梁。

### 5.1 为什么必须有解码阶段

模型在前向传播结束后，输出的是：

- `cls_logits: [B, NumAnchors]`
- `reg_preds: [B, NumAnchors, NumY]`

这些结果本身还不是最终车道线，它们只是：

- 每条 Anchor 是否存在车道的分类分数
- 每条 Anchor 在各 y 位置上应修正多少偏移

因此在真正得到“车道线坐标”之前，还必须经历后处理链路：

1. 将 logits 做 sigmoid 分数化
2. 筛掉低分 Anchor
3. 将 `anchor_x + offset` 还原成真实车道坐标
4. 清理无效点
5. 对重复车道做 NMS
6. 对最终曲线做平滑
7. 转成 TuSimple 评估格式

这整段工作由 `LaneDecoder + TuSimpleConverter + LaneEval` 共同完成。

### 5.2 验证和独立推理其实共用一条主链路

当前项目中，训练时验证和离线推理虽然入口不同，但核心流程高度一致：

#### 5.2.1 训练中的验证

`tools/train.py` 中的 `validate()` 会：

1. 用模型前向得到 `cls_logits` 和 `reg_preds`
2. 用 `LaneDecoder` 解码
3. 用 `TuSimpleConverter` 转换格式
4. 用 `LaneEval.bench()` 计算指标

#### 5.2.2 独立推理脚本

`tools/infer.py` 会：

1. 加载 checkpoint
2. 对指定 split 做前向推理
3. 用 `LaneDecoder` 解码
4. 用 `TuSimpleConverter` 导出 `pred.json`

#### 5.2.3 二者的关系

可以把它理解成：

- `validate()` 是“训练中的内置推理评估版”
- `infer.py` 是“独立运行的完整导出版”

底层核心逻辑几乎一样，区别主要在于：

- 训练验证会当场计算指标
- 独立推理主要负责保存结果文件

### 5.3 独立推理脚本 `tools/infer.py` 是怎么工作的

`infer.py` 是项目对外最重要的推理入口之一。

它的大致流程是：

1. 读取配置文件
2. 确定运行 split
3. 构建数据集和 DataLoader
4. 构建模型并加载 checkpoint
5. 对每个 batch 做前向传播
6. 用 `LaneDecoder` 解码模型输出
7. 用 `TuSimpleConverter` 转成提交格式
8. 将所有样本保存为 jsonl 形式的 `pred.json`

#### 5.3.1 split 是如何决定的

`infer.py` 支持：

- `train`
- `val`
- `test`

若命令行没有手动指定 `--split`，脚本会根据配置里的 `list_file` 自动推断：

- 若像 `test_label.json`，则默认走 `test`
- 否则通常走 `val`

#### 5.3.2 为什么推理时还要用 Dataset

虽然推理阶段不需要训练标签，但依然使用 `TuSimpleDataset`，原因是 Dataset 还承担了这些职责：

- 读取图像
- 按配置 resize / normalize
- 生成当前图像尺寸下的 AnchorSet
- 保留 meta 信息，例如 `raw_file` 和原始 `h_samples`

这说明当前数据层不仅是训练服务的，也是整个推理流程的统一入口。

### 5.4 `LaneDecoder` 的核心职责

真正把模型输出恢复成车道线的是：

- `lane_det/postprocess/decoder.py`

`LaneDecoder` 可以看成整个后处理系统的核心模块。

它要解决的问题是：

“从所有 Anchor 的分类和回归结果中，恢复出一组干净、去重、平滑的最终车道线候选。”

### 5.5 解码的第一步：对分类 logits 做分数化和筛选

在 `decode()` 开头，`LaneDecoder` 会先对：

- `cls_logits`

做：

- `scores = sigmoid(cls_logits)`

因为训练时分类头输出的是 logits，推理时要先做 sigmoid，把它转成与匹配质量对齐的筛选分数。

接着会按阈值：

- `score_thr`

进行筛选，只保留高于阈值的 Anchor。

这一层的作用是：

- 直接过滤掉大多数明显不成线的候选 Anchor
- 降低后续 NMS 和转换的计算量
- 避免大量低质量假阳性进入最终结果

如果某张图经过这一轮筛选后没有任何 Anchor 留下，则该图直接输出空车道列表。

### 5.6 解码的第二步：从偏移量恢复真实车道坐标

通过分数筛选后，解码器会取出这些 Anchor 的：

- `keep_anchor_xs`
- `keep_reg`

然后按公式恢复真实预测坐标：

- `pred_xs = keep_anchor_xs + keep_reg`

这一步非常关键，因为模型学习的从头到尾都不是绝对坐标，而是：

- “相对于 Anchor 的偏移”

所以解码时必须把 Anchor 模板加回来，才能得到真正的车道线横坐标。

### 5.7 解码的第三步：确定哪些点最终有效

恢复出 `pred_xs` 之后，解码器还会做一层有效性判断：

- 若预测点 `x` 超出图像宽度范围，则记为无效
- 再与原始 Anchor 的 `valid_mask` 做交集

得到：

- `final_mask`

也就是说，一个点最终是否有效，必须同时满足两件事：

1. 该点对应的 Anchor 位置本身在图像范围内
2. 加上偏移后的预测点也仍在图像范围内

这一步的意义是防止：

- 模型把某些点预测到图像外
- 或本来就位于 Anchor 无效区域的点被误当作真实车道点

### 5.8 解码的第四步：生成候选车道列表

对每条通过阈值的 Anchor，解码器会构造一个候选 lane dict，大致包含：

- `score`
- `x_list`
- `valid_mask`
- `y_samples`
- `length`

其中 `length` 是当前车道的有效点数量。

如果某条候选线的有效点太少，例如：

- 少于 `min_valid_points` 个点（当前默认值为 4）

就直接丢弃，因为这样的“车道线”几乎没有几何意义，也无法可靠评估和可视化。

### 5.9 为什么候选线现在只按 `score` 排序

解码器在做 NMS 之前，会对候选车道进行排序。当前实现不再混入长度项，而是直接按：

- `score`

做降序排序。

这样设计的原因是：

- 当前分类分数本身就是质量感知分数
- decode 阶段的保留优先级需要和训练时的分类语义保持一致
- 车道过短的问题已经由 `min_valid_points` 过滤和后续几何 NMS 负责约束

因此这里的策略是：

- 先按分类质量分数决定候选优先级
- 再通过几何约束去掉重复或不稳定的候选

### 5.10 车道线 NMS 为什么不能照搬检测框 NMS

车道线不是矩形框，因此常见 box NMS 不适用。

两个车道候选是否重复，应该比较的是：

- 它们在共同采样区间内是否基本重合

因此当前项目实现了自己的 lane-level NMS。

### 5.11 `_is_duplicate_lane()` 是如何判断重复车道的

重复判断逻辑位于 `LaneDecoder._is_duplicate_lane()`。

它不是简单比较整体均值距离，而是分几层条件判断。

#### 5.11.1 先看是否有足够共同有效点

两个候选 lane 首先必须满足：

- 在足够多的 `y` 位置上同时有效

也就是共同有效点数要达到：

- `nms_min_common_points`

若共同点太少，就直接认为没法可靠比较，不判为重复。

#### 5.11.2 再看共同区间占比是否够高

即使共同点数够，也不能说明它们真的是同一条线。

所以还会进一步检查：

- `overlap_ratio = common_len / min(len_a, len_b)`

只有当 overlap ratio 高于：

- `nms_overlap_ratio_thr`

时，才继续判重。

这一步的作用是防止：

- 只有一小段重叠，但大部分路段并不一致的两条车道被误合并

#### 5.11.3 再看整体平均距离

在共同区间上，解码器会计算：

- 两条线对应点的横向绝对距离

并取平均值：

- `mean_dist`

若平均距离大于：

- `nms_thr`

则认为这两条线不是重复车道。

#### 5.11.4 还要单独检查顶部和底部距离

当前实现还有一个更细的约束：

- 把共同区间切成前 1/3 和后 1/3
- 分别计算 `top_dist` 和 `bottom_dist`

并要求：

- `top_dist < nms_thr * nms_top_dist_ratio`
- `bottom_dist < nms_thr * nms_bottom_dist_ratio`

这一步的作用非常关键，因为有些候选线：

- 中间某段可能接近
- 但顶部或底部已经明显分叉

如果只看整体平均距离，容易错误合并。

因此当前 lane NMS 的核心思想不是粗暴去重，而是：

- 只有在“整体接近且上下段都接近”的情况下才当作重复

### 5.12 NMS 的实际执行过程

当前 NMS 的执行流程是典型的贪心式流程：

1. 按优先级排序候选车道
2. 取出当前最优的一条作为保留项
3. 将与它判为重复的候选全部删掉
4. 对剩余候选重复该过程

如果配置中：

- `nms_thr <= 0`

则可以视作关闭 NMS，直接保留所有候选线。

### 5.13 为什么还要做 polyfit 平滑

NMS 之后，解码器还会对每条保留的车道进行平滑，这部分由：

- `use_polyfit`

控制。

其目的主要是解决：

- 逐点回归带来的抖动
- 局部不平滑
- 某些线起始段漂移

也就是说，模型即使整体预测正确，逐 y 位置的偏移也可能有锯齿感，需要做轻量曲线拟合来提升几何连续性。

### 5.14 polyfit 是如何做的

对于每条保留车道：

1. 先取有效点的 `(y_valid, x_valid)`
2. 根据有效点数量动态决定拟合阶数
3. 用 `np.polyfit` 拟合：
   - `x = f(y)`
4. 再用拟合得到的多项式重算这些有效点处的 `x`

当前拟合阶数规则大致是：

- 点数很少：一次拟合
- 点数中等：二次拟合
- 点数较多：三次拟合

这样做的原因是：

- 点太少时用高阶拟合容易严重过拟合
- 点多时允许更高灵活度去逼近弯曲车道

#### 5.14.1 为什么不是在 Converter 里再拟合一次

`TuSimpleConverter` 里有明确注释，说明：

- 这里已经移除了二次 polyfit
- 依赖 Decoder 里的 polyfit 即可

这说明当前项目有意识避免：

- 在 decoder 和 converter 两个阶段重复 reshape 车道线

否则容易把本来已经合理的预测再次扭曲。

### 5.15 `LaneDecoder` 的最终输出是什么

经过：

- 分数筛选
- 坐标恢复
- 有效性修正
- 候选构造
- lane NMS
- polyfit 平滑

之后，`LaneDecoder.decode()` 最终输出的是：

- 一个 batch 对应的车道线列表

对 batch 中每张图而言，其结果是一个 `lanes` 列表，每条 lane 至少包含：

- `x_list`
- `valid_mask`
- `y_samples`
- `score`

这已经是“模型内部表达”向“几何车道线表达”的完成态。

### 5.16 为什么还需要 `TuSimpleConverter`

即使经过 `LaneDecoder`，当前结果仍然是项目内部格式，还不能直接用于官方评估。

TuSimple 官方评估所要求的每条样本格式是：

- `raw_file`
- `lanes`
- `h_samples`
- `run_time`

其中 `lanes` 还必须是：

- 在指定 `h_samples` 位置上的横坐标列表
- 无效点使用 `-2`

所以必须经过一个专门的格式转换过程，这就是 `TuSimpleConverter` 的职责。

### 5.17 `TuSimpleConverter` 如何做格式转换

`TuSimpleConverter` 位于 `lane_det/metrics/tusimple_converter.py`。

其处理流程可以拆成以下几步。

#### 5.17.1 先确定目标 `h_samples`

Converter 可以使用：

- 默认 TuSimple `h_samples`

也可以使用外部传入的：

- `target_h_samples`

当前项目在推理导出时，优先使用当前样本原始标注中的 `original_h_samples`。

这样做的好处是：

- 预测结果和 GT 的采样位置一一对齐
- 避免由于采样点定义不一致带来的评估偏差

#### 5.17.2 将模型坐标恢复到原图尺度

因为模型是在 resize 后的输入图像上做预测的，所以 Converter 会按：

- `scale_x = ori_w / img_w`
- `scale_y = ori_h / img_h`

把解码后的 `x` 和 `y` 重新映射回原图坐标系。

这一步很关键，因为 TuSimple 的 GT 和评估标准都定义在原始图像尺度上。

#### 5.17.3 对目标 `h_samples` 做插值

Converter 会将每条 lane 的有效点：

- 按 `y` 排序
- 然后在目标 `h_samples` 上做插值

这样可以把模型内部的预测结果整理成：

- 与 TuSimple 标准完全一致的一条固定长度车道线

#### 5.17.4 如何处理无效点

若某个目标 `h_sample` 超出了当前预测车道的有效 `y` 范围，或插值得到的 `x` 超出图像宽度范围，则：

- 该位置输出 `-2`

这符合 TuSimple 的无效点约定。

#### 5.17.5 为什么还要检查有效点数量

若某条 lane 最终只剩极少数有效点，Converter 会直接丢弃该 lane。

原因很简单：

- 过短、过碎的预测线对评估没有价值
- 还可能引入额外 FP

### 5.18 `pred.json` 最终长什么样

通过 `TuSimpleConverter.save_json()` 保存后，结果文件是 jsonl 格式，每行对应一张图像，例如每行包含：

- `raw_file`
- `lanes`
- `h_samples`
- `run_time`

其中：

- `lanes` 是一个二维列表
- 每一条子列表是一条车道线在所有目标 `h_samples` 上的 `x` 值
- 无效点是 `-2`

这就是当前项目最终可评估、可提交的标准结果格式。

### 5.19 `tools/evaluate.py` 如何计算 TuSimple 指标

`tools/evaluate.py` 提供的是 TuSimple 指标的 Python 实现。

核心入口是：

- `LaneEval.bench_one_submit(pred_file, gt_file)`

它的逻辑大致如下：

1. 读取预测文件每一行 JSON
2. 读取 GT 文件每一行 JSON
3. 按 `raw_file` 建立 GT 映射
4. 对每个预测样本找到对应 GT
5. 调用 `LaneEval.bench()` 计算单样本指标
6. 对所有样本求平均
7. 输出：
   - `Accuracy`
   - `FP`
   - `FN`

### 5.20 单样本指标 `LaneEval.bench()` 是怎么计算的

这一部分是理解最终评估标准的关键。

#### 5.20.1 为什么阈值和车道角度有关

评估时，GT 的每条车道会先拟合一个角度，再根据角度确定像素误差阈值：

- `thresh = pixel_thresh / cos(angle)`

这么做的原因是：

- 倾斜更大的车道，在图像中相同横向误差对应的几何偏差意义不同

因此 TuSimple 并不是对所有车道一刀切使用固定阈值，而是做了角度归一。

#### 5.20.2 单条车道准确率怎么定义

对于某条 GT 车道，会把预测车道逐条拿来比较，在所有 `h_samples` 上统计：

- 横向误差小于阈值的点占比

这就是 `line_accuracy()`。

然后从所有预测车道里取与这条 GT 最匹配的那一条，作为该 GT 的最高匹配精度。

#### 5.20.3 FN 如何定义

若某条 GT 的最佳匹配精度仍低于：

- `pt_thresh = 0.85`

则这条 GT 被视为漏检，对应：

- `FN += 1`

也就是说，在 TuSimple 评估里，不是只要大致靠近就算命中，而是要求大部分采样点都足够准确。

#### 5.20.4 FP 如何定义

对于当前图像：

- 若预测的车道数比成功匹配的 GT 数更多

多出来的部分记作：

- `FP = len(pred) - matched`

因此过多重复线、碎线、伪线都会直接提升 FP。

#### 5.20.5 Accuracy 如何汇总

脚本会把每条 GT 的最佳 line accuracy 累加，再按 TuSimple 规则归一化得到当前图像的 Accuracy。

同时还实现了 TuSimple 中一个特殊规则：

- 若 GT 车道数大于 4，会做额外修正

这说明当前评估实现已经尽量贴近官方逻辑。

### 5.21 路径匹配问题是怎么处理的

`bench_one_submit()` 中还做了一个实际工程很有用的兼容处理：

- 若 GT 中的 `raw_file` 带有 `test_set/` 前缀
- 则同时建立去掉此前缀的映射

这样做的原因是：

- 不同数据准备方式下，预测文件里的路径可能和 GT 略有差异

这个兼容逻辑避免了仅因路径前缀不同而导致“找不到匹配样本”的评估失败。

### 5.22 推理结果可视化是怎么做的

`tools/visualize.py` 的职责是：

- 读取 `pred.json`
- 找到对应原图
- 把预测车道画回图像
- 保存可视化结果

其绘制逻辑比较直接：

- 对每条 lane 遍历所有 `h_samples`
- 取有效 `x`
- 将相邻点用直线连接起来

虽然这个模块不参与训练和评估本身，但它在工程上非常重要，因为它能帮助检查：

- 预测是否偏移
- 是否有重复线
- 是否有断裂
- 是否有明显 FP / FN

### 5.23 从 checkpoint 到最终指标的一次完整链路

为了把步骤 5 串完整，这里把推理评估链路从头到尾再总结一次。

#### 5.23.1 输入

输入包括：

- 一份配置文件
- 一个训练好的 checkpoint
- 一个待评估的数据集 split

#### 5.23.2 模型前向

对每张图像：

- Dataset 完成读取、预处理和 Ray Anchor 生成
- 模型输出 `cls_logits` 和 `reg_preds`

#### 5.23.3 解码

`LaneDecoder` 完成：

- sigmoid 分数化
- score threshold 筛选
- `anchor + offset` 恢复坐标
- 有效点裁剪
- 候选车道构造
- lane NMS 去重
- polyfit 平滑

#### 5.23.4 格式转换

`TuSimpleConverter` 完成：

- 恢复到原图尺度
- 按目标 `h_samples` 插值
- 用 `-2` 填充无效点
- 组织成 TuSimple 标准 jsonl 记录

#### 5.23.5 指标评估

`LaneEval` 完成：

- 逐条 GT 匹配预测车道
- 计算 line accuracy
- 聚合得到：
  - `Accuracy`
  - `FP`
  - `FN`

#### 5.23.6 输出

最终可以得到两类结果：

- `pred.json` 这样的标准预测文件
- 指标数值与可视化图像

### 5.24 当前后处理链路体现出的设计取向

从当前实现可以看出，这个项目的后处理并不是非常简单的“阈值一过就输出”，而是带有明显的结构化设计。

#### 5.24.1 它假设车道线是整条几何曲线

这体现在：

- 用 `anchor + offset` 逐点恢复整条线
- 用 lane NMS 比较整条线的重合程度
- 用 polyfit 修正整条线的连续性

#### 5.24.2 它用质量分数定优先级，再用几何规则做去重

这体现在：

- 候选排序只按 `score`
- NMS 同时看共同区间、整体均距、顶部和底部距离

这说明作者更看重：

- 先让质量感知分类分数决定保留顺序
- 再让整条线的几何关系决定是否视为重复

而不是把所有因素都混进一个手工排序公式里。

#### 5.24.3 它把训练验证和离线推理统一在同一条后处理链上

这意味着：

- 训练时看到的验证指标
- 与最终导出评估的指标

在处理逻辑上基本一致，这对实验判断是很有帮助的。

### 5.25 步骤 5 的本质

步骤 5 的本质可以概括为：

`将模型输出的 Anchor 分类与偏移预测，经过阈值筛选、几何恢复、lane NMS、曲线平滑和 TuSimple 格式转换，最终得到可提交、可评估、可可视化的车道线检测结果，并通过官方风格指标计算 Accuracy、FP、FN。`

如果再压缩成更工程化的一句话，就是：

`前向输出张量 -> 解码成车道曲线 -> 去重和平滑 -> 转官方格式 -> 计算最终任务指标`

这就是当前项目从“模型预测”走到“最终结果文件”的完整技术实现。

---

## 步骤 6：可视化、实验产物、目录组织与端到端闭环总结

这一步解决的问题是：

`当前项目在真实工程使用中，除了模型训练和推理本身，还如何检查数据质量、检查匹配质量、查看训练曲线、查看预测结果、管理实验产物，以及如何从头到尾跑通完整闭环`

这一步主要对应的文件与目录有：

- `tools/vis_dataset.py`
- `tools/visualize.py`
- `tools/visualize_row.py`
- `tools/plot_loss.py`
- `tests/test_models.py`
- `tests/test_losses.py`
- `tests/test_row_lane_head.py`
- `outputs/`
- `configs/`
- `data/`

如果说前 5 步已经把“算法主链路”讲清楚，那么第 6 步要解决的就是：

“这个项目在真实实验中到底如何被使用、检查、记录和复现。”

### 6.1 为什么工程里必须有可视化与实验管理

车道线检测不是只看最终数字就够的任务，原因主要有三点：

1. 数据和标注问题很常见
2. Anchor 匹配问题不容易只靠日志看出来
3. 即使指标变化不大，预测形态也可能明显不同

所以当前项目除了训练、推理、评估三条主链外，还专门配了几类辅助工具：

- 数据与匹配可视化
- 预测结果可视化
- 训练 loss 曲线可视化
- 单元测试
- 分目录保存 checkpoint 和输出结果

这些工具共同构成了项目的工程闭环。

### 6.2 `tools/vis_dataset.py` 的作用

`tools/vis_dataset.py` 是当前项目里非常重要的诊断工具，它主要用于：

- 查看原始 GT 车道线是否被正确解析
- 查看数据增强和 resize 后标注是否仍对齐
- 查看 Anchor 与 GT 的匹配结果
- 查看哪些点真正参与了回归监督

也就是说，它不是普通的“看看数据长什么样”，而是一个：

- 数据层 + 标签层 + Anchor 匹配层

联合诊断工具。

### 6.3 `vis_dataset.py` 实际能看什么

这个脚本在可视化时主要能呈现 4 类信息。

#### 6.3.1 GT 车道点

脚本会把 Dataset 输出的：

- `lanes`
- `valid_mask`
- `h_samples`

画到图像上，帮助检查：

- 车道点位置是否正确
- resize 后是否还与道路结构对齐
- 数据解析是否存在偏移或丢点

#### 6.3.2 正样本 Anchor 曲线

如果开启：

- `--show_match`

脚本还会把匹配到 GT 的正 Anchor 画出来。

这对检查以下问题非常关键：

- 某条 GT 是否真的找到了合适 Anchor
- 正样本是不是过少
- 是否存在明显交叉误配
- side anchor 是否真的覆盖到了侧向车道

#### 6.3.3 实际参与回归监督的点

如果再开启：

- `--show_reg_mask`

脚本会额外绘制：

- `offset_valid_mask`

对应的监督点。

这一步能帮助判断：

- Anchor 虽然匹配成功，但真正参与回归损失的点有多少
- 是否存在“匹配上了但监督点太少”的情况

#### 6.3.4 Anchor 标签统计信息

脚本在控制台还会输出：

- 正样本数
- 负样本数
- 忽略样本数
- 每条 GT 的最大 IoU 分布
- 每条 GT 的正样本数分布
- ignore 原因统计

这和训练前的匹配统计是相互呼应的。

### 6.4 为什么 `vis_dataset.py` 在这个项目中很关键

对于 Ray-Anchor-based 车道线检测而言，很多训练问题其实不是模型造成的，而是：

- 数据没解析好
- Anchor 设计覆盖不足
- 匹配阈值不合适
- 回归监督区域太少

这些问题只看 loss 曲线往往很难定位，而 `vis_dataset.py` 可以直接回答：

- 当前 GT 是怎么被看见的
- 当前 Anchor 是怎么配上的
- 当前回归究竟监督了哪些点

所以它本质上是这个项目最重要的“训练前排错工具”之一。

### 6.5 `tools/visualize.py` 的作用

与 `vis_dataset.py` 不同，`tools/visualize.py` 关注的是：

- 模型最终预测结果

它读取的是：

- `pred.json`

然后把预测车道线重新画回原图上。

它的核心用途包括：

- 检查是否有明显漏检
- 检查是否有明显误检
- 检查预测线是否重复
- 检查预测线是否断裂或抖动
- 对比不同版本模型的预测形态

### 6.6 `tools/visualize.py` 的工作逻辑

这个脚本的流程相对直接：

1. 读取配置或显式指定的数据集根目录
2. 读取 `pred.json`
3. 对每条样本，根据 `raw_file` 找到原图
4. 将每条 lane 在各个 `h_samples` 的有效点连接成线
5. 保存到指定输出目录

它不再关心训练时的 Anchor、offset、mask，而是只处理最终预测结果。

因此可以把它理解成：

- “最终结果展示层”

### 6.7 `tools/plot_loss.py` 的作用

`tools/plot_loss.py` 用于从训练日志里提取 loss，并画出曲线。

当前它主要解析：

- `Total Loss`
- `Cls Loss`
- `Reg Loss`

这虽然不覆盖所有细粒度项，但已经足够用于检查：

- 总体收敛是否正常
- 分类损失是否过高
- 回归损失是否下降
- 是否存在明显震荡

### 6.8 为什么 loss 曲线和最终指标要结合看

在这个项目里，只看 loss 曲线不够，原因是：

- 训练 loss 下降不一定代表最终 TuSimple 指标提升
- 车道线的几何质量和 NMS / polyfit 的影响也会反映到最终结果中

所以更合理的实验观察方式是联合看：

- `train.log` 中的训练损失
- 每个 epoch 的验证集 `Accuracy / FP / FN`
- `plot_loss.py` 输出的曲线
- `visualize.py` 输出的可视化结果

这四者一起，才能较完整地判断实验是否真的变好。

### 6.9 `tests/` 目录目前起什么作用

项目中还包含基础测试：

- `tests/test_models.py`
- `tests/test_losses.py`
- `tests/test_row_lane_head.py`

它们的定位不是完整回归测试，而是最基础的 smoke test。

#### 6.9.1 `test_models.py`

这个测试会：

1. 构造一份简化配置
2. 生成一套 Anchor
3. 构造随机输入图像
4. 前向运行 `LaneDetector`
5. 检查输出 shape 是否符合预期

它主要用于验证：

- 模型至少能顺利完成一次 forward
- 输出张量维度和 Anchor 数量是匹配的

#### 6.9.2 `test_losses.py`

这个测试会：

1. 构造随机 logits 和 target
2. 测试 `QualityFocalLoss`
3. 构造随机回归输入、目标和 mask
4. 测试 `RegLoss`

它主要验证：

- 损失函数至少能正常运行
- 输出标量在基本范围内

这类测试虽然不深，但它们在工程中仍有意义，因为至少能防止：

- 接口改坏
- shape 对不上
- loss 直接报错

#### 6.9.3 `test_row_lane_head.py`

这个测试会：

1. 构造简化版 `RowLaneHead`
2. 输入随机特征图
3. 检查 `exist_logits`、`row_valid_logits` 和 `grid_logits` 的 shape
4. 检查横向布局变化是否会影响位置分类输出

它主要用于验证：

- row/grid 分支的 Head 接口没有被改坏
- 位置分类输出仍然真正依赖横向空间结构

### 6.10 项目目录在完整技术流程中的角色

为了真正理解整个项目的端到端闭环，必须把主要目录和流程对应起来。

#### 6.10.1 `configs/`

这里定义实验方案，决定：

- 数据路径
- Anchor 参数
- 模型结构
- 损失权重
- 训练超参数
- 验证与推理解码参数

可以把它看成：

- “实验配置层”

#### 6.10.2 `data/`

这里放置数据集和索引文件，例如：

- 原始 TuSimple 标注
- `train.json`
- `val.json`

它对应：

- “数据来源层”

#### 6.10.3 `lane_det/`

这里是全部核心算法实现，包括：

- 数据加载
- Anchor
- 模型
- 损失
- 后处理
- 指标转换

它对应：

- “核心算法层”

#### 6.10.4 `tools/`

这里是对外脚本入口，包括：

- 数据准备
- 数据可视化
- `anchor-based` 训练与推理
- `row/grid-based` 训练与推理
- 评估
- 预测可视化
- 曲线绘图

它对应：

- “工程执行层”

#### 6.10.5 `outputs/`

这里用于保存实验产物，例如：

- checkpoints
- 可视化图像
- 推理输出
- loss 曲线

它对应：

- “实验结果层”

### 6.11 当前项目里典型实验产物有哪些

当一次完整实验跑完后，通常会产出以下几类结果。

#### 6.11.1 训练产物

- `train.log`
- `epoch_x.pth`
- `last.pth`
- `best.pth`

它们分别对应：

- 训练过程记录
- 每轮快照
- 最新模型
- 最优模型

#### 6.11.2 中间诊断产物

- GT / 匹配可视化图像
- Anchor 匹配统计信息
- loss 曲线图

这些结果帮助判断：

- 数据是否正确
- 匹配是否合理
- 训练是否稳定

#### 6.11.3 最终输出产物

- `pred.json`
- 预测可视化图像
- 评估指标结果

这些结果是面向最终性能判断的。

### 6.12 从头到尾一次 Anchor-based 完整实验怎么跑

如果从工程使用角度，把这个项目完整走一遍，典型顺序如下。

#### 6.12.1 第一步：准备数据列表

运行：

- `tools/prepare_tusimple_split.py`

产物：

- `train.json`
- `val.json`

#### 6.12.2 第二步：检查数据与匹配

运行：

- `tools/vis_dataset.py`

检查：

- GT 解析是否正确
- Anchor 匹配是否合理
- 正负样本数量是否正常
- 监督点是否足够

#### 6.12.3 第三步：启动训练

运行：

- `tools/train.py`

产物：

- `train.log`
- 各类 checkpoint
- 每轮验证指标

#### 6.12.4 第四步：检查训练曲线

运行：

- `tools/plot_loss.py`

检查：

- total / cls / reg loss 是否正常下降

#### 6.12.5 第五步：离线推理导出

运行：

- `tools/infer.py`

产物：

- `pred.json`

#### 6.12.6 第六步：计算最终指标

运行：

- `tools/evaluate.py`

产物：

- `Accuracy`
- `FP`
- `FN`

#### 6.12.7 第七步：可视化最终预测

运行：

- `tools/visualize.py`

检查：

- 预测几何质量
- FP / FN 具体出现在哪些图
- 是否有重复线、抖动、断裂等问题

这 7 步串起来，就是 `anchor-based` 分支的一次完整实验闭环。

### 6.13 Anchor-based 分支的完整总串联

到这里，可以把 `anchor-based` 分支的技术流程从头到尾再完整压缩成一条主线：

1. 准备 TuSimple 原始标注和图像
2. 合并原始标注，生成 `train.json` / `val.json`
3. `TuSimpleDataset` 读取图像和标注
4. 将原始 `lanes + h_samples` 转为统一内部表示
5. 同步做翻转、颜色扰动、缩放和归一化
6. 按当前图像尺寸生成 Anchor 集
7. 通过 `LabelAssigner` 完成 Anchor-GT 匹配
8. 生成 `cls_label + offset_label + offset_valid_mask`
9. `DataLoader` 将样本组成 batch
10. `LaneDetector` 通过 `ResNet18 + FPN + AnchorHead` 提取和预测
11. `AnchorFeaturePooler` 沿 Anchor 轨迹采样多尺度特征
12. 分类头输出每条 Anchor 的质量感知分类分数
13. 回归头输出每条 Anchor 的逐点偏移
14. refinement 阶段对回归结果做二次细化
15. 使用 `QualityFocalLoss + SmoothL1 + 可选 line loss` 联合训练
16. 使用 `Adam + MultiStepLR + 梯度累计` 更新参数
17. 每个 epoch 后在验证集上做完整解码与评估
18. 保存 `epoch`、`last`、`best` checkpoint
19. 用 checkpoint 在验证集或测试集上离线推理
20. `LaneDecoder` 做分数筛选、坐标恢复、NMS 与 polyfit 平滑
21. `TuSimpleConverter` 将结果转为官方格式 `pred.json`
22. `LaneEval` 计算 `Accuracy / FP / FN`
23. 用 `visualize.py` 查看预测结果
24. 用 `vis_dataset.py` 与 `plot_loss.py` 回查问题并继续迭代

这 24 个环节共同组成了 `anchor-based` 分支的完整技术闭环。

### 6.14 Anchor-based 分支的总结性判断

从整体上看，`anchor-based` 分支属于一种非常明确的：

- Ray-Anchor-based
- 固定 y 采样
- 序列化回归
- 多尺度特征采样
- 两阶段 refinement

的车道线检测方案。

它不同于纯分割法的地方在于：

- 它显式建模车道线模板
- 显式做 Anchor-GT 匹配
- 显式回归每条车道的轨迹偏移
- 显式做 lane-level NMS 和结构化后处理

因此，这条分支的技术思想从头到尾都比较统一：

- 把车道线看作“结构化几何曲线”
- 而不是简单的像素区域

### 6.15 步骤 6 的本质

步骤 6 的本质可以概括为：

`通过数据可视化、匹配可视化、训练日志、loss 曲线、预测可视化、测试脚本和实验目录管理，把前面 5 个步骤形成一个可检查、可复现、可迭代优化的完整工程闭环。`

如果再压缩成更工程化的一句话，就是：

`算法主链路之外，再补齐诊断工具、实验产物和复现路径，项目才真正成为一个完整可用的车道线检测工程。`

---

## 步骤 7：两条技术分支的关系与当前 row/grid 主线

这一步解决的问题是：

`在共享数据底座之上，项目为什么会从 anchor-based 路线继续演化出 row/grid-based 路线；两条路线分别解决什么问题；在论文里应该如何定义它们之间的关系`

这一步主要对应的文件为：

- `configs/tusimple_row_res18_fpn.yaml`
- `lane_det/datasets/row_target_builder.py`
- `lane_det/models/row_lane_head.py`
- `lane_det/models/row_lane_detector.py`
- `tools/train_row.py`
- `tools/infer_row.py`
- `tools/visualize_row.py`
- `tests/test_row_lane_head.py`

如果说步骤 2 到步骤 6 详细展开的是第一条完整工程路线，那么步骤 7 说明的就是：在保留原有数据、骨干网络和评估体系的基础上，项目如何演化出当前重点推进的 row/grid 主线。

### 7.1 为什么会从 Anchor 路线继续分化出新分支

`anchor-based` 路线最早被完整实现出来，因此它在工程上价值很大：

- 它验证了“固定 y 采样的结构化车道线表达”是可行的
- 它跑通了从数据、训练、推理到 TuSimple 评估的完整闭环
- 它为项目提供了第一条可复现实验基线

但随着实验深入，这条路线也暴露出明显问题：

- 训练质量对匹配超参数非常敏感
- 分类、回归、匹配和后处理之间耦合较强
- 一个改动往往同时改变多个机制，诊断困难
- 调参成本高，而且很多实验很难给出清晰结论

因此，项目后续没有直接抛弃结构化建模思路，而是保留共享的数据底座和骨干网络，把“车道如何表示、如何监督、如何解码”这一部分重新设计，形成了新的 `row/grid-based` 分支。

### 7.2 两条路线共享什么，分歧点在哪里

两条路线共享的部分包括：

- 同一份 TuSimple 数据和 `train.json` / `val.json`
- 同一个 `TuSimpleDataset` 与标注解析逻辑
- 同一套图像增强与缩放归一化
- 同样的固定 `y_samples=56` 表示
- 同样的 `ResNet18 + LaneFPN` 特征提取骨架
- 同样的 `TuSimpleConverter` 与 `LaneEval`

两条路线真正分歧的地方发生在“标准化车道线表示之后”：

- `anchor-based`：先生成 Ray Anchor，再做 Anchor-GT 匹配，最后回归相对 Anchor 的逐点偏移
- `row/grid-based`：不再生成 Anchor，而是直接把车道整理进固定槽位，并在固定 row 上预测横向位置网格

从论文写作角度，这意味着项目的整体框架可以被表述为：

`共享数据与评估底座 + 两种结构化车道线建模方案`

### 7.3 Row/Grid 分支的数据表示与监督目标

当前 `row/grid-based` 分支的核心配置位于 `configs/tusimple_row_res18_fpn.yaml`，关键参数为：

- 最多预测 `K=5` 条车道
- 固定 `num_y=56` 个 row
- 图像宽度离散为 `num_grids=100` 个位置网格

它对应的监督构造由 `lane_det/datasets/row_target_builder.py` 完成。其核心思想是：

1. 先按所有有效点的平均 `x` 对 GT 车道从左到右排序
2. 最多保留前 `K` 条车道，不足则补空槽位
3. 为每条槽位生成 `exist`
4. 为每条槽位生成连续坐标 `x_coords` 和有效点掩码 `coord_mask`
5. 保留 `row_h_samples`，供后续解码与导出使用
6. `RowTargetBuilder` 内部仍会额外生成 `grid_targets`，但当前训练主链在 `collate_fn` 中主要使用的是归一化连续坐标 `x_targets_norm`

这样一来，任务就从：

`候选 Anchor 分类 + 相对 Anchor 偏移回归`

变成了：

`固定槽位是否存在车道 + 每个 row 是否有效 + 每个有效 row 的横向位置分布`

这条路线最核心的变化是：

- 不再依赖 Anchor 生成
- 不再依赖 Anchor-GT 匹配
- 不再依赖 `topk_per_gt` 之类的正样本兜底策略

### 7.4 Row/Grid 分支的模型结构

`row/grid-based` 分支的检测器是 `RowLaneDetector`，结构仍然延续：

- Backbone：`ResNet18`
- Neck：`LaneFPN`
- Head：`RowLaneHead`

其中 `RowLaneDetector` 会从 `LaneFPN` 的多尺度输出中取最高分辨率特征图，送入 `RowLaneHead`。

`RowLaneHead` 的核心流程可以概括为：

1. 对 FPN 输出特征做卷积投影
2. 通过 `AdaptiveAvgPool2d` 将特征整理成固定的 `row-grid` 布局
3. 使用 `grid_encoder` 编码二维网格特征
4. 使用 `row_encoder` 编码逐 row 的上下文
5. 引入可学习的 `lane_queries` 作为固定槽位的语义查询
6. 将 `row_feat`、`lane_queries` 及其逐元素交互项拼接后，经 `row_fuse` 得到每个槽位、每个 row 的 lane feature
7. 输出：
   - `exist_logits [B, K]`
   - `row_valid_logits [B, K, num_y]`
   - `grid_logits [B, K, num_y, num_grids]`

因此，这条路线虽然仍然是结构化车道线检测，但它已经不再沿 Anchor 轨迹建模，而是改成了：

- 先预测槽位级存在性
- 再预测每个 row 是否有效
- 最后对有效 row 的横向位置分布建模，并通过期望解码恢复连续横坐标

### 7.5 Row/Grid 分支的训练、推理与可视化

训练脚本 `tools/train_row.py` 主要完成五类损失计算：

- `exist_loss`：`BCEWithLogitsLoss`
- `row_valid_loss`：`BCEWithLogitsLoss`
- `grid_loss`：基于 `x_targets_norm + coord_masks` 的 soft grid supervision
- `coord_loss`：对期望解码坐标和连续目标坐标做逐点约束
- `smooth_loss`：基于期望解码坐标的二阶差分平滑约束

总体损失形式可以概括为：

`loss = exist_weight * exist_loss + row_valid_weight * row_valid_loss + grid_weight * grid_loss + coord_weight * coord_loss + diff_weight * smooth_loss`

推理脚本 `tools/infer_row.py` 的核心流程是：

1. 读取 checkpoint
2. 前向得到 `exist_logits + row_valid_logits + grid_logits`
3. 对 `exist_logits` 做 sigmoid，筛掉低分槽位
4. 对 `row_valid_logits` 做 sigmoid，得到每个槽位在每个 row 上的有效性掩码
5. 通过 `grid_logits` 的期望解码恢复连续 `x` 坐标
6. 使用连续性规则 `build_continuous_valid_mask(...)` 清理离散断点和异常跳变
7. 生成车道线字典，再通过 `TuSimpleConverter` 导出

这意味着相较于 `anchor-based` 路线，当前 row/grid 分支在推理阶段显著简化了链路：

- 不需要 Anchor 解码
- 不需要 Anchor 级匹配回放
- 不需要把 `anchor_x + offset` 再恢复成车道线
- 不把 Anchor-NMS 作为主链路的必要步骤

可视化脚本 `tools/visualize_row.py` 用于把 row/grid 分支的预测和 GT 直接叠加检查，从而观察：

- 槽位排序是否稳定
- 中间车道是否容易偏移
- 弯道、遮挡和短线场景下的连续性是否足够

### 7.6 当前 Row/Grid 分支的阶段性判断

按照仓库内现有总结文档，当前 row/grid 分支已经从更早期的连续坐标回归版本，演进为统一的网格分类版本。其当前阶段性结论可以概括为：

- 它已经不是“试错失败的分支”
- 它是当前更值得继续作为主线推进的方案
- 它在训练收敛性和结构稳定性上，已经明显优于更早期的连续回归版本

现有阶段性记录显示：

- 在完整训练日志中，`epoch 20` 的 best `Accuracy = 0.756733`
- 同时对应 `FP = 0.578499`
- `FN = 0.577670`
- `epoch 19` 的 `FP/FN` 略优，分别为 `0.571133 / 0.570304`

这说明当前主要矛盾已经不是“模型完全训不起来”，而是：

- 结构大体能对上，但仍存在系统性横向偏移
- 遮挡和中间车道上的局部连续性仍不足
- 弯道和困难场景下的稳定性还不够成熟

因此，对当前项目更准确的判断不是“row/grid 路线已经最终定型”，而是：

- 它已经成为当前主线
- 但它仍处在继续优化几何贴线精度和鲁棒性的阶段

### 7.7 在论文中应如何定义两条路线

如果本文档要作为毕业论文的重要依据，建议在论文中把两条路线关系写成下面这种结构：

1. 先介绍共享底座：
   - TuSimple 固定 `h_samples`
   - 统一的固定 `y` 采样车道线表示
   - `ResNet18 + FPN` 特征提取框架
   - TuSimple 官方指标评估闭环
2. 再介绍第一阶段方法：
   - 基于 Ray Anchor 的结构化车道线检测基线
   - 说明其匹配、回归与后处理设计
   - 说明它暴露出的超参数敏感和链路耦合问题
3. 最后介绍当前主线：
   - 将车道表达重构为固定槽位、固定 row 的横向网格分类
   - 减少对匹配规则和 Anchor 模板的依赖
   - 作为当前继续优化与汇报实验结果的主线方案

如果压缩成一段更适合论文方法章节开头的话，可以写成：

`本项目首先实现了一条基于 Ray Anchor 的结构化车道线检测基线，并据此完成了完整的训练、推理与评估闭环；在此基础上，针对 Anchor 匹配敏感、模块耦合较强和诊断成本较高的问题，进一步将车道表示重构为固定 row 上的横向位置网格分类形式，形成了当前重点推进的 row/grid-based 主线方案。`

---

## 结论：当前项目的完整技术流程

综合前 7 个步骤，当前车道线项目的完整技术流程应当概括为：

`数据准备 -> 标注解析 -> 固定 y 采样的统一车道线表示 -> 分支 A：Ray Anchor 生成、匹配与偏移回归 / 分支 B：固定槽位、固定 row 的位置网格分类 -> TuSimple 格式导出 -> Accuracy / FP / FN 评估 -> 可视化分析与实验迭代`

这条链路覆盖了：

- 数据层
- 标签层
- 模型层
- 损失层
- 训练层
- 推理层
- 评估层
- 工程管理层

如果进一步按论文口径压缩，当前项目最合理的定位应当是：

- 项目共享一套统一的数据、特征提取和评估底座
- `anchor-based` 路线是最早完成的结构化检测基线
- `row/grid-based` 路线是当前重点推进的主线方案

到这里，这个项目从头到尾的完整技术流程，以及两条技术分支之间的关系，已经可以形成一个适合论文引用的闭环表述。

## 文档维护说明

本文档已完成当前版本的主流程整理。后续若项目结构继续变化，可以继续补充以下内容：

- 不同配置版本的演化对比
- Anchor 匹配策略版本差异
- refinement 模块迭代记录
- row/grid 分支的持续优化记录
- 后处理参数调优经验
- 与其他车道线方法的对比分析
