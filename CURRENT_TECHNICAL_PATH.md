# 当前项目技术路径总览（可直接发给 ChatGPT）

## 1. 项目目标与方法定位
- 任务：2D 车道线检测（TuSimple 数据格式）。
- 方法范式：`Anchor-based`，以固定 `y` 采样上的 `x` 序列表示车道线。
- 当前主干：`ResNet18 + FPN + Anchor Head`。
- 当前增强：`Anchor Feature Pooling` + `2-stage offset refinement`。

---

## 2. 端到端技术路径（当前实现）

### A. 数据与预处理
- 数据集类：`lane_det/datasets/tusimple.py`
- 输入标注：TuSimple JSON 行格式（`raw_file`, `lanes`, `h_samples`）。
- 预处理：`Resize + Normalize`（`lane_det/datasets/transforms.py`）。
- 统一采样：将车道插值/重采样到配置里的固定 `y_samples=56`。

### B. Anchor 生成
- 模块：`lane_det/anchors/anchor_generator.py`
- Anchor 类型：
  - Bottom anchors：由 `x_positions × angles` 生成。
  - Side anchors：从左右边界多起点扩展，增强侧向/短线覆盖。
- 结果字段：`anchor_xs`, `valid_mask`, `x_bottom`, `angles`, `y_samples`。
- 缓存：`outputs/cache/anchors/*.npz`。

### C. Anchor-GT 匹配与标签分配
- 模块：`lane_det/anchors/label_assigner.py`
- 匹配度量：Line IoU（按共享有效点计算）。
- 约束/gating：
  - 共享点数与比例约束（side/bottom分开）。
  - top 区域一致性约束。
  - 几何约束（角度、bottom 交点）。
- 标签：
  - `cls_label`：`1/0/-1`（正/负/忽略）
  - `offset_label`：GT 相对 anchor 的 x 偏移
  - `offset_valid_mask`：回归监督掩码
- 当前还包含：
  - Dynamic Top-K 强制正样本（可配）
  - Soft-gating（可配，惩罚 IoU 而非直接废弃）

### D. 模型结构
- 检测器：`lane_det/models/detector.py`
  - Backbone：`lane_det/models/backbone.py`（ResNet18）
  - Neck：`lane_det/models/fpn.py`（LaneFPN，输出高分辨率单层特征）
  - Head：`lane_det/models/head.py`
- Head 关键路径：
  1. `AnchorFeaturePooler`（`lane_det/models/anchor_feature_pooler.py`）通过 `grid_sample` 沿 anchor 轨迹采样特征
  2. 分类分支输出 `cls_logits [B, N_anchor]`
  3. 回归分支：
     - stage1：粗偏移 `reg_stage1`
     - stage2：细化增量 `reg_delta_stage2`
     - 最终：`reg_final = reg_stage1 + reg_delta_stage2`

### E. 损失与训练
- 训练脚本：`tools/train.py`
- 损失：
  - 分类：`FocalLoss`（`lane_det/losses/focal_loss.py`）
  - 回归：`SmoothL1`（`lane_det/losses/reg_loss.py`）
- 回归总损失（refinement 开启时）：
  - `reg_loss = w1 * stage1 + w2 * stage2`
  - 再与分类损失按 `cls_weight/reg_weight` 加权
- 训练细节：
  - Adam + MultiStepLR（milestones 12/16）
  - 梯度累积步数固定为 2
  - 日志与权重保存到 `outputs/checkpoints/...`

### F. 推理、后处理与格式转换
- 推理脚本：`tools/infer.py`
- 后处理解码：`lane_det/postprocess/decoder.py`
  - `score_thr` 过滤
  - 基于轨迹距离的 lane-NMS
  - 多项式平滑（降低抖动）
- TuSimple 输出转换：`lane_det/metrics/tusimple_converter.py`
  - 转成 `pred.json` 的 TuSimple 格式行
  - 使用目标 `h_samples`（默认 160~710 step 10）

### G. 评估与可视化
- 评估：`tools/evaluate.py`（TuSimple Python 实现，输出 Accuracy/FP/FN）
- 可视化：
  - GT/匹配可视化：`tools/vis_dataset.py`
  - 预测可视化：`tools/visualize.py`
  - 训练 loss 曲线：`tools/plot_loss.py`

---

## 3. 当前主要配置与分支

### 基线配置
- `configs/tusimple_res18_fpn.yaml`
- 特点：ResNet18+FPN+Anchor；已包含 side anchors 与 IoU 匹配约束。

### 当前主实验配置（match+refinement）
- `configs/tusimple_res18_fpn_matchv2.yaml`
- 特点：
  - `use_masked_pooling: true`
  - `use_refinement: true`
  - 更强调回归（`reg_weight=2.0`）
  - stage1/stage2 回归损失联合训练
  - match 参数单独可调（Top-K、soft gating、side/bottom 阈值分离）

### 测试集配置
- `configs/tusimple_test.yaml`
- 指向 `archive/TUSimple/test_set` 与 `archive/TUSimple/test_label.json`。

---

## 4. 代码入口命令（当前项目惯用）

```bash
# 1) 数据集与匹配可视化
python tools/vis_dataset.py --cfg configs/tusimple_res18_fpn_matchv2.yaml --out outputs/visualizations/m2_check --show_match --max 20

# 2) 训练
python tools/train.py --cfg configs/tusimple_res18_fpn_matchv2.yaml --work-dir outputs/checkpoints/v2_match_loss

# 3) 推理
python tools/infer.py --cfg configs/tusimple_test.yaml --ckpt outputs/checkpoints/v2_match_loss/<time>/last.pth --out pred_vXX.json

# 4) 评估
python tools/evaluate.py --pred pred_vXX.json --gt archive/TUSimple/test_label.json

# 5) 可视化预测
python tools/visualize.py --cfg configs/tusimple_test.yaml --pred pred_vXX.json --out outputs/visualizations/pred_vXX
```

---

## 5. 当前产物状态（仓库可见）
- 推理结果已迭代到：`pred_v1.json` ~ `pred_v13.json`。
- 检查点目录包含：
  - `outputs/checkpoints/refine_smoke/...`
  - `outputs/checkpoints/v2_match_loss/...`
- 可视化目录包含多个版本：`outputs/visualizations/pred_v*`。

---

## 6. 给 ChatGPT 的沟通建议（可直接贴）
- 这是一个 **Anchor-based 车道线检测工程**，不是分割法。
- 当前重点是 **匹配策略优化 + Anchor feature pooling + 2-stage refinement**。
- 请优先围绕以下方向给建议：
  1. Line IoU 匹配阈值/soft-gating/top-k 的协同调参策略  
  2. stage1/stage2 回归损失权重与稳定性  
  3. 解码阶段 NMS 与平滑策略对 Accuracy/FP/FN 的影响  
  4. 如何在不大改框架下提升曲线拟合和远端稳定性

