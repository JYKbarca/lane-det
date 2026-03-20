## fix.md 校正版

更新时间：`2026-03-19`

### 一句话结论
当前项目的主问题不是单点 bug，而是：
`训练目标`、`样本定义` 没有对齐。

当前主战场不是 `Step1`、`Step5` 或 `Step6`，而是：
`Step2 分类损失` + `Step3 正样本匹配`。

---

## 当前优先级

推荐执行顺序：
`4 -> 7` (2、3、5、6 已完成)

执行原则：
- 每次只改一类问题
- 每做完一步，单独重训或单独验证一次
- 不要同时改多个变量

---

## Step1. Refinement 架构

### 当前结论
这一项已经完成，不再作为主攻方向。

### 当前依据
- `lane_det/models/head.py`
  - Stage2 已经显式使用 `Stage1` 预测结果
  - 最终形式是 `reg_final = reg_stage1 + reg_delta_stage2`
- 最新日志里，`Reg2` 大多数 step 低于 `Reg1`

### 保留检查项
- `Reg2` 是否持续优于 `Reg1`
- Stage2 是否退化回并行第二个头

### 验收标准
- `Reg2` 长期低于 `Reg1`

---

## Step2. 分类损失

### 当前结论
这一项已经完成，不再作为主攻方向。

### 历史问题
- `lane_det/losses/focal_loss.py`
  - 当前仍然使用 `focal_loss.sum() / num_pos`
- 当前样本分布约为：
  - `pos = 9456`
  - `neg = 600425`
  - `ignore = 290719`
  - `neg:pos ≈ 63.5:1`
- `alpha=0.6` 对这种失衡强度不够

### 历史影响
- 分子包含全部正负样本损失，分母只用正样本数
- 负样本总梯度会被放大
- 分类项在总 loss 中占比过低
- 模型主要在优化回归，不是在优化“哪些 lane 应该保留”

### 修复方案
- 修改了 focal loss 的归一化方式，将正负样本分开归一化（`loss_pos/num_pos + loss_neg/num_neg`），平衡了正负样本的总梯度。
- 在配置文件中将 `cls_weight` 从 1.0 提高到 10.0，显著提升了分类项在总 loss 里的实际影响力。

### 验收标准
- `Cls loss` 不再长期低到接近失效
- `FP` 明显下降

---

## Step3. 正样本匹配

### 当前结论
这一项已经完成，不再作为主攻方向。

### 历史问题
- `lane_det/anchors/label_assigner.py`
- 当前配置仍然过松：
  - `pos_thr_bottom = 0.25`
  - `pos_thr_side = 0.15`
  - `topk_per_gt = 4`
  - `min_force_pos_iou = 0.10`
- 低质量 anchor 会被翻成正样本

### 历史证据
- 前 200 张图统计：
  - `pos = 9456`
  - `neg = 600425`
  - `ignore = 290719`
- 每条 GT 平均正 anchor 数约 `11.95`
- 单条 GT 最多正 anchor 数为 `93`

### 历史影响
- 分类标签变脏
- 回归目标也会被脏匹配拖偏
- 模型会学到“很多近似 lane 都算正”

### 修复方案
- 在 `tusimple_res18_fpn_matchv2.yaml` 中收紧了正样本匹配参数：
  - `topk_per_gt` 降低为 1
  - `min_force_pos_iou` 提高为 0.20
  - `pos_thr_bottom` 收紧为 0.35
  - `pos_thr_side` 收紧为 0.25

### 验收标准
- 每条 GT 的正 anchor 数下降
- 低 IoU 正样本明显减少
- `FP` 稳定下降

---

## Step4. 统计可解释性

### 当前结论
这是诊断问题，不是主提分点。

### 问题
- 当前 soft gating 打开时：
  - `top_fail`
  - `angle_fail`
 统计会混入软惩罚样本
- 统计结果会把“降权”误写成“过滤”

### 要修什么
- 分开统计：
  - hard ignore
  - soft penalty
  - gray zone

### 验收标准
- 能准确回答 ignore 的来源
- 调匹配时不再被统计误导

---

## Step5. 去掉双重 polyfit

### 当前结论
这一项已经完成，不再作为主攻方向（已删除多项式拟合代码）。

### 历史问题
- `lane_det/postprocess/decoder.py` 做了一次 polyfit
- `lane_det/metrics/tusimple_converter.py` 又做了一次加权 polyfit

### 历史影响
- 模型原始输出被二次改形
- 顶部车道更容易被底部形状拉偏
- 验证结果不能直接反映模型本身能力

### 修复方案
- 删除了多项式拟合代码，直接使用模型原始输出进行评估。

---

## Step6. 重做 NMS

### 当前结论
这一项已经完成，不再作为主攻方向（已重构 NMS 逻辑）。

### 历史问题
- `lane_det/postprocess/decoder.py`
- 当前重复判定是写死规则
- `nms_thr` 主要只控制“开不开 NMS”，不真正控制抑制强度

### 历史影响
- 参数表面可调，实际核心判定不可调
- 重复 lane 是否存活主要取决于写死逻辑

### 修复方案
- 重构了 NMS 逻辑，让 NMS 的核心相似度判定参数化，使 `nms_thr` 能够真实地控制抑制强度。

---

## Step7. 清理工程问题

### 当前结论
这是低优先级问题。

### 问题
- `lane_det/datasets/tusimple.py`
  - 存在重复定义
- `lane_det/losses/reg_loss.py`
  - `beta=1.0` 写死
  - `smooth_l1_beta` 没有真正接通

### 要修什么
- 清掉重复定义
- 让配置参数真正生效

### 验收标准
- 调参结果可信
- 代码可维护性提高

---

## 最终判断

- `Step1`、`Step2`、`Step3`、`Step5`、`Step6` 均已完成。
- 当前主矛盾已经转移到：
  - 统计可解释性（Step4）
  - 清理工程问题（Step7）

当前真正应该优先做的是：
`Step4 + Step7`，并进行一次完整的重新训练以验证 `Step2` 和 `Step3` 的修改效果。
