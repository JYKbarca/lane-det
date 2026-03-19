## fix.md 校正版

更新时间：`2026-03-19`

### 一句话结论
当前项目的主问题不是单点 bug，而是：
`训练目标`、`样本定义`、`后处理输出`、`验证标准` 没有对齐。

当前主战场不是 `Step1`，而是：
`Step2 分类损失` + `Step3 正样本匹配` + `Step5/6 验证链路与后处理`。

---

## 当前优先级

推荐执行顺序：
`2 -> 3 -> 5 -> 6 -> 4 -> 7`

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
这是第一优先级问题。

### 问题
- `lane_det/losses/focal_loss.py`
  - 当前仍然使用 `focal_loss.sum() / num_pos`
- 当前样本分布约为：
  - `pos = 9456`
  - `neg = 600425`
  - `ignore = 290719`
  - `neg:pos ≈ 63.5:1`
- `alpha=0.6` 对这种失衡强度不够

### 影响
- 分子包含全部正负样本损失，分母只用正样本数
- 负样本总梯度会被放大
- 分类项在总 loss 中占比过低
- 模型主要在优化回归，不是在优化“哪些 lane 应该保留”

### 要修什么
- 改 focal loss 的归一化方式
- 重新检查 `alpha` 和 `gamma`
- 提高分类项在总 loss 里的实际影响力

### 验收标准
- `Cls loss` 不再长期低到接近失效
- `FP` 明显下降

---

## Step3. 正样本匹配

### 当前结论
这是第一优先级问题。

### 问题
- `lane_det/anchors/label_assigner.py`
- 当前配置仍然过松：
  - `pos_thr_bottom = 0.25`
  - `pos_thr_side = 0.15`
  - `topk_per_gt = 4`
  - `min_force_pos_iou = 0.10`
- 低质量 anchor 会被翻成正样本

### 当前证据
- 前 200 张图统计：
  - `pos = 9456`
  - `neg = 600425`
  - `ignore = 290719`
- 每条 GT 平均正 anchor 数约 `11.95`
- 单条 GT 最多正 anchor 数为 `93`

### 影响
- 分类标签变脏
- 回归目标也会被脏匹配拖偏
- 模型会学到“很多近似 lane 都算正”

### 要修什么
- 收紧正样本阈值
- 降低或关闭 `topk_per_gt`
- 提高 `min_force_pos_iou`
- 重新检查 side anchor 的通过条件

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
这是验证链路问题，优先级高。

### 问题
- `lane_det/postprocess/decoder.py` 做了一次 polyfit
- `lane_det/metrics/tusimple_converter.py` 又做了一次加权 polyfit

### 影响
- 模型原始输出被二次改形
- 顶部车道更容易被底部形状拉偏
- 验证结果不能直接反映模型本身能力

### 要修什么
- 只保留一个 polyfit 位置，或者先全部关闭
- 先评估原始输出，再决定是否保留单次平滑

### 验收标准
- 顶部车道不再明显失真
- 验证波动减小

---

## Step6. 重做 NMS

### 当前结论
这是验证链路问题，优先级高。

### 问题
- `lane_det/postprocess/decoder.py`
- 当前重复判定是写死规则
- `nms_thr` 主要只控制“开不开 NMS”，不真正控制抑制强度

### 影响
- 参数表面可调，实际核心判定不可调
- 重复 lane 是否存活主要取决于写死逻辑

### 要修什么
- 让 NMS 的核心相似度判定参数化
- 让 `nms_thr` 对实际保留结果产生真实影响

### 验收标准
- 重复 lane 明显减少
- `FP` 明显下降
- NMS 参数变化能反映到验证结果

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

- `Step1` 已完成，但不是当前瓶颈
- 当前主矛盾是：
  - 分类监督过弱
  - 正样本定义过脏
  - 后处理和验证链路会改写模型输出

当前真正应该优先做的是：
`Step2 + Step3 + Step5/6`
