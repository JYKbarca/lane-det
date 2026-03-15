请在我当前的车道线检测项目中，实现一个“Soft Line Overlap Loss”，作为现有 SmoothL1 回归损失的补充项，而不是替代项。

【目标】
我现在的问题不是“车道线检不出来”，而是“每条车道线大多能检测出来，但曲线拟合不够准，尤其远端容易发飘”。  
所以我希望新增一个“线级几何损失”，让模型除了优化逐点 offset 误差之外，还要优化“整条车道线和 GT 的整体贴合度”。

【当前项目基础】
当前项目已经具备这些条件：
- Anchor-based lane detection
- 车道线表示为固定 y_samples 上的 x 序列
- 已有 offset_label 和 offset_valid_mask
- 已有 2-stage refinement
- 当前 reg loss 是 SmoothL1
- 匹配阶段已经使用 Line IoU 思想

所以这次不要重构框架，只是在现有训练损失上增加一个 line-level 几何约束。

【原理】
对于每个正样本 anchor：
1. 根据 anchor_xs + pred_offsets 得到预测车道线 pred_xs
2. 根据 anchor_xs + gt_offsets 得到真实车道线 gt_xs
3. 只在 offset_valid_mask 为 1 的采样点上计算
4. 对每个有效点，计算横向距离 d = |pred_x - gt_x|
5. 把距离映射成一个 soft 相似度，例如：
   s = exp(-d^2 / (2*sigma^2))
6. 对整条 lane 的所有有效点求平均，得到整体 overlap 分数 S
7. 定义 line loss = 1 - S

这样做的意义是：
- SmoothL1 负责“点级精修”
- Soft Line Overlap Loss 负责“整条线整体贴合”
- 两者结合，更适合解决曲线 lane 和远端漂移问题

【实现要求】
1. 新增一个 loss 模块，比如 SoftLineOverlapLoss
2. 只对正样本计算
3. 只在有效点 mask 上计算
4. 同时支持 stage1 和 stage2 / final 输出
5. 总损失改为：
   total_loss = cls_loss + reg_loss + lambda_line * line_loss
6. 默认先用较小权重，比如 lambda_line = 0.1
7. 不要删除现有 SmoothL1
8. 不要改成 segmentation、anchor-free 或大重构
9. 不要直接用不可微的硬 IoU，必须做成 soft、可导版本

【配置建议】
请把下面参数做成可配置：
- use_line_overlap_loss
- line_loss_weight
- line_sigma
- line_min_valid_points

【输出要求】
请完成：
1. 代码实现
2. 训练流程接入
3. 配置文件增加参数
4. 训练日志里打印 line_loss
5. 给我一份简短说明：这个 loss 的公式、输入输出、以及它为什么能改善曲线拟合和远端漂移

【重点】
这次改动的本质不是提升“能不能检测到”，而是提升“已经检测到的车道线，是否和 GT 更贴合”。  
请优先复用我当前项目里已有的 lane 表示、valid mask 和 Line IoU 思想，做最小侵入式实现。