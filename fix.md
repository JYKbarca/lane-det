按优先级，我建议这样拆成 7 步，严格一类问题一类问题地修，不要一次全改。

1. 修 `Refinement` 架构。
把 [head.py](f:/毕业设计/code/lane_det/models/head.py#L133) 的 Stage2 从“再跑一个并行回归头”改成“显式利用 Stage1 输出做残差细化”；同时检查 [train.py](f:/毕业设计/code/tools/train.py#L375) 的两阶段监督，让 Stage1 学粗定位、Stage2 学残差。验收标准是 `Reg2` 明显低于 `Reg1`，不再长期几乎重合。

2. 修分类损失的归一化和权重。
重点改 [focal_loss.py](f:/毕业设计/code/lane_det/losses/focal_loss.py#L40) 的 `sum()/num_pos`，避免负样本总梯度被放大；必要时同时提高分类项在总 loss 里的占比。验收标准是 `Cls loss` 不再长期压到极低，同时验证集 `FP` 开始下降。

3. 收紧正样本匹配策略。
调整 [label_assigner.py](f:/毕业设计/code/lane_det/anchors/label_assigner.py#L400) 对应的阈值和强制正样本逻辑，优先处理 `pos_thr_bottom/side`、`topk_per_gt`、`min_force_pos_iou`。验收标准是 pretrain match stats 里每条 GT 的正 anchor 数下降，低 IoU 正样本减少，`FP` 继续降。

4. 把“统计”和“真实过滤”分开。
当前 soft gating 开着时，[label_assigner.py](f:/毕业设计/code/lane_det/anchors/label_assigner.py#L361) 和 [label_assigner.py](f:/毕业设计/code/lane_det/anchors/label_assigner.py#L374) 的 `top_fail/angle_fail` 统计会误导判断。先把 hard ignore、soft penalty、gray zone 分开记清楚，否则后面调匹配会盲飞。验收标准是能准确回答 ignore 到底来自哪里。

5. 去掉双重 polyfit。
[decoder.py](f:/毕业设计/code/lane_det/postprocess/decoder.py#L157) 和 [tusimple_converter.py](f:/毕业设计/code/lane_det/metrics/tusimple_converter.py#L65) 现在都在拟合，必须只保留一个位置，最好先全部关掉再评估原始输出。验收标准是顶部车道不再被底部强行拉形，验证结果更稳定。

6. 重做 NMS，让阈值真正生效。
现在 [decoder.py](f:/毕业设计/code/lane_det/postprocess/decoder.py#L25) 的重复判定是固定 5 条规则的 AND，而 [decoder.py](f:/毕业设计/code/lane_det/postprocess/decoder.py#L17) 的 `nms_thr` 基本只决定“开不开”。要把它改成真正可调的抑制标准。验收标准是重复车道明显减少，`FP` 再降一截。

7. 清理代码和假配置项。
清掉 [tusimple.py](f:/毕业设计/code/lane_det/datasets/tusimple.py#L58)、[tusimple.py](f:/毕业设计/code/lane_det/datasets/tusimple.py#L145) 的重复定义；把 [reg_loss.py](f:/毕业设计/code/lane_det/losses/reg_loss.py#L17) 里的 `beta=1.0` 和配置接通，避免“配置写了但没生效”。这一步不一定直接提分，但能避免后续调参失真。

推荐执行顺序就是 `1 -> 2 -> 3 -> 5 -> 6 -> 7`，第 4 步穿插在第 3 步前后做。核心原则是每做完一步就单独重训/复现一次，不要把多个变量一起改掉。
