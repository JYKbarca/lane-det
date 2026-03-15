你现在需要在现有的 anchor-based 车道线检测工程上继续实现两个增强模块：

==================================================
Step2: Anchor Feature Pooling（沿车道采样特征）
Step3: 2-Stage Refinement（多阶段预测细化）
==================================================

注意：
- 不要推翻现有工程结构
- 不要改动现有数据集读取、anchor 生成、label assign 的接口
- 优先做“最小侵入式改造”
- 目标是在现有代码基础上提升曲线车道线拟合能力和预测位置稳定性
- 保证 train.py 和 infer.py 仍然可以直接运行

==================================================
一、现状理解
==================================================

当前模型已经具备：
1. backbone + FPN 特征提取
2. anchor 生成
3. label assign
4. cls / offset 预测
5. train / infer / vis_dataset 基本跑通

当前主要问题：
- 预测出的车道线对曲线拟合不够准确
- 远端位置容易飘
- infer 可视化中，车道线形状不够贴合 GT

因此现在不优先大改匹配，而是增强网络的表达能力：
- Step2：让每条 anchor 从 feature map 上“沿着自己的轨迹采样特征”
- Step3：让 offset 预测从“一次预测”变成“粗预测 + 细化修正”

==================================================
二、Step2 要做什么：Anchor Feature Pooling
==================================================

【目标】
目前 head 可能是直接对整张 feature map 做卷积后预测 cls 和 offset。
现在需要改成：
对每一条 anchor，在 feature map 上沿着 anchor 的采样点提取特征，再基于这些“沿车道方向的特征”预测 cls 和 offset。

【核心思想】
anchor 本身已经有 num_y 个采样点：
    anchor_xs: [num_anchors, num_y]
以及对应的 y 采样位置。

这些点可以看成一条“离散车道线”。
现在需要在 FPN 的某一层特征图上，对这些点做双线性采样（grid_sample），得到每条 anchor 的 feature sequence。

也就是：

输入：
- feature_map: [B, C, Hf, Wf]
- anchor_points: [num_anchors, num_y, 2]   # 每个点是 (x, y)

输出：
- pooled_feature: [B, num_anchors, C, num_y]

然后对 pooled_feature 做进一步编码，例如：
- 先 reshape 为 [B*num_anchors, C, num_y]
- 再经过 Conv1D / 小型 MLP / pooling
- 得到每条 anchor 的 embedding
- 最后输出：
    cls_logits: [B, num_anchors]
    offset_pred: [B, num_anchors, num_y]

【实现要求】
1. 优先只使用 FPN 中分辨率最高的一层特征图（例如 P2 或当前用于 lane head 的那一层）
   不要一开始就做多层融合，先保证实现简单稳定

2. 使用 torch.nn.functional.grid_sample 实现沿 anchor 采样
   需要把图像坐标中的 anchor 点映射到 feature map 坐标，并归一化到 [-1, 1]

3. grid_sample 的输入输出需要严格检查 shape，避免坐标维度错误

4. 对于 anchor 中无效点（例如超出图像范围的点）：
   - 仍然可以送入 grid_sample，但需要在后续用 valid_mask 屏蔽
   - 或者将这些采样点 clamp 到合法范围，并结合 mask 处理
   - 不要因为个别点无效就丢掉整条 anchor

5. pooled_feature 编码建议用轻量结构：
   - Conv1D(in_channels=C, out_channels=C, kernel_size=3, padding=1)
   - ReLU
   - 再做一次 Conv1D 或全局 pooling
   最终得到每条 anchor 的 feature embedding

6. cls head 和 reg head 都基于这个 anchor embedding 来预测
   不要再直接对全图 feature map 预测

【你需要新建或修改的模块】
建议：
- 新建 lane_det/models/anchor_feature_pooler.py
  实现 AnchorFeaturePooler
- 修改 lane head，使其接收 pooled anchor feature，而不是整图 feature

【验收标准】
1. forward 能跑通，shape 正确
2. train.py 能正常开始训练
3. infer.py 能正常输出结果
4. 可视化中，预测车道线应比之前更稳定，远端不那么飘
5. 不要求第一版就明显提升指标，但必须保证结构合理、实现正确、代码清晰

==================================================
三、Step3 要做什么：2-Stage Refinement
==================================================

【目标】
当前 offset 预测大概率是单阶段：
    anchor -> offset_pred

现在改成两阶段：
    anchor -> stage1_offset -> stage2_delta -> final_offset

即：
- Stage1 先做一个粗预测
- Stage2 再在 Stage1 基础上做修正
- 最终输出：
    final_offset = offset_stage1 + delta_offset_stage2

【核心思想】
由于 anchor 本身是直线，而真实车道线可能是曲线，
单次 offset 回归往往难以准确拟合。
使用 2-stage refinement 后，第一阶段先把大致位置找对，
第二阶段专门修正弯曲和局部误差。

【实现要求】
1. 在 lane head 中保留一个 stage1 reg branch
   输出：
   - offset_stage1: [B, num_anchors, num_y]

2. 再增加一个 stage2 reg branch
   输入建议有两种简单方案，优先选更容易实现的：
   方案A（推荐，最简单）：
   - 直接使用与 stage1 相同的 anchor pooled feature
   - 再额外预测一个 delta_offset_stage2
   这样实现最简单，先保证可跑通

   方案B（进阶）：
   - 将 stage1 的 offset 经过小型 embedding 后，与 pooled feature 拼接
   - 再预测 delta_offset_stage2
   如果实现复杂度高，先不要做

3. 最终输出：
   - final_offset = offset_stage1 + delta_offset_stage2

4. 分类分支先不用做多阶段，保持单阶段 cls 即可
   即：
   - cls 仍然只输出一次
   - refinement 只针对 offset

5. 训练时对两个阶段都加回归损失：
   - loss_reg_stage1
   - loss_reg_stage2
   推荐总损失：
       loss_reg = 0.5 * loss_reg_stage1 + 1.0 * loss_reg_stage2
   具体权重可配置

6. 所有回归损失仍然使用现有的：
   - offset_label
   - offset_valid_mask
   不能破坏原有标签接口

7. infer 时默认输出 final_offset
   如果方便，可以增加一个 debug 开关，在可视化中同时画出：
   - stage1 预测
   - final 预测
   用于观察 refinement 是否有效

【你需要修改的模块】
- 现有 lane head / model forward
- train.py 中的 loss 计算，加入 stage1 + stage2 reg loss
- infer.py / visualize 只使用 final_offset

【验收标准】
1. 训练和推理都能跑通
2. loss 中同时包含两个阶段的 reg loss
3. infer 输出 final_offset 正常
4. 若启用 debug，可看到 stage2 相对 stage1 有修正效果
5. 可视化中，曲线车道应更贴合，抖动更少

==================================================
四、实现顺序要求
==================================================

请严格按下面顺序进行，不要一次性大改所有东西：

第一步：
实现 Step2 的 Anchor Feature Pooling
- 先只接一层 feature map
- 只替换 head 的输入
- 保证 forward / train / infer 跑通

第二步：
在 Step2 跑通的基础上，实现 Step3 的 2-stage refinement
- 先做最简单版本：stage2 直接基于同一个 pooled feature 预测 delta
- 不做复杂交互
- 保证 loss 和 infer 跑通

第三步：
增加必要的 debug 输出
包括但不限于：
- pooled feature shape
- stage1 offset shape
- stage2 delta shape
- final offset shape
避免 silent bug

==================================================
五、代码风格要求
==================================================

1. 所有新增类和函数都写清楚注释
2. 所有 tensor shape 在关键位置写明注释
3. 不引入新依赖
4. 不删除现有逻辑，只做增强
5. 配置项尽量写入 yaml 或 cfg，保证可开关控制，例如：
   - use_anchor_feature_pooling: true
   - use_refinement: true
   - refinement_stages: 2
   - reg_loss_stage1_weight: 0.5
   - reg_loss_stage2_weight: 1.0

==================================================
六、最终交付要求
==================================================

完成后请给出：
1. 改动了哪些文件
2. 新增了哪些类/函数
3. 每个新增模块的输入输出 shape
4. 如何训练
5. 如何推理
6. 若出现 shape 对不上或 grid_sample 坐标问题，请优先修复，不要跳过

请开始实现，先完成 Step2，再完成 Step3，不要省略中间说明。