# 2D Lane Detection Technical Pipeline (Paper Implementation)

论文：《基于深度学习的车道线检测算法研究》

该论文提出一种 **基于 Ray Anchor 的 Anchor-based 车道线检测方法**。  
为了避免理解歧义，这里按照 **真实工程执行顺序**描述技术实现流程。

---

# 1. Initialization Stage (初始化阶段)

在模型运行之前，需要进行一次性的初始化操作：

1. 读取配置文件  
   - 图像尺寸  
   - anchor 参数 (x_positions, angles, y_samples)

2. 生成 Ray Anchors  
   - 根据图像尺寸和预设参数生成一组固定的候选车道线  
   - 每个 anchor 表示一条射线形式的车道假设

```
Anchor = (start_x, angle, y_samples)
```

注意：

- Ray Anchor **不依赖图像特征**
- Ray Anchor **在模型初始化时生成**
- 训练和推理过程中都复用同一组 anchors

---

# 2. Forward Stage (网络前向过程)

对于每一张输入图像，执行以下流程：

## Step 1 输入图像

```
Input Image
```

---

## Step 2 Backbone 特征提取

使用 **ResNet** 提取图像特征：

```
Image
↓
ResNet Backbone
↓
Feature Maps
```

---

## Step 3 Feature Pyramid Network

通过 FPN 融合多尺度特征：

```
Feature Maps
↓
FPN
↓
Multi-scale Feature Maps
```

---

## Step 4 Anchor Feature Pooling

将 **预生成的 Ray Anchors** 投影到 Feature Map 上，并沿 anchor 采样特征：

```
Ray Anchors + Feature Maps
↓
Anchor Feature Pooling
↓
Anchor Features
```

这是 **Anchor 与图像特征第一次结合的阶段**。

---

## Step 5 Feature Enhancement

融合 Anchor Feature 与 Image Feature：

```
Anchor Feature
↓
Pooling Feature Enhancement
↓
Enhanced Anchor Feature
```

---

## Step 6 Prediction Head

检测头对每个 anchor 进行预测：

```
Enhanced Feature
↓
Prediction Head
```

输出：

```
cls score
lane offset
lane length
```

---

## Step 7 Lane Prediction

得到每个 anchor 的车道线预测结果。

---

# 3. Training Stage (训练阶段)

在训练过程中，需要进行 GT 匹配和损失计算。

---

## Step 8 Line IoU Matching

使用车道线 IoU 进行 Anchor 与 GT 的匹配：

```
Anchor
↓
Line IoU
↓
Matched GT
```

---

## Step 9 Dynamic Label Assignment

根据预测质量动态分配正样本 Anchor：

```
Matched Anchors
↓
Dynamic Label Assignment
↓
Positive Anchors
```

---

## Step 10 Loss Computation

计算损失函数：

```
Classification Loss
Regression Loss
```

---

# 4. Inference Stage (推理阶段)

推理阶段流程：

```
Input Image
↓
Backbone + FPN
↓
Anchor Feature Pooling
↓
Prediction Head
↓
Lane Prediction
↓
Final Lane Output
```

---

# Summary Pipeline

完整技术路径可以总结为：

```
Initialization
↓
Generate Ray Anchors

Forward
↓
Input Image
↓
Backbone (ResNet)
↓
FPN
↓
Anchor Feature Pooling
↓
Feature Enhancement
↓
Prediction Head
↓
Lane Prediction

Training
↓
Line IoU Matching
↓
Dynamic Label Assignment
↓
Loss
```

---

# Instruction for Code Alignment

请扫描当前项目代码，并完成以下任务：

1. 根据代码结构生成当前项目的 pipeline。
2. 将当前项目 pipeline 与上述论文 pipeline 进行对齐。
3. 找出每个论文模块在代码中的对应实现。
4. 标出：
   - 缺失模块
   - 实现顺序不一致模块
   - 可能影响训练效果的模块。