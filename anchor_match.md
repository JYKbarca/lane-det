## 核心改动概述
### A. 引入 Dynamic Label Assignment（强制 Top-K）
对每条 GT：
1) 计算所有 anchors 的 IoU（使用现有 Line IoU）
2) 取 IoU 最大的 top_k 个 anchors（top_k = 5~10）
3) 将这些 anchors 标为正样本（force positive），前提是 IoU >= min_pos_iou（如 0.05~0.10）
4) 若满足条件的不足 top_k，则使用“全量 top_k”作为 fallback（避免 GT 无正样本）

说明：
- Top-K 只负责“保底 + 提升正样本密度”
- 仍可保留全局 best_iou 的阈值机制作为补充，但优先保证 GT 覆盖

### B. 前置筛选从“硬作废”改为“软惩罚”
目前做法：不满足 common/top/geo 条件 → IoU=-1
建议：改成 penalty（将 IoU 乘一个系数），而不是直接 -1

示例：
- common_ratio 不足：IoU *= 0.5
- top_mean_err 超阈值：IoU *= 0.7
- angle 过大：IoU *= 0.6
只在极端情况下才作废（例如 common_count=0）

### C. 缩小灰区（减少 ignore）
将 neg_thr 提高一些，pos_thr 适当降低一些：
- 建议起点：pos_thr_bottom 0.30（原 0.38）
- pos_thr_side 0.22~0.25（原 0.27）
- neg_thr 0.15~0.20（原 0.10）
让更多样本成为明确 neg，减少 ignore 对 cls 的“样本浪费”。

注意：
- cls loss 本身使用 focal loss，neg 多并不致命
- 可配合 hard negative mining 限制参与训练的 neg 数量（可选）

## 代码改动点
文件：lane_det/anchors/label_assigner.py

新增/修改：
1) 在 assign(...) 中新增 per-GT 分配逻辑：
   - 构建 IoU 矩阵 [num_gt, num_anchors]
   - 对每条 GT 取 top_k anchors，并写入 force_pos_mask

2) 修改 gating：
   - 将 “置 -1 作废” 替换为 “IoU *= penalty”
   - 仅在 common_count=0 等极端情况置 -1

3) 标签合成规则：
   - 先根据 force_pos_mask 标记 pos
   - 再对剩余 anchors 按 best_iou 与阈值分配 pos/neg/ignore

## Debug 输出（必须加）
在 tools/vis_dataset.py --show_match 和训练前统计中，输出：
- per-GT max_iou 分布：mean / p50 / p90 / min
- per-GT 分到的 pos 数量：mean / min / max
- ignore 的来源计数（按 gating 失败原因统计）

## 可配置参数（写进 yaml）
新增建议字段：
- match.topk_per_gt: 5
- match.min_force_pos_iou: 0.08
- match.use_soft_gating: true
- match.penalty_common_ratio: 0.5
- match.penalty_top_err: 0.7
- match.penalty_angle: 0.6
- match.pos_thr_bottom: 0.30
- match.pos_thr_side: 0.24
- match.neg_thr: 0.18

## 预期效果
- pos 数量：显著上升（通常 5~15 倍）
- cls loss：不再一开始塌到 0
- 可视化匹配：中部/远端车道线能找到候选 anchor
- 最终预测：贴合度提升、漏检减少