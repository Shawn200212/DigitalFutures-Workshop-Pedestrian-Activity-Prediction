# 基于建成环境的街道行人活动强度图神经网络预测

**SVI-Supervised Pedestrian Intensity Prediction with Inductive GraphSAGE**

> Predicting street-level pedestrian activity intensity in Shanghai using only built-environment,
> transit-accessibility and road-network-topology features. Street View Imagery (SVI) is used **only
> as a supervision signal** (YOLO11 person detection + VLM quality audit), never as a model input —
> so the trained model can genuinely infer pedestrian intensity for streets *without* any SVI
> coverage (residential inner roads, branch roads, alleys).

---

## 项目简介

本项目构建了一个基于 **GraphSAGE 的归纳式图神经网络**：以路段为节点、路网拓扑为边，仅使用
**建成环境 + 交通可达性 + 路网拓扑** 共 60 维自动化特征，预测城市街道的行人活动强度相对水平。

核心设计原则：

- **归纳式学习（Inductive）**——训练图与测试图在拓扑上完全独立，模型学习的是跨空间可迁移的邻域聚合函数；
- **街景仅作标签，不作特征**——街景经 YOLO11 行人检测 + VLM 复核后生成路段级活动强度标签，仅为有街景覆盖的核心节点提供监督信号，因此模型对**无街景区域具备真正的推断能力**；
- **零人工语义标注**——不识别道路等级、不标注出入口、不建超节点，全部依赖几何计算与自动特征工程；
- **空间瓦片化**——上海外环内切分为 100 个 1.5km×1.5km 独立图样本（含 800m 缓冲区幽灵节点），图级划分训练/验证/测试集。

## 成果预览

| 热点概率空间分布（y≥5 二分类） | 回归预测 vs 真实值 |
| :---: | :---: |
| ![hotspot](docs/figures/fig_spatial_hotspot_probability.png) | ![pred](docs/figures/fig_pred_vs_true.png) |

| 离散化空间分类结果 | Top-20 特征重要性 |
| :---: | :---: |
| ![spatial](docs/figures/fig_spatial_class.png) | ![importance](docs/figures/fig_feature_importance_top20.png) |

更多诊断图（训练面板、误差分箱等）见 `docs/figures/`。

## 技术管线

```
阶段一：标签生产（街景 → 路段级监督标签）
  街景全景图（四正交视角投影）
    → YOLO11 行人计数（app/batch_detect.py）
    → YOLO11 姿态估计（app/batch_pose.py）
    → 检测/姿态合并（app/merge_detect_pose.py）
    → 自动 QC 路由：auto-accept / VLM 复核 / 人工（scripts/qc/auto_qc_router.py）
    → SiliconFlow VLM 审核（scripts/qc/vlm_audit_images_siliconflow.py）
    → 最终图片标签解析（scripts/qc/resolve_final_counts.py）
    → 按经纬度 snap 至最近路段并聚合（scripts/qc/aggregate_final_to_segment_labels.py）

阶段二：图学习（建成环境 → 行人活动强度）
  路网打断成路段节点 + 几何交叉建边
    → 60 维特征：道路本体(3) + 三级缓冲建筑环境(27) + 交通可达(6) + 人口(6) + sDNA 拓扑(18)
    → GraphSAGE 回归（scripts/gnn/train_graphsage_npz.py）
    → GraphSAGE 热点分类（scripts/gnn/train_graphsage_classifier_npz.py）
    → 5-fold 图级交叉验证 / 超参数扫描（scripts/gnn/run_graphsage_cv_sweep.py）
    → 特征重要性 / 后处理离散化 / 结果可视化
```

## 仓库结构

```
├── app/                      # YOLO11 批量检测、姿态估计、结果合并
├── scripts/
│   ├── gnn/                  # GraphSAGE 训练、CV 扫描、特征重要性、后处理
│   ├── qc/                   # QC 路由、VLM 审核、标签解析与路段聚合
│   └── ...                   # manifest 构建、增量批次、全瓦片编排
├── GraphSAGE_行人强度可视化_便携版_*/   # 离线交互式可视化网页（2D/3D，双击即用）
├── docs/
│   ├── WORKFLOW.md           # 完整命令行工作流（原 README）
│   └── figures/              # 成果图
├── 上海城市街道行人活动强度图神经网络预测_Demo技术大纲_0702.md   # 技术大纲全文
└── requirements.txt
```

> **注意**：`data/`（约 49 GB 街景瓦片）、`dataset/`（预构建图样本 .npz）、`outputs/`（训练结果）
> 与 YOLO 模型权重（`*.pt`）体积过大，不随仓库分发。目录结构与复现方法见
> [docs/WORKFLOW.md](docs/WORKFLOW.md)。

## 快速开始

```bash
pip install -r requirements.txt
# GNN 训练额外需要 PyTorch Geometric（按 CUDA 版本安装）
pip install torch_geometric
```

标签生产管线与训练的完整命令（含当前最优超参数配置）见 **[docs/WORKFLOW.md](docs/WORKFLOW.md)**，
100 瓦片数据接入说明见 **[README_100tile_yolo_vlm_pipeline.md](README_100tile_yolo_vlm_pipeline.md)**。

最小示例——训练 GraphSAGE 回归：

```bash
python scripts/gnn/train_graphsage_npz.py \
  --dataset-root dataset/SHdataset \
  --tile-start 1 --tile-end 100 \
  --hidden-dim 64 --num-layers 2 --dropout 0.15 \
  --loss huber \
  --feature-transform skew_log1p_clip --feature-clip-quantile 0.999 \
  --target-transform log1p_zscore \
  --device cuda
```

VLM 审核需要设置环境变量 `SILICONFLOW_API_KEY`（仓库中不含任何密钥）。

## 交互式可视化

`GraphSAGE_行人强度可视化_便携版_*` 是一个零依赖的离线网页：双击 `index.html` 即可浏览
2D 俯视预测图与 3D 沉浸式结果，支持预测/真值/残差/热点等多图层切换、tile 与 split 筛选、
节点悬停详情、变量影响展示与 PNG 导出。

## 应用前景

1. **规划阶段预判**——仅凭建筑方案 + 路网 + 站点选址，即可输出规划路段的行人活力热力图，把活力评估从"建成后调研"前移到"方案比选"；
2. **方案行人友好性对比**——量化"功能—形态—人流"耦合关系，支持步行优先的设计决策；
3. **活力洼地筛查**——"模型说这里理应人多、实际却冷清"的路段，正是值得调查干预的对象；
4. **低数据城市迁移**——冻结特征提取层、用少量本地标签微调回归头，即可迁移到只有路网/建筑/站点数据的城市。

## Roadmap

- [ ] 整理 `dataset/SHdataset` 样例子集（1–2 个瓦片）供快速复现
- [ ] 发布训练好的模型权重与 normalizer
- [ ] 可视化网页在线部署（GitHub Pages）
- [ ] 英文完整文档

## 关于作者

中央美术学院设计学院 · 沉浸体验与游戏设计研究方向硕士在读。
研究兴趣：游戏引擎实时创作 × 具身智能 × 城市数据。

---

*本仓库为研究 Demo，代码与文档持续整理中，欢迎 Issue 交流。*
