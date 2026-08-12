# 阶段 0——街景图像采集

[English](README.md) | **中文**

该模块位于 [app](../app) 与 [scripts](../scripts) 的上游：把道路采样点转换为全景图和透视视图，
为 GraphSAGE 模型提供**标签**。它与研究流水线保持自包含和隔离：使用独立依赖，不跨边界导入，
也不依赖模型代码。运行仓库其他部分不要求先运行本模块。

> **范围说明：**图像只用于监督，不进入特征空间（见根目录[中文 README](../README.zh-CN.md)）。
> 仓库不分发街景图像，本目录只包含采集代码和采样点。

## 内容

```text
data_acquisition/
├── stvscrawler/            # 百度街景客户端
├── flat_images_pano.py     # 全景图 → 4 × 720×720 透视图
├── run_all_tiles.py        # 100 个分块批处理与 MASTER 清单
├── tiles/tile_001..100.csv # 每个分块的 WGS-84 采样点
└── requirements.txt
```

`tiles/*.csv` 覆盖上海外环内的 100 个 1.5 km × 1.5 km 核心分块，共约 0.6 MB 经纬度采样点，
不包含个人数据。

## 工作方式

1. **采样点到全景图：**将 WGS-84 转为百度墨卡托，解析最近的街景 `sid` 与拍摄日期。
2. **日期选择：**`TimeFuncs` 选择最接近 `TARGET_DATES` 的拍摄时间，使标签来自可比较窗口。
3. **一次下载，四个视图：**每个点与日期只下载一次全景图，再本地投影为四个正交视图，
   比分别请求四张视图减少约 67% 网络请求。
4. **批处理与续跑：**按序遍历 `tile_*.csv`，每个分块写入独立目录，并在每个分块后刷新
   MASTER 清单，使中断任务可恢复，也可从现有日志重建汇总表。

`MASTER_flat_image_manifest_for_training.csv` 是交接点，由
[prepare_yolo_manifest_from_tiles.py](../scripts/prepare_yolo_manifest_from_tiles.py) 读取并启动阶段 1。

## 使用

```bash
pip install -r data_acquisition/requirements.txt
set SVI_DATA_ROOT=D:/svi/images
python data_acquisition/run_all_tiles.py
```

默认输出到 `<repo>/data/images/`（已加入 gitignore），可通过 `SVI_DATA_ROOT` 改到其他磁盘。
配置块包含 `TARGET_DATES`、`Z_LEVEL`、`FLAT_IMG_SIZE`、`DIRECTIONS`、`YAW_OFFSET`、
续跑和去重策略。百度全景内部方向相对 API 约偏移 90°；只有确认自己的数据偏移后才应修改
`YAW_OFFSET = 90`。

## 来源与使用条款

- `stvscrawler/` 改编自第三方开源百度街景爬虫代码；所依据副本没有上游作者头，因此按衍生作品
  收录，而非声明为原创。如你是原作者，请提交 Issue，项目将补充适当署名。
- `flat_images_pano.py`、`run_all_tiles.py` 与分块采样点由本项目编写。
- 客户端调用未公开的百度端点，仅供学术研究。运行者必须遵守数据提供方条款并控制请求速率。
- 不需要也不保存 API 密钥或凭据；仓库也不分发下载图像。
