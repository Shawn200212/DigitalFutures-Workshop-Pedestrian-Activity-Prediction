"""
flat_images_pano.py

用途：
    基于全景图（equirectangular）批量下载百度街景，并在本地投影生成
    4 个方向（0°/90°/180°/270°）的 720×720 透视图。

核心改进：
    1) 每个点每个目标日期只下载一次全景图，避免重复查询 sid/stv_info；
    2) 本地完成等距柱状投影 → 透视投影，减少网络请求约 67%；
    3) 最终 4 张图的命名、目录结构与 flat_images.py 完全一致；
    4) 同时保留原始 360° 全景图，便于后续标准化流程使用。

保存结构：
    root/
      actual_YYYY_MM/
        2019_01.121.40123456.31.23010000.dir_000.png
        2019_01.121.40123456.31.23010000.dir_090.png
        2019_01.121.40123456.31.23010000.dir_180.png
        2019_01.121.40123456.31.23010000.dir_270.png
      panorama_YYYY_MM/
        2019_01.121.40123456.31.23010000.pano.png
"""

import os
import re
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import map_coordinates

# 进度条：优先使用 tqdm；未安装时使用简易回退。
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

    class _FallbackTqdm:
        def __init__(self, total=None, desc=None, **kwargs):
            self.total = total or 0
            self.n = 0
            self.desc = desc or ""

        def update(self, k=1):
            self.n += k
            if self.total:
                pct = 100.0 * self.n / self.total
                print(
                    f"\r{self.desc} {self.n}/{self.total} ({pct:5.1f}%)",
                    end="",
                    flush=True,
                )

        def set_description(self, desc=""):
            self.desc = desc

        def set_postfix(self, **kwargs):
            pass

        def close(self):
            print()

        @staticmethod
        def write(msg=""):
            print(msg)

    def tqdm(*args, **kwargs):
        return _FallbackTqdm(*args, **kwargs)

    tqdm.write = _FallbackTqdm.write

from stvscrawler import StreetViewScrawler as svs
from stvscrawler import TimeFuncs
from stvscrawler.wgs2bd import wgs2bdmc


# ============================================================
# 1. 参数区：你主要修改这里
# ============================================================

input_csv = "tiles/tile_001.csv"

# 影像输出根目录。
# 默认写到仓库根目录下的 data/images/（已被 .gitignore 排除，不会入库）。
# 也可以用环境变量覆盖：set SVI_DATA_ROOT=D:/somewhere/images
out_dir = Path(
    os.environ.get(
        "SVI_DATA_ROOT",
        Path(__file__).resolve().parent.parent / "data" / "images",
    )
)

# 全景图缩放等级。
# z=3 -> 2048×1024（用户所谓“原片 2048”）
# z=4 -> 4096×2048
# z=5 -> 8192×4096
z_level = 3

# 输出透视图参数
flat_img_size = 720
flat_fov_x = 90            # 水平视场角（度）
directions = [0, 90, 180, 270]

# 全景图相对道路方向的偏航补偿。
# 经测试：百度全景图内部方向与 API pr3d 方向存在约 90° 系统偏移，
# 即 API 的 direction=0（道路前进方向）对应全景图中 yaw=90°。
yaw_offset = 90

n_tries = 3

target_dates = [
    "201703", "201706", "201709", "201712",
    "201803", "201806", "201809", "201812",
    "201903", "201906", "201909", "201912",
]

date_mode = "nearest"
coord_decimals = 8
filename_style = "compact"
duplicate_policy = "skip"
sort_points_by_lon_lat = True

# 是否同时保存 360° 全景图原片
save_panorama = True

# 新增参数
verbose = True
enable_resume = True
log_flush_every = 5


# ============================================================
# 2. 日期选择器
# ============================================================

def exact_month(timelines, target_date=None):
    target_date = str(target_date)
    for timeline in timelines:
        if str(timeline.get("TimeLine")) == target_date:
            return timeline
    return []


def get_selector_and_targets(mode, target_dates):
    mode = str(mode).lower().strip()

    if mode == "nearest":
        return TimeFuncs.Nearest, list(map(str, target_dates))

    if mode == "exact":
        return exact_month, list(map(str, target_dates))

    if mode == "latest":
        return TimeFuncs.Latest, ["latest"]

    if mode == "earliest":
        return TimeFuncs.Earliest, ["earliest"]

    raise ValueError("date_mode 只能是 nearest、exact、latest 或 earliest")


# ============================================================
# 3. 投影函数：等距柱状全景图 → 透视图
# ============================================================

def equirect_to_perspective(
    equi_img: Image.Image,
    fov_x_deg: float,
    out_w: int,
    out_h: int,
    yaw_deg: float,
    pitch_deg: float = 0,
    roll_deg: float = 0,
) -> Image.Image:
    """
    将等距柱状（equirectangular）全景图转换为透视图。

    参数：
        equi_img: PIL Image, RGB
        fov_x_deg: 水平视场角（度）
        out_w, out_h: 输出图像尺寸
        yaw_deg: 水平偏航角（度）
        pitch_deg: 俯仰角（度），正值向上看
        roll_deg: 滚转角（度）
    """
    W, H = equi_img.size
    arr = np.array(equi_img).astype(np.float32) / 255.0

    fov_x = math.radians(fov_x_deg)
    f = out_w / (2 * math.tan(fov_x / 2))

    u = np.arange(out_w)
    v = np.arange(out_h)
    u, v = np.meshgrid(u, v)

    # 以图像中心为原点，相机坐标系：z 向前，x 向右，y 向上
    x = u - out_w / 2
    y = out_h / 2 - v
    z = np.full_like(x, f)

    norm = np.sqrt(x**2 + y**2 + z**2)
    x_cam = x / norm
    y_cam = y / norm
    z_cam = z / norm

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    # roll（绕 z 轴）
    if roll != 0:
        x_r = x_cam * math.cos(roll) - y_cam * math.sin(roll)
        y_r = x_cam * math.sin(roll) + y_cam * math.cos(roll)
        x_cam, y_cam = x_r, y_r

    # pitch（绕 x 轴）
    if pitch != 0:
        y_p = y_cam * math.cos(pitch) - z_cam * math.sin(pitch)
        z_p = y_cam * math.sin(pitch) + z_cam * math.cos(pitch)
        y_cam, z_cam = y_p, z_p

    # yaw（绕 y 轴）
    x_w = x_cam * math.cos(yaw) + z_cam * math.sin(yaw)
    z_w = -x_cam * math.sin(yaw) + z_cam * math.cos(yaw)
    x_cam, z_cam = x_w, z_w

    # 转换到球坐标
    theta = np.arctan2(x_cam, z_cam)            # 水平角，范围 [-pi, pi]
    phi = np.arcsin(np.clip(y_cam, -1, 1))      # 垂直角，范围 [-pi/2, pi/2]

    # 映射回等距柱状图坐标
    eq_u = (theta / math.pi + 1) / 2 * W
    eq_v = (0.5 - phi / math.pi) * H

    # 双线性插值（水平方向 wrap）
    coords = np.stack([eq_v, eq_u], axis=0)
    out_r = map_coordinates(arr[:, :, 0], coords, order=1, mode="wrap")
    out_g = map_coordinates(arr[:, :, 1], coords, order=1, mode="wrap")
    out_b = map_coordinates(arr[:, :, 2], coords, order=1, mode="wrap")

    out = np.stack([out_r, out_g, out_b], axis=-1)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# ============================================================
# 4. 文件夹和文件名工具
# ============================================================

def normalize_actual_date_for_path(actual_date):
    if actual_date is None or pd.isna(actual_date):
        return "actual_unknown", "unknown"

    s = str(actual_date).strip()
    digits = re.sub(r"\D", "", s)

    if len(digits) >= 6:
        yyyy = digits[:4]
        mm = digits[4:6]
        file_prefix = f"{yyyy}_{mm}"
        return f"actual_{file_prefix}", file_prefix

    safe = re.sub(r'[\\/:*?"<>|]+', "_", s)
    safe = re.sub(r"\s+", "_", safe)

    if safe == "":
        safe = "unknown"

    return f"actual_{safe}", safe


def format_coord(value, decimals=8):
    return f"{float(value):.{decimals}f}"


def build_base_filename(actual_date, lon, lat, direction=None, suffix=""):
    _, date_prefix = normalize_actual_date_for_path(actual_date)

    lon_s = format_coord(lon, coord_decimals)
    lat_s = format_coord(lat, coord_decimals)

    if filename_style == "compact":
        if direction is not None:
            return f"{date_prefix}.{lon_s}.{lat_s}.dir_{int(direction):03d}.png"
        return f"{date_prefix}.{lon_s}.{lat_s}.{suffix}.png"

    if filename_style == "labeled":
        if direction is not None:
            return f"{date_prefix}.lon_{lon_s}.lat_{lat_s}.dir_{int(direction):03d}.png"
        return f"{date_prefix}.lon_{lon_s}.lat_{lat_s}.{suffix}.png"

    raise ValueError("filename_style 只能是 compact 或 labeled")


def _norm_path(path):
    return os.path.normcase(os.path.abspath(str(path)))


def _path_exists(path, existing_index=None):
    if existing_index is not None:
        return _norm_path(path) in existing_index
    return Path(path).exists()


def resolve_duplicate_path(path, existing_index=None):
    if not _path_exists(path, existing_index):
        return path, "new"

    if duplicate_policy == "skip":
        return path, "skip_existing"

    if duplicate_policy == "overwrite":
        return path, "overwrite_existing"

    if duplicate_policy == "suffix":
        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        idx = 1
        while True:
            candidate = parent / f"{stem}__dup{idx:03d}{suffix}"
            if not _path_exists(candidate, existing_index):
                return candidate, "suffix_duplicate"
            idx += 1

    raise ValueError("duplicate_policy 只能是 skip、suffix 或 overwrite")


# ============================================================
# 5. 点位读取
# ============================================================

def load_points(csv_path):
    df = pd.read_csv(csv_path)

    required = {"iid", "lon", "lat"}
    if not required.issubset(set(df.columns)):
        first_three = list(df.columns[:3])

        if len(first_three) < 3:
            raise ValueError("CSV 至少需要三列：iid, lon, lat")

        df = df.rename(columns={
            first_three[0]: "iid",
            first_three[1]: "lon",
            first_three[2]: "lat",
        })

    df["iid"] = df["iid"].astype(int)
    df["lon"] = df["lon"].astype(float)
    df["lat"] = df["lat"].astype(float)

    if sort_points_by_lon_lat:
        df = df.sort_values(
            by=["lon", "lat", "iid"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    return df


# ============================================================
# 6. skip 优化相关工具
# ============================================================

def build_existing_file_index(root):
    """启动时扫描一次输出目录，建立 .png 文件的内存索引。"""
    index = set()
    if Path(root).exists():
        for p in Path(root).rglob("*.png"):
            index.add(_norm_path(p))
    return index


def build_resume_index(log_path, existing_files):
    """
    读取上一次日志，构造“已完成”键集合。
    key = (iid, direction_or_pano, target_date)
    """
    done_keys = set()
    prior_rows_by_key = {}

    if not Path(log_path).exists():
        return done_keys, prior_rows_by_key

    try:
        prior_df = pd.read_csv(log_path, encoding="utf-8-sig")
    except Exception:
        return done_keys, prior_rows_by_key

    columns = list(prior_df.columns)
    needed = {"iid", "target_date", "save_path", "success"}
    if not needed.issubset(set(columns)):
        return done_keys, prior_rows_by_key

    for _, r in prior_df.iterrows():
        success = r["success"]
        if isinstance(success, str):
            success = success.strip().lower() in ("true", "1", "yes")
        if not bool(success):
            continue

        save_path = "" if pd.isna(r["save_path"]) else str(r["save_path"])
        if save_path == "":
            continue
        if _norm_path(save_path) not in existing_files:
            continue

        try:
            iid = int(r["iid"])
            target_date = str(r["target_date"])
            # 用 direction 或 is_pano 区分键
            direction = r.get("direction")
            is_pano = r.get("is_pano", False)
            if isinstance(is_pano, str):
                is_pano = is_pano.strip().lower() in ("true", "1", "yes")
            key_dir = "pano" if is_pano else int(direction)
        except Exception:
            continue

        key = (iid, key_dir, target_date)
        done_keys.add(key)
        prior_rows_by_key[key] = {col: r[col] for col in columns}

    return done_keys, prior_rows_by_key


def flush_log(log_records, log_path):
    """把当前累积的日志整体写盘（覆盖）。"""
    pd.DataFrame(log_records).to_csv(log_path, index=False, encoding="utf-8-sig")


# ============================================================
# 7. 工具：按实际日期对 target_dates 分组
# ============================================================

def group_dates_by_actual(scrawler, lon, lat, selector, run_dates):
    """
    对每个点只查询一次时间线，然后把 run_dates 按实际拍摄日期分组。
    返回：dict[actual_date_str] = {
        'udt_id': str,
        'target_dates': [str, ...],
    }
    如果该点无街景或查询失败，返回空 dict。
    """
    try:
        bdx, bdy = wgs2bdmc(lon, lat)
    except Exception as e:
        return {}, f"wgs2bdmc error: {e}"

    sid, sid_info = None, None
    for try_count in range(1, n_tries + 1):
        try:
            sid, sid_info = scrawler.xy2sid(bdx, bdy)
            break
        except Exception as e:
            if try_count == n_tries:
                return {}, f"xy2sid failed: {e}"
            time.sleep(0.5)

    if sid == -1 or sid is None:
        return {}, sid_info

    try:
        stv_info = scrawler.sid2all_stv_info(sid)
        timelines = stv_info.get("time_lines", [])
        if not timelines:
            return {}, "no timelines"
    except Exception as e:
        return {}, f"sid2all_stv_info failed: {e}"

    groups = {}
    for target_date in run_dates:
        try:
            selected = selector(timelines, target_date)
            if not selected:
                continue
            actual_date = str(selected.get("TimeLine"))
            udt_id = str(selected.get("ID"))
            if actual_date not in groups:
                groups[actual_date] = {"udt_id": udt_id, "target_dates": []}
            groups[actual_date]["target_dates"].append(str(target_date))
        except Exception:
            continue

    return groups, "success"


# ============================================================
# 8. 主程序
# ============================================================

def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    points_df = load_points(input_csv)
    selector, run_dates = get_selector_and_targets(date_mode, target_dates)

    # 基础 scrawler：仅用于查询时间线和下载全景图瓦片，不依赖 img_date
    base_scrawler = svs(
        date_selector=TimeFuncs.Latest,
        z_level=z_level,
        flat_img_size=flat_img_size,
    )

    log_path = out_dir / "crawl_log_flat_actualdate_training_review.csv"

    existing_files = build_existing_file_index(out_dir)
    if enable_resume:
        done_keys, prior_rows_by_key = build_resume_index(log_path, existing_files)
    else:
        done_keys, prior_rows_by_key = set(), {}

    log_records = []
    total_points = len(points_df)

    print("输出结构：")
    print("root/")
    print("  actual_YYYY_MM/")
    print("    YYYY_MM.longitude.latitude.dir_000.png")
    print("  panorama_YYYY_MM/")
    print("    YYYY_MM.longitude.latitude.pano.png")
    print()
    print(f"date_mode = {date_mode}")
    print(f"duplicate_policy = {duplicate_policy}")
    print(f"filename_style = {filename_style}")
    print(f"z_level = {z_level}（全景图尺寸约 {2**z_level * 256} x {2**(z_level-1) * 256}）")
    print(f"flat_img_size = {flat_img_size}")
    print(f"flat_fov_x = {flat_fov_x}")
    print(f"yaw_offset = {yaw_offset}")
    print(f"points = {total_points}")
    print(f"run_dates = {run_dates}")
    print(f"directions = {directions}")
    print(f"enable_resume = {enable_resume}（已识别可跳过下载的键 = {len(done_keys)}）")
    print(f"verbose = {verbose}, log_flush_every = {log_flush_every}, tqdm = {_HAS_TQDM}")
    print()

    counters = {
        "saved_pano": 0,
        "saved_flat": 0,
        "skip": 0,
        "empty": 0,
        "error": 0,
        "resume": 0,
        "out_of_range": 0,
    }

    pbar = tqdm(total=total_points, desc="爬取进度(按点)", unit="pt", dynamic_ncols=True)

    for point_no, (row_index, row) in enumerate(points_df.iterrows(), start=1):
        iid = int(row["iid"])
        lon = float(row["lon"])
        lat = float(row["lat"])

        pbar.set_description(f"按点爬取 {point_no}/{total_points} iid={iid}")

        # 一次性查询该点的时间线，并按实际日期分组
        st_query = time.time()
        date_groups, query_status = group_dates_by_actual(
            base_scrawler, lon, lat, selector, run_dates
        )
        query_cost = time.time() - st_query

        if not date_groups:
            counters["empty"] += 1
            if verbose:
                tqdm.write(
                    f"[empty] iid={iid}, "
                    f"lon={lon:.{coord_decimals}f}, "
                    f"lat={lat:.{coord_decimals}f}, "
                    f"reason={query_status}, "
                    f"cost={query_cost:.2f}s"
                )
            for target_date in run_dates:
                log_records.append({
                    "row_index": row_index,
                    "iid": iid,
                    "lon": lon,
                    "lat": lat,
                    "target_date": target_date,
                    "actual_date": "",
                    "actual_folder": "",
                    "filename": "",
                    "save_path": "",
                    "sid": "",
                    "udt_sid": "",
                    "direction": "",
                    "is_pano": True,
                    "success": False,
                    "status": f"empty: {query_status}",
                    "duplicate_status": "",
                    "cost_seconds": round(query_cost, 2),
                    "manual_keep": "",
                    "manual_note": "",
                })
            pbar.update(1)
            continue

        # 每个实际日期只下载一次全景图
        for actual_date, group_info in date_groups.items():
            udt_id = group_info["udt_id"]
            group_target_dates = group_info["target_dates"]

            actual_year = int(str(actual_date)[:4])
            if actual_year < 2017 or actual_year > 2019:
                counters["out_of_range"] += 1
                if verbose:
                    tqdm.write(
                        f"[out_of_range] iid={iid}, actual_year={actual_year}, "
                        f"targets={group_target_dates}"
                    )
                for target_date in group_target_dates:
                    log_records.append({
                        "row_index": row_index,
                        "iid": iid,
                        "lon": lon,
                        "lat": lat,
                        "target_date": target_date,
                        "actual_date": actual_date,
                        "actual_folder": "",
                        "filename": "",
                        "save_path": "",
                        "sid": "",
                        "udt_sid": udt_id,
                        "direction": "",
                        "is_pano": True,
                        "success": False,
                        "status": "out_of_year_range",
                        "duplicate_status": "",
                        "cost_seconds": "",
                        "manual_keep": "",
                        "manual_note": "",
                    })
                continue

            folder_name, _ = normalize_actual_date_for_path(actual_date)
            actual_dir = out_dir / folder_name
            panorama_dir = out_dir / folder_name.replace("actual_", "panorama_")
            actual_dir.mkdir(parents=True, exist_ok=True)
            if save_panorama:
                panorama_dir.mkdir(parents=True, exist_ok=True)

            # 检查该实际日期的全景图是否已存在（断点续跑）
            pano_filename = build_base_filename(
                actual_date=actual_date,
                lon=lon,
                lat=lat,
                suffix="pano",
            )
            pano_path = panorama_dir / pano_filename

            # 如果全景图文件已存在，直接加载；否则下载
            st_pano = time.time()
            pano = None
            pano_status = None
            pano_downloaded = False

            if _path_exists(pano_path, existing_files):
                try:
                    pano = Image.open(pano_path)
                    pano_status = "skipped_existing"
                    counters["skip"] += 1
                    pano_downloaded = False
                except Exception as e:
                    if verbose:
                        tqdm.write(f"[warn] 加载已存在全景图失败：{pano_path}, err={e}")
                    pano = None

            if pano is None:
                try:
                    pano_data = base_scrawler.udt_id2panorama(udt_id, n_tries=n_tries)
                    pano = pano_data.image
                    pano_downloaded = True
                except Exception as e:
                    counters["error"] += 1
                    if verbose:
                        tqdm.write(
                            f"[error] iid={iid}, actual={actual_date}, "
                            f"udt_id={udt_id}, err={repr(e)}"
                        )
                    for target_date in group_target_dates:
                        log_records.append({
                            "row_index": row_index,
                            "iid": iid,
                            "lon": lon,
                            "lat": lat,
                            "target_date": target_date,
                            "actual_date": actual_date,
                            "actual_folder": "",
                            "filename": "",
                            "save_path": "",
                            "sid": "",
                            "udt_sid": udt_id,
                            "direction": "",
                            "is_pano": True,
                            "success": False,
                            "status": f"error: {repr(e)}",
                            "duplicate_status": "",
                            "cost_seconds": round(time.time() - st_pano, 2),
                            "manual_keep": "",
                            "manual_note": "",
                        })
                    continue

            if pano is None:
                counters["empty"] += 1
                for target_date in group_target_dates:
                    log_records.append({
                        "row_index": row_index,
                        "iid": iid,
                        "lon": lon,
                        "lat": lat,
                        "target_date": target_date,
                        "actual_date": actual_date,
                        "actual_folder": "",
                        "filename": "",
                        "save_path": "",
                        "sid": "",
                        "udt_sid": udt_id,
                        "direction": "",
                        "is_pano": True,
                        "success": False,
                        "status": "empty_or_no_matching_date",
                        "duplicate_status": "",
                        "cost_seconds": round(time.time() - st_pano, 2),
                        "manual_keep": "",
                        "manual_note": "",
                    })
                continue

            # 保存全景图（如果需要）
            if save_panorama and pano_status != "skipped_existing":
                pano_save_path, pano_dup_status = resolve_duplicate_path(
                    pano_path, existing_files
                )
                if pano_dup_status == "skip_existing":
                    pano_status = "skipped_existing"
                    counters["skip"] += 1
                else:
                    pano.save(pano_save_path)
                    existing_files.add(_norm_path(pano_save_path))
                    pano_status = "saved" if pano_dup_status == "new" else pano_dup_status
                    counters["saved_pano"] += 1
            else:
                pano_save_path = pano_path
                pano_dup_status = "skip_existing" if pano_status == "skipped_existing" else "new"

            pano_cost = time.time() - st_pano

            # 为该实际日期下的所有 target_date 生成本地投影
            pano_first_target = True
            for target_date in group_target_dates:
                # 后续 target_date 的全景图状态应显示为已存在
                if not pano_first_target and pano_status == "saved":
                    pano_log_status = "skipped_existing"
                else:
                    pano_log_status = pano_status
                pano_first_target = False

                pano_key = (iid, "pano", str(target_date))
                if pano_key in done_keys:
                    prior = prior_rows_by_key.get(pano_key)
                    if prior is not None:
                        rec = dict(prior)
                        rec["row_index"] = row_index
                        rec["status"] = "skipped_resume_existing"
                        rec["duplicate_status"] = "skip_existing"
                        log_records.append(rec)
                    counters["resume"] += 1
                    counters["skip"] += 1
                else:
                    log_records.append({
                        "row_index": row_index,
                        "iid": iid,
                        "lon": lon,
                        "lat": lat,
                        "target_date": target_date,
                        "actual_date": actual_date,
                        "actual_folder": folder_name.replace("actual_", "panorama_"),
                        "filename": pano_save_path.name,
                        "save_path": str(pano_save_path),
                        "sid": "",
                        "udt_sid": udt_id,
                        "direction": "",
                        "is_pano": True,
                        "success": True,
                        "status": pano_status,
                        "duplicate_status": pano_dup_status,
                        "cost_seconds": round(pano_cost, 2),
                        "manual_keep": "",
                        "manual_note": "",
                    })

                if verbose and pano_key not in done_keys:
                    tqdm.write(
                        f"[{pano_status}] "
                        f"actual={actual_date}, iid={iid}, "
                        f"lon={lon:.{coord_decimals}f}, "
                        f"lat={lat:.{coord_decimals}f}, "
                        f"target={target_date}, "
                        f"pano={pano_save_path.name}"
                    )

                # 4 个方向透视图
                for direction in directions:
                    flat_key = (iid, int(direction), str(target_date))

                    if flat_key in done_keys:
                        prior = prior_rows_by_key.get(flat_key)
                        if prior is not None:
                            rec = dict(prior)
                            rec["row_index"] = row_index
                            rec["status"] = "skipped_resume_existing"
                            rec["duplicate_status"] = "skip_existing"
                            log_records.append(rec)
                        counters["resume"] += 1
                        counters["skip"] += 1
                        if verbose:
                            tqdm.write(
                                f"[skipped_resume] iid={iid}, "
                                f"target={target_date}, "
                                f"direction={int(direction):03d} (已存在)"
                            )
                        continue

                    st_dir = time.time()
                    yaw = (direction + yaw_offset) % 360

                    flat = equirect_to_perspective(
                        pano,
                        fov_x_deg=flat_fov_x,
                        out_w=flat_img_size,
                        out_h=flat_img_size,
                        yaw_deg=yaw,
                        pitch_deg=0,
                        roll_deg=0,
                    )

                    base_filename = build_base_filename(
                        actual_date=actual_date,
                        lon=lon,
                        lat=lat,
                        direction=direction,
                    )
                    candidate_path = actual_dir / base_filename
                    save_path, duplicate_status = resolve_duplicate_path(
                        candidate_path, existing_files
                    )

                    if duplicate_status == "skip_existing":
                        status = "skipped_existing"
                        counters["skip"] += 1
                    else:
                        flat.save(save_path)
                        existing_files.add(_norm_path(save_path))
                        status = "saved" if duplicate_status == "new" else duplicate_status
                        counters["saved_flat"] += 1

                    cost_dir = time.time() - st_dir

                    if verbose:
                        tqdm.write(
                            f"[{status}] "
                            f"actual={actual_date}, iid={iid}, "
                            f"lon={lon:.{coord_decimals}f}, "
                            f"lat={lat:.{coord_decimals}f}, "
                            f"target={target_date}, "
                            f"direction={int(direction):03d}, "
                            f"yaw={yaw:.1f}, "
                            f"cost={cost_dir:.2f}s"
                        )

                    log_records.append({
                        "row_index": row_index,
                        "iid": iid,
                        "lon": lon,
                        "lat": lat,
                        "target_date": target_date,
                        "actual_date": actual_date,
                        "actual_folder": folder_name,
                        "filename": save_path.name,
                        "save_path": str(save_path),
                        "sid": "",
                        "udt_sid": udt_id,
                        "direction": int(direction),
                        "is_pano": False,
                        "success": True,
                        "status": status,
                        "duplicate_status": duplicate_status,
                        "cost_seconds": round(cost_dir, 2),
                        "manual_keep": "",
                        "manual_note": "",
                    })

        # 一个“点”处理完：推进进度条 + 周期性落盘
        pbar.update(1)
        pbar.set_postfix(
            saved_pano=counters["saved_pano"],
            saved_flat=counters["saved_flat"],
            skip=counters["skip"],
            empty=counters["empty"],
            err=counters["error"],
            resume=counters["resume"],
        )

        if log_flush_every and (point_no % log_flush_every == 0):
            flush_log(log_records, log_path)

    pbar.close()

    # 最终写盘
    flush_log(log_records, log_path)

    print()
    print(f"日志已保存：{log_path}")
    print(
        "统计："
        f"saved_pano={counters['saved_pano']}, "
        f"saved_flat={counters['saved_flat']}, "
        f"skipped(total)={counters['skip']}, "
        f"resume_skipped={counters['resume']}, "
        f"empty={counters['empty']}, "
        f"error={counters['error']}, "
        f"out_of_year_range={counters['out_of_range']}"
    )
    print("人工筛选时可以直接在日志中的 manual_keep / manual_note 列做标记。")


if __name__ == "__main__":
    main()
