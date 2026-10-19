# -*- coding: utf-8 -*-
"""DEM 剖切对比演示截图 (真实 ATL03 + AWS Terrain Tiles DEM).

用法::

    C:\\Users\\ASUS\\miniconda3\\envs\\waterenv\\python.exe tests\\dem_compare_shot.py

会自动在轨迹上挑一段"地形起伏明显"的走廊, 拉取免密钥 DEM 瓦片,
开启「DEM 剖切对比」后截图到 tests/_out/。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from main import enable_windows_dpi_awareness, setup_matplotlib  # noqa: E402

enable_windows_dpi_awareness()
setup_matplotlib()

from icesat_viewer import geo                                    # noqa: E402
from icesat_viewer.app import IcesatViewerApp                    # noqa: E402
from icesat_viewer.corridor import CorridorGeometry              # noqa: E402
from icesat_viewer.dem import (TerrariumDem, compare_with_profile,  # noqa: E402
                               corridor_bounds, sample_along_corridor)

OUT = ROOT / "tests" / "_out"
REAL = Path(r"D:\Paper\Data\satellite image\ICESAT\Taihu"
            r"\ATL03_20250105065215_03082606_007_01.h5")
#: 候选走廊中心 (按顺序尝试, 选地形起伏最大的)
CANDIDATES = [(120.95, 41.55), (120.60, 40.10), (121.70, 43.30), (120.22, 31.18)]


def pump(app, seconds=0.4):
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.01)


def wait_for(app, cond, timeout=240, label=""):
    end = time.time() + timeout
    while time.time() < end:
        app.update()
        if cond():
            return True
        time.sleep(0.05)
    print("!! 超时:", label)
    return False


def pick_corridor(ds, lon0, lat0, span_deg=0.03):
    best = None
    for label, lon, lat, _c in ds.track_full_arrays():
        d = np.hypot((lon - lon0) * 95.0, (lat - lat0) * 111.0)
        i = int(np.argmin(d))
        if best is None or d[i] < best[0]:
            best = (float(d[i]), label, i, lon, lat)
    _, label, i, lon_arr, lat_arr = best
    half = max(int(span_deg / 0.0002), 30)          # ~20 m/段
    i0, i1 = max(i - half, 0), min(i + half, lon_arr.size - 1)
    return (float(lon_arr[i0]), float(lat_arr[i0])), (float(lon_arr[i1]), float(lat_arr[i1])), label


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    app = IcesatViewerApp(provider="esri_img")
    app.geometry("1520x960+40+20")
    pump(app, 1.0)
    app.load_file(str(REAL))
    wait_for(app, lambda: len(app.entries) > 0, 300, "加载数据")
    ds = app.entries[0].dataset

    # 选一段地形起伏明显的走廊
    chosen = None
    for (lon0, lat0) in CANDIDATES:
        p0, p1, label = pick_corridor(ds, lon0, lat0)
        cg = CorridorGeometry(p0, p1, 25.0)
        try:
            dem = TerrariumDem(corridor_bounds(cg), cache_dir=Path.home() / ".icesat_tdt_viewer" / "dem")
        except Exception as exc:
            print(f"  {label}@({lon0},{lat0}) DEM 不可用: {exc}")
            continue
        aa, zz = sample_along_corridor(dem, cg, step_m=10.0)
        v = zz[np.isfinite(zz)]
        rng = float(v.max() - v.min()) if v.size else 0.0
        print(f"  候选 ({lon0},{lat0}) 波束 {label} 走廊 {cg.length_m / 1000:.1f} km, "
              f"DEM 起伏 {rng:.0f} m (min {v.min():.0f} / max {v.max():.0f})")
        if chosen is None or rng > chosen[0]:
            chosen = (rng, p0, p1, label, dem, cg)
        if rng > 80:
            break
    if chosen is None:
        print("!! 没有可用的 DEM")
        return 1
    rng, p0, p1, label, dem, cg = chosen
    print(f"选中: 波束 {label}, 走廊 {cg.length_m / 1000:.2f} km, DEM 起伏 {rng:.0f} m")

    app.map.zoom_to_bounds(min(p0[0], p1[0]) - 0.02, max(p0[0], p1[0]) + 0.02,
                           min(p0[1], p1[1]) - 0.02, max(p0[1], p1[1]) + 0.02)
    pump(app, 1.0)
    app.conf_var.set("全部")
    app.channel_var.set("自动")
    app.elev_var.set("椭球高")
    app.point_size_var.set(2.0)
    app.reference_var.set("ATL03 高置信 (≥4)")
    app.ref_mode_var.set("高亮参考信号")
    app._on_pick(p0, p1)
    wait_for(app, lambda: app._busy_count == 0 and app.profile.selection is not None,
             300, "剖面")
    pump(app, 1.5)

    # 加载 DEM 并开启对比
    app.dem = dem
    app.dem_var.set(True)
    app.settings["dem_enabled"] = True
    app.settings["dem_datum"] = "正交高 (EGM96 等)"
    app._refresh_profile_display()
    pump(app, 2.0)
    cmp = app.last_dem_compare
    print("DEM:", dem.describe())
    print("对比:", cmp.summary() if cmp else "无", "|", (cmp.note if cmp else ""))
    print("剖面高程基准:", app.elev_var.get())

    app.profile.save_png(str(OUT / "dem_compare_profile.png"))
    try:
        from PIL import ImageGrab

        app.lift()
        app.attributes("-topmost", True)
        pump(app, 1.0)
        x, y = app.winfo_rootx(), app.winfo_rooty()
        img = ImageGrab.grab(bbox=(x, y, x + app.winfo_width(), y + app.winfo_height()))
        img.save(OUT / "window_dem.png")
        print("整窗截图:", OUT / "window_dem.png", img.size)
    except Exception as exc:
        print("整窗截图失败:", exc)

    app.after(200, app._on_close)
    pump(app, 1.0)
    print("DEM 对比演示图生成完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
