# -*- coding: utf-8 -*-
"""生成 M0"参考模型"效果对比图 (原始 / 高亮 / 仅信号), 便于直观核对.

用法::

    C:\\Users\\ASUS\\miniconda3\\envs\\waterenv\\python.exe tests\\reference_compare.py
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

from icesat_viewer import geo                              # noqa: E402
from icesat_viewer.app import IcesatViewerApp              # noqa: E402
from icesat_viewer.corridor import CorridorGeometry        # noqa: E402

OUT = ROOT / "tests" / "_out"
REAL = Path(r"D:\Paper\Data\satellite image\ICESAT\Taihu"
            r"\ATL03_20250105065215_03082606_007_01.h5")


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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    app = IcesatViewerApp(provider="esri_img")
    app.geometry("1520x960+40+20")
    pump(app, 1.0)
    app.load_file(str(REAL))
    wait_for(app, lambda: len(app.entries) > 0, 300, "加载")
    ds = app.entries[0].dataset

    best = None
    for label, lon, lat, _c in ds.track_full_arrays():
        d = np.hypot((lon - 120.2) * 95.0, (lat - 31.2) * 111.0)
        i = int(np.argmin(d))
        if best is None or d[i] < best[0]:
            best = (float(d[i]), label, i, lon, lat)
    _, label, i, lon_arr, lat_arr = best
    p0 = (float(lon_arr[i - 320]), float(lat_arr[i - 320]))
    p1 = (float(lon_arr[i + 320]), float(lat_arr[i + 320]))
    app.map.zoom_to_bounds(min(p0[0], p1[0]) - 0.015, max(p0[0], p1[0]) + 0.015,
                           min(p0[1], p1[1]) - 0.015, max(p0[1], p1[1]) + 0.015)
    pump(app, 1.0)

    # 用"全部光子"进入, 便于看出淡化了什么
    app.conf_var.set("全部")
    app.channel_var.set("自动")
    app.elev_var.set("椭球高")
    app.point_size_var.set(2.0)
    app.surface_var.set(True)
    app._on_pick(p0, p1)
    wait_for(app, lambda: app._busy_count == 0 and app.profile.selection is not None,
             300, "剖面")
    pump(app, 1.5)
    print("剖面光子数:", len(app.profile.selection))

    for mode, fname in (("全部光子", "compare_plain.png"),
                        ("高亮参考信号", "compare_highlight.png"),
                        ("仅参考信号", "compare_only_signal.png")):
        app.ref_mode_var.set(mode)
        app.reference_var.set("ATL03 高置信 (≥4)")
        app._refresh_profile_display()
        pump(app, 1.2)
        app.profile.save_png(str(OUT / fname))
        ref = app.last_reference or {}
        print(f"  {mode:8s} -> {fname}  参考信号 {ref.get('n_signal', 0):,}/"
              f"{ref.get('n_total', 0):,} ({ref.get('ratio', 0):.1f}%)")

    # 整窗截图 (高亮模式)
    app.ref_mode_var.set("高亮参考信号")
    app._refresh_profile_display()
    pump(app, 1.5)
    try:
        from PIL import ImageGrab

        app.lift()
        app.attributes("-topmost", True)
        pump(app, 1.0)
        x, y = app.winfo_rootx(), app.winfo_rooty()
        img = ImageGrab.grab(bbox=(x, y, x + app.winfo_width(), y + app.winfo_height()))
        img.save(OUT / "window_reference.png")
        print("整窗截图:", OUT / "window_reference.png", img.size)
    except Exception as exc:
        print("整窗截图失败:", exc)

    app.after(200, app._on_close)
    pump(app, 1.0)
    print("对比图生成完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
