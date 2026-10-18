# -*- coding: utf-8 -*-
"""最终验收截图: 默认配置 (天地图底图) + 真实太湖 ATL03 + 两点剖切."""

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

from icesat_viewer import geo                             # noqa: E402
from icesat_viewer.app import IcesatViewerApp             # noqa: E402
from icesat_viewer.corridor import CorridorGeometry       # noqa: E402

OUT = ROOT / "tests" / "_out"
REAL = Path(r"D:\Paper\Data\satellite image\ICESAT\Taihu"
            r"\ATL03_20250105065215_03082606_007_01.h5")


def pump(app, seconds=0.4):
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.01)


def wait_for(app, cond, timeout=180, label=""):
    end = time.time() + timeout
    while time.time() < end:
        app.update()
        if cond():
            return True
        time.sleep(0.05)
    print("!! 超时:", label)
    return False


def main() -> int:
    app = IcesatViewerApp()                     # 使用 config.json 默认配置 (天地图)
    app.geometry("1520x960+40+20")
    pump(app, 1.0)
    print("底图:", app.map.provider_key, "| 密钥:", app.settings.token)
    wait_for(app, lambda: bool(app.map.tile_info), 60, "天地图底图")

    app.load_file(str(REAL))
    wait_for(app, lambda: len(app.entries) > 0, 300, "加载数据")
    ds = app.entries[0].dataset

    best = None
    for label, lon, lat, _c in ds.track_full_arrays():
        d = np.hypot((lon - 120.20) * 95.0, (lat - 31.20) * 111.0)
        i = int(np.argmin(d))
        if best is None or d[i] < best[0]:
            best = (float(d[i]), label, i, lon, lat)
    _, label, i, lon_arr, lat_arr = best
    p0 = (float(lon_arr[i - 320]), float(lat_arr[i - 320]))
    p1 = (float(lon_arr[i + 320]), float(lat_arr[i + 320]))

    app.map.zoom_to_bounds(min(p0[0], p1[0]) - 0.015, max(p0[0], p1[0]) + 0.015,
                           min(p0[1], p1[1]) - 0.015, max(p0[1], p1[1]) + 0.015)
    pump(app, 1.0)
    wait_for(app, lambda: "张瓦片缺失" not in app.map.tile_info, 60, "瓦片补齐")
    app.half_width_var.set(25.0)

    # 模拟真实鼠标点击 (故意偏开轨道若干像素, 以展示"自动吸附"效果)
    app.pick_var.set(True)
    app._toggle_pick_mode()
    pump(app, 0.3)
    widget = app.map.canvas.get_tk_widget()

    def to_widget_xy(lon, lat):
        x, y = geo.lonlat_to_mercator(lon, lat)
        dx, dy = app.map.ax.transData.transform((float(x), float(y)))
        return int(round(dx)), int(round(widget.winfo_height() - dy))

    for i, (plon, plat) in enumerate((p0, p1)):
        px, py = to_widget_xy(plon, plat)
        off = 9 if i == 0 else -9
        widget.event_generate("<Motion>", x=px + off, y=py + off)
        widget.event_generate("<ButtonPress-1>", x=px + off, y=py + off)
        widget.event_generate("<ButtonRelease-1>", x=px + off, y=py + off)
        pump(app, 0.8)
    print("吸附结果:", app.map.pick_labels,
          "| 原始点击偏移(m):",
          [round(float(geo.GEOD.distance(*r, *s)), 1)
           for r, s in zip(app.map.pick_raw, app.map.picks) if r])
    wait_for(app, lambda: app._busy_count == 0 and app.profile.selection is not None,
             300, "剖面")
    app.conf_var.set("仅高置信")
    app.compute_profile("最终截图")
    wait_for(app, lambda: app._busy_count == 0, 300, "高置信剖面")
    pump(app, 4.0)

    print("底图信息:", app.map.tile_info)
    print(app.profile.selection.describe())
    app.map.save_png(str(OUT / "map.png"))
    app.profile.save_png(str(OUT / "profile.png"))

    try:
        from PIL import ImageGrab

        app.lift()
        app.attributes("-topmost", True)
        pump(app, 1.0)
        x, y = app.winfo_rootx(), app.winfo_rooty()
        img = ImageGrab.grab(bbox=(x, y, x + app.winfo_width(), y + app.winfo_height()))
        img.save(OUT / "window.png")
        print("整窗截图:", OUT / "window.png", img.size)
    except Exception as exc:
        print("截图失败:", exc)

    app.after(200, app._on_close)
    pump(app, 1.0)
    print("最终验收截图完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
