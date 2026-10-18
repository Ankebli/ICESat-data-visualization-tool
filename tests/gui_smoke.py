# -*- coding: utf-8 -*-
"""GUI 冒烟测试: 真实 ATL03 数据 -> 两点剖切 -> 截图.

以真实太湖 ATL03 数据驱动整个界面 (打开数据 / 地图叠加轨迹 / 拾取两点 /
计算剖面), 最后把地图、剖面图与整个窗口截图保存到 ``tests/_out``.

运行::

    C:\\Users\\ASUS\\miniconda3\\envs\\waterenv\\python.exe tests\\gui_smoke.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from main import enable_windows_dpi_awareness, setup_matplotlib  # noqa: E402

enable_windows_dpi_awareness()
setup_matplotlib()

from icesat_viewer import geo                                        # noqa: E402
from icesat_viewer.app import IcesatViewerApp                        # noqa: E402
from icesat_viewer.basemap import is_valid_token_format, mask_token  # noqa: E402
from icesat_viewer.corridor import CorridorGeometry                  # noqa: E402

OUT = Path(__file__).resolve().parent / "_out"
REAL_ATL03 = Path(r"D:\Paper\Data\satellite image\ICESAT\Taihu"
                  r"\ATL03_20250105065215_03082606_007_01.h5")
DEMO = ROOT / "demo_ATL03.h5"


def pump(app, seconds: float = 0.3):
    """跑事件循环若干时间 (替代 mainloop)."""
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.01)


def wait_for(app, condition, timeout: float = 180.0, label: str = ""):
    end = time.time() + timeout
    while time.time() < end:
        app.update()
        if condition():
            return True
        time.sleep(0.05)
    print(f"!! 等待超时: {label}")
    return False


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    use_demo = (len(sys.argv) > 1 and sys.argv[1] == "--demo") or os.environ.get("ICESAT_DEMO")
    data = None if use_demo else (REAL_ATL03 if REAL_ATL03.exists() else None)
    if data is None:
        from icesat_viewer.demo_data import generate_demo_atl03

        print("使用合成演示数据 (demo_ATL03.h5)")
        generate_demo_atl03(DEMO)
        data = DEMO

    app = IcesatViewerApp(provider="esri_img")
    app.geometry("1520x960+40+20")
    pump(app, 0.5)
    print(f"[1] 界面已创建, 底图 = {app.map.provider_key}")

    t0 = time.time()
    app.load_file(str(data))
    ok = wait_for(app, lambda: len(app.entries) > 0, 300, "加载数据")
    if not ok:
        return 1
    ds = app.entries[0].dataset
    print(f"[2] 数据已加载: {ds.kind} / {ds.point_count:,} 光子 / "
          f"耗时 {time.time() - t0:.1f}s")

    # 视野定位到太湖: 取最靠近太湖中心的轨迹, 沿轨迹方向取 ±6 km 作为剖切线
    best = None
    for label, lon, lat, _color in ds.track_full_arrays():
        d = np.hypot((lon - 120.20) * 95.0, (lat - 31.20) * 111.0)
        i = int(np.argmin(d))
        if best is None or d[i] < best[0]:
            best = (float(d[i]), label, i, lon, lat)
    _, label, i, lon_arr, lat_arr = best
    half = 320
    i0, i1 = max(i - half, 0), min(i + half, lon_arr.size - 1)
    p0 = (float(lon_arr[i0]), float(lat_arr[i0]))
    p1 = (float(lon_arr[i1]), float(lat_arr[i1]))
    cg = CorridorGeometry(p0, p1, float(app.half_width_var.get()))
    print(f"[3] 剖切线: {label}  Along {cg.length_m / 1000:.1f} km, 起点 {p0}, 终点 {p1}")

    app.map.zoom_to_bounds(min(p0[0], p1[0]) - 0.02, max(p0[0], p1[0]) + 0.02,
                           min(p0[1], p1[1]) - 0.02, max(p0[1], p1[1]) + 0.02)
    pump(app, 3.0)                      # 等瓦片下载完成

    app._on_pick(p0, p1)                # 等价于界面上点击两点
    ok = wait_for(app, lambda: app.profile.selection is not None
                  and app._busy_count == 0, 300, "剖面计算")
    sel = app.profile.selection
    if not ok or sel is None:
        return 1
    print(f"[4] 剖面完成: {len(sel):,} 光子, 波束 {sel.present_beams()}")
    print(sel.describe())

    app._set_info(sel.describe())
    pump(app, 2.0)

    app.map.save_png(str(OUT / "map.png"))
    app.profile.save_png(str(OUT / "profile.png"))

    # 切换到天地图 (无密钥时是占位块) 前先记录; 这里顺带验证密钥提示逻辑
    print(f"[5] 底图信息: {app.map.tile_info or '(尚未完成)'}")

    # 整窗截图
    try:
        from PIL import ImageGrab

        app.lift()
        app.attributes("-topmost", True)
        pump(app, 0.8)
        x, y = app.winfo_rootx(), app.winfo_rooty()
        w, h = app.winfo_width(), app.winfo_height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        img.save(OUT / "window_smoke.png")
        print(f"[6] 窗口截图: {OUT / 'window_smoke.png'} ({img.size[0]}x{img.size[1]})")
    except Exception as exc:
        print("[6] 窗口截图失败:", exc)

    # 验证导出
    csv_path = OUT / "profile.csv"
    app.profile.export_csv(str(csv_path))
    print(f"[7] CSV 导出: {csv_path} ({csv_path.stat().st_size / 1024:.1f} KB)")

    # 置信度筛选后重算
    app.conf_var.set("仅高置信")
    app.compute_profile("测试")
    wait_for(app, lambda: app._busy_count == 0, 300, "高置信剖面")
    print(f"[8] 仅高置信: {len(app.profile.selection):,} 光子")

    # ---------- 真实鼠标事件回归测试 (Tk 事件 -> matplotlib MouseEvent) ----------
    errors = []
    app.report_callback_exception = lambda exc, val, tb: (
        errors.append(f"{exc.__name__}: {val}"), traceback.print_exception(exc, val, tb))

    app.map.clear_pick()
    app.pick_var.set(True)
    app._toggle_pick_mode()
    widget = app.map.canvas.get_tk_widget()
    pump(app, 0.3)

    def to_widget_xy(lon, lat):
        """经纬度 -> Tk 控件像素 (左上角原点)."""
        x, y = geo.lonlat_to_mercator(lon, lat)
        dx, dy = app.map.ax.transData.transform((float(x), float(y)))
        return int(round(dx)), int(round(widget.winfo_height() - dy))

    # 取真实轨迹上的两点, 保证廊道里一定有光子
    _label, lon_arr, lat_arr, _color = ds.track_full_arrays()[0]
    mid = int(lon_arr.size * 0.5)
    i0, i1 = mid, min(mid + 300, lon_arr.size - 1)
    p0 = (float(lon_arr[i0]), float(lat_arr[i0]))
    p1 = (float(lon_arr[i1]), float(lat_arr[i1]))
    app.map.zoom_to_bounds(min(p0[0], p1[0]) - 0.01, max(p0[0], p1[0]) + 0.01,
                           min(p0[1], p1[1]) - 0.01, max(p0[1], p1[1]) + 0.01)
    pump(app, 1.5)

    xy0, xy1 = to_widget_xy(*p0), to_widget_xy(*p1)
    # 故意偏离轨道 ~3 像素, 检验"自动吸附"是否把点拉回轨道
    off = 3
    clicks = [(xy0[0] + off, xy0[1] + off), (xy1[0] - off, xy1[1] - off)]
    for xy in clicks:                           # 左键点击两点
        widget.event_generate("<Motion>", x=xy[0], y=xy[1])
        widget.event_generate("<ButtonPress-1>", x=xy[0], y=xy[1])
        widget.event_generate("<ButtonRelease-1>", x=xy[0], y=xy[1])
        pump(app, 0.6)
    ok = wait_for(app, lambda: app._busy_count == 0 and app.profile.selection is not None,
                  300, "鼠标点击剖面")
    sel_mouse = app.profile.selection
    print(f"[9] 鼠标点击两点 -> 拾取 {len(app.map.picks)} 个点, "
          f"剖面 {0 if sel_mouse is None else len(sel_mouse):,} 光子, "
          f"光标读数: {app.cursor_var.get()}")
    if not ok or sel_mouse is None or len(sel_mouse) == 0:
        print("!! 鼠标事件路径未能产生剖面")
        return 1

    # 吸附校验: 拾取点应落在真实轨道点附近 (远小于点击偏移), 且两点在同一波束
    labels = [l for l in app.map.pick_labels if l]
    track_pts = {lab: (lo, la) for lab, lo, la, _c in ds.track_full_arrays()}
    snap_err = []
    for (plon, plat), lab in zip(app.map.picks, app.map.pick_labels):
        if lab not in track_pts:
            snap_err.append(f"未吸附({lab})")
            continue
        lo, la = track_pts[lab]
        snap_err.append(round(float(np.min(np.hypot((lo - plon) * 95e3,
                                                    (la - plat) * 111e3))), 1))
    raw_err = [round(float(geo.GEOD.distance(*raw, *snap)), 1)
               for raw, snap in zip(app.map.pick_raw, app.map.picks) if raw]
    print(f"[9.1] 吸附校验: 拾取点到最近轨道点距离 = {snap_err} m, "
          f"吸附位移 = {raw_err} m, 波束 = {app.map.pick_labels}")
    if any(not isinstance(v, float) or v > 5.0 for v in snap_err):
        print("!! 选点未正确吸附到轨道")
        return 1
    if len(set(labels)) != 1:
        print("!! 两点未吸附到同一条轨道:", labels)
        return 1

    # 确定直线后应自动缩放到该直线范围 (而不是整个数据集)
    def view_bounds():
        v = app.map.view
        lo0, la0 = geo.mercator_to_lonlat(v[0], v[2])
        lo1, la1 = geo.mercator_to_lonlat(v[1], v[3])
        return float(lo0), float(lo1), float(la0), float(la1)

    lo0, lo1, la0, la1 = view_bounds()
    cg_now = CorridorGeometry(app.map.picks[0], app.map.picks[1], 25.0)
    ring = cg_now.polygon_lonlat()
    inside = (lo0 <= ring[:, 0].min() and ring[:, 0].max() <= lo1
              and la0 <= ring[:, 1].min() and ring[:, 1].max() <= la1)
    span_km = (app.map.view[1] - app.map.view[0]) / 1000.0
    data_bounds = ds.bounds
    data_km = geo.GEOD.distance(data_bounds[0], data_bounds[2],
                                data_bounds[1], data_bounds[3]) / 1000.0
    print(f"[9.2] 确定直线后自动缩放: 视野 {span_km:.1f} km 宽, "
          f"剖切线完全在视野内 = {inside}, 数据集跨度 ≈ {data_km:,.0f} km")
    if not inside or span_km > 100.0 or span_km >= data_km:
        print("!! 未按要求缩放到剖切线范围")
        return 1
    if "剖面线" not in str(app.zoom_button.cget("text")):
        print("!! 按钮文字未切换为“缩放到剖面线”")
        return 1

    # 缩到很大范围后再点"缩放到剖面线", 应重新框回该直线
    widget_mb = app.map.canvas.get_tk_widget()
    for _ in range(5):                       # 滚轮缩小, 视野拉大到上千公里
        widget_mb.event_generate("<MouseWheel>", delta=-120, x=400, y=200)
        pump(app, 0.2)
    pump(app, 0.8)
    away_span = (app.map.view[1] - app.map.view[0]) / 1000.0
    app.zoom_current_target()
    pump(app, 1.2)
    back_span = (app.map.view[1] - app.map.view[0]) / 1000.0
    lo0b, lo1b, la0b, la1b = view_bounds()
    back_inside = (lo0b <= ring[:, 0].min() and ring[:, 0].max() <= lo1b
                   and la0b <= ring[:, 1].min() and ring[:, 1].max() <= la1b)
    print(f"[9.3] 缩小到 {away_span:,.0f} km 后点击“缩放到剖面线”: "
          f"-> {back_span:.1f} km, 直线重新入框 = {back_inside}")
    if not back_inside or back_span > away_span * 0.5:
        print("!! 缩放到剖面线未生效")
        return 1

    # 左键拖动不应平移底图, 中键拖动才平移
    before = app.map.view
    widget.event_generate("<ButtonPress-1>", x=300, y=200)
    for x in (340, 400):
        widget.event_generate("<Motion>", x=x, y=200)
    widget.event_generate("<ButtonRelease-1>", x=400, y=200)
    pump(app, 0.5)
    left_drag_moved = abs(app.map.view[0] - before[0]) / 1000.0
    widget.event_generate("<ButtonPress-2>", x=300, y=200)
    for x in (340, 400):
        widget.event_generate("<Motion>", x=x, y=200)
    widget.event_generate("<ButtonRelease-2>", x=400, y=200)
    pump(app, 0.6)
    after = app.map.view
    moved = (after[0] - before[0]) / 1000.0
    print(f"[10] 左键拖动位移 {left_drag_moved:.3f} km (应为 0), "
          f"中键拖动平移 {moved:.2f} km (宽 {(after[1] - after[0]) / 1000:.1f} km), "
          f"拾取点保留 {len(app.map.picks)} 个")

    # 滚轮缩放
    w_before = after[1] - after[0]
    widget.event_generate("<MouseWheel>", delta=120, x=400, y=200)
    pump(app, 0.5)
    w_after = app.map.view[1] - app.map.view[0]
    print(f"[11] 滚轮缩放: {w_before / 1000:.1f} km -> {w_after / 1000:.1f} km "
          f"(z={app.map.zoom_level})")

    if errors:
        print("!! 交互过程中出现回调异常:", errors[:3])
        return 1
    if left_drag_moved > 0.001:
        print("!! 左键拖动竟然平移了底图")
        return 1
    if abs(moved) < 0.1 or not (w_after < w_before):
        print("!! 中键平移/滚轮缩放未生效")
        return 1
    print("[12] 鼠标交互 (左键吸附拾取/中键平移/滚轮缩放) 全部正常, 无回调异常")

    # ---------- 天地图底图 + 密钥 ----------
    from icesat_viewer import app as app_module

    app.map.zoom_to_bounds(p0[0] - 0.03, p1[0] + 0.03, p0[1] - 0.02, p1[1] + 0.02)
    if is_valid_token_format(app.settings.token):
        app.set_provider("tianditu_img")
        wait_for(app, lambda: bool(app.map.tile_info), 120, "天地图瓦片")
        pump(app, 6.0)
        app.map.save_png(str(OUT / "map_tianditu.png"))
        print(f"[13] 天地图底图: {app.map.tile_info}  (tk={mask_token(app.settings.token)})")
        if "错误" in app.map.tile_info or "不可用" in app.map.tile_info:
            print("!! 天地图底图未能加载")
            return 1
    else:
        print(f"[13] 跳过天地图: 当前密钥无效 ({mask_token(app.settings.token)})")

    # 无效密钥必须抛异常并在界面上报出来 (换 key 即刻校验, 不受瓦片缓存影响)
    ask_backup = app_module.messagebox.askyesno
    app_module.messagebox.askyesno = lambda *a, **k: False     # 自动化: 不弹交互框
    try:
        app._last_basemap_error = ""
        app._apply_token("0" * 32)                             # 格式合法但服务端拒绝
        wait_for(app, lambda: bool(app._last_basemap_error), 120, "无效密钥异常")
        pump(app, 1.0)
        msg = app._last_basemap_error
        print(f"[14] 无效密钥 -> 状态栏/弹窗异常: {msg[:96]}")
        if "TiandituTokenError" not in msg:
            print("!! 无效密钥未抛出 TiandituTokenError")
            return 1
    finally:
        app_module.messagebox.askyesno = ask_backup
        app._apply_token(app.settings.token)                   # 恢复
        app.map.refresh_tiles(force=True)
        pump(app, 3.0)
    print(f"[15] 恢复密钥后底图: {app.map.tile_info}")

    app.after(200, app._on_close)
    pump(app, 1.0)
    print("GUI smoke test PASSED [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
