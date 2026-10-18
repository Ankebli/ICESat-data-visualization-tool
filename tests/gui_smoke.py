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
    # 用"全部 + 自动通道"保证走廊全长都有光子 (否则可能被高置信/特定通道筛掉)
    app.conf_var.set("全部")
    app.channel_var.set("自动")
    app.afterpulse_var.set(False)
    widget = app.map.canvas.get_tk_widget()
    pump(app, 0.3)

    def to_widget_xy(lon, lat):
        """经纬度 -> Tk 控件像素 (左上角原点)."""
        x, y = geo.lonlat_to_mercator(lon, lat)
        dx, dy = app.map.ax.transData.transform((float(x), float(y)))
        return int(round(dx)), int(round(app.map.figure.bbox.height - dy))

    def send(widget_, ev, x, y, **kw):
        """合成鼠标事件。

        必须带正确的 root 坐标: matplotlib 的 Tk 后端会用
        ``winfo_containing(x_root, y_root)`` 过滤事件, 缺了会被丢弃。
        """
        widget_.event_generate(ev, x=int(x), y=int(y),
                               rootx=int(widget_.winfo_rootx() + x),
                               rooty=int(widget_.winfo_rooty() + y), **kw)

    # 取真实轨迹上的两点, 保证廊道里一定有光子
    _label, lon_arr, lat_arr, _color = ds.track_full_arrays()[0]
    mid = int(lon_arr.size * 0.5)
    i0, i1 = mid, min(mid + 300, lon_arr.size - 1)
    p0 = (float(lon_arr[i0]), float(lat_arr[i0]))
    p1 = (float(lon_arr[i1]), float(lat_arr[i1]))
    app.map.zoom_to_bounds(min(p0[0], p1[0]) - 0.01, max(p0[0], p1[0]) + 0.01,
                           min(p0[1], p1[1]) - 0.01, max(p0[1], p1[1]) + 0.01)
    pump(app, 1.5)

    # 故意偏离轨道 ~3 像素, 检验"自动吸附"是否把点拉回轨道。
    # 每次点击前都重新按"当前视野"换算像素 —— 布局回调可能已改变视野。
    clicks = []
    for i, pt in enumerate((p0, p1)):
        pump(app, 0.8)
        xy = to_widget_xy(*pt)
        off = 3 if i == 0 else -3
        cx, cy = xy[0] + off, xy[1] + off
        clicks.append((cx, cy))
        send(widget, "<Motion>", cx, cy)
        send(widget, "<ButtonPress-1>", cx, cy)
        send(widget, "<ButtonRelease-1>", cx, cy)
        pump(app, 0.8)
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
    if any(v > 500.0 for v in raw_err):
        print("!! 点击位置与轨道偏差过大 (像素换算/视野不稳定?):", raw_err)
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
    for _ in range(5):                       # 用 API 缩小 (滚轮路径另在 [11] 验证)
        app.map.zoom(1 / 1.3)
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
    send(widget_mb, "<ButtonPress-1>", 300, 200)
    for x in (340, 400):
        send(widget_mb, "<Motion>", x, 200)
    send(widget_mb, "<ButtonRelease-1>", 400, 200)
    pump(app, 0.5)
    left_drag_moved = abs(app.map.view[0] - before[0]) / 1000.0
    send(widget_mb, "<ButtonPress-2>", 300, 200)
    for x in (340, 400):
        send(widget_mb, "<Motion>", x, 200)
    send(widget_mb, "<ButtonRelease-2>", 400, 200)
    pump(app, 0.6)
    after = app.map.view
    moved = (after[0] - before[0]) / 1000.0
    print(f"[10] 左键拖动位移 {left_drag_moved:.3f} km (应为 0), "
          f"中键拖动平移 {moved:.2f} km (宽 {(after[1] - after[0]) / 1000:.1f} km), "
          f"拾取点保留 {len(app.map.picks)} 个")

    # 滚轮缩放
    w_before = after[1] - after[0]
    send(widget_mb, "<MouseWheel>", 400, 200, delta=120)
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

    # ---------- 地图 <-> 剖面 联动 ----------
    app.pick_var.set(False)                 # 浏览模式下左键 = 联动定位
    app._toggle_pick_mode()
    pump(app, 0.5)
    sel_now = app.profile.selection
    corr_now = app.map.corridor
    if sel_now is None or corr_now is None:
        print("!! 切换模式后剖面丢失 (联动无法验证)")
        return 1

    # (a) 地图轨迹上点一下 -> 剖面高亮对应位置
    target_along = corr_now.length_m * 0.42
    tlon, tlat = geo.GEOD.forward(corr_now.p0[0], corr_now.p0[1], corr_now.azimuth,
                                  target_along)
    pump(app, 0.5)
    mx, my = to_widget_xy(tlon, tlat)
    send(widget_mb, "<ButtonPress-1>", mx, my)
    send(widget_mb, "<ButtonRelease-1>", mx, my)
    pump(app, 0.9)
    m_along = app.profile.marker_along
    a_chk, c_chk = corr_now.project(tlon, tlat)
    print(f"[16] 地图点击 (目标沿迹 {target_along:,.0f} m) -> 剖面高亮 "
          f"{'None' if m_along is None else format(m_along, ',.0f')} m, "
          f"地图标记={'有' if app.map.link_marker else '无'}")
    print(f"     点击点回投: 沿迹 {float(a_chk):,.1f} m, 横向 {float(c_chk):+.1f} m")
    print(f"     状态栏: {app.status_var.get()}")
    loc = getattr(app, "last_locate", {}) or {}
    # (1) 应用自洽: 高亮的光子 / 状态栏 / last_locate 必须是同一位置
    if loc.get("source") != "map" or abs(float(loc.get("along", -1)) - m_along) > 1.0:
        print(f"!! 联动定位不自洽: marker={m_along}, last_locate={loc}")
        return 1
    # (2) 与目标位置接近 (合成点击有若干像素误差, 这里给宽松上限)
    loose = max(300.0, corr_now.length_m * 0.05)
    if abs(m_along - target_along) > loose or app.map.link_marker is None:
        print(f"!! 地图 -> 剖面 联动偏差过大 (容差 {loose:.0f} m)")
        return 1

    # (b) 剖面上点一下 -> 地图 zoom 到对应位置
    pick_along = corr_now.length_m * 0.75
    med_elev = float(np.median(app.profile._plot_elev))
    ax_px = app.profile.ax.transData.transform((pick_along, med_elev))
    prof_widget = app.profile.canvas.get_tk_widget()
    px = int(round(ax_px[0]))
    py = int(round(app.profile.figure.bbox.height - ax_px[1]))
    probe = []
    app.profile.canvas.mpl_connect(
        "button_press_event",
        lambda e: probe.append((round(e.x), round(e.y),
                                None if e.xdata is None else round(float(e.xdata)),
                                e.inaxes is not None)))
    send(prof_widget, "<ButtonPress-1>", px, py)
    send(prof_widget, "<ButtonRelease-1>", px, py)
    pump(app, 1.4)
    print(f"     剖面收到事件: {probe} (期望 x={px}, xdata≈{pick_along:.0f})")
    pa = app.profile._plot_along
    print(f"     剖面数据: N={0 if pa is None else pa.size}, "
          f"along {0 if pa is None else pa.min():.0f}~{0 if pa is None else pa.max():.0f} m "
          f"(走廊 {corr_now.length_m:,.0f} m)")
    exp_lon, exp_lat = geo.GEOD.forward(corr_now.p0[0], corr_now.p0[1],
                                        corr_now.azimuth, pick_along)
    clon, clat = app.map.screen_center_lonlat()
    err_m = geo.GEOD.distance(float(clon), float(clat), float(exp_lon), float(exp_lat))
    print(f"[17] 剖面点击 (沿迹 {pick_along:,.0f} m) -> 地图中心 "
          f"({float(clon):.5f}, {float(clat):.5f}), 距期望位置 {err_m:.0f} m, "
          f"视野 {(app.map.view[1] - app.map.view[0]) / 1000:.1f} km, "
          f"剖面高亮 {'None' if app.profile.marker_along is None else round(app.profile.marker_along)} m")
    print(f"     点击像素=({px},{py}) 画布={prof_widget.winfo_width()}x"
          f"{prof_widget.winfo_height()} axes={[round(v) for v in app.profile.ax.get_window_extent().bounds]} "
          f"last_locate={app.last_locate.get('source')}")
    if err_m > 900.0 or app.map.link_marker is None:
        print("!! 剖面 -> 地图 联动失败")
        return 1

    # ---------- 天地图底图 + 密钥 ----------
    from icesat_viewer import app as app_module

    app.map.zoom_to_bounds(p0[0] - 0.03, p1[0] + 0.03, p0[1] - 0.02, p1[1] + 0.02)
    if is_valid_token_format(app.settings.token):
        app.set_provider("tianditu_img")
        wait_for(app, lambda: bool(app.map.tile_info), 150, "天地图瓦片")
        pump(app, 6.0)
        app.map.save_png(str(OUT / "map_tianditu.png"))
        print(f"[13] 天地图底图: {app.map.tile_info}  (tk={mask_token(app.settings.token)})")
        if app.map.provider_key.startswith("tianditu"):
            if "错误" in app.map.tile_info or "不可用" in app.map.tile_info:
                print("!! 天地图底图未能加载")
                return 1
        else:
            # 密钥被限流 (HTTP 418) 时程序会临时降级到 Esri —— 这是预期行为
            print(f"     天地图暂不可用 (可能限流), 已自动降级: {app.status_var.get()[:60]}")
            if "418" not in (app._last_basemap_error or "") \
                    and "限流" not in (app._last_basemap_error or ""):
                print("!! 底图既没加载也没有给出限流原因")
                return 1
    else:
        print(f"[13] 跳过天地图: 当前密钥无效 ({mask_token(app.settings.token)})")

    # 无效密钥必须抛异常上报, 但**绝不能弹模态对话框**卡住用户
    try:
        app._last_basemap_error = ""
        app._apply_token("0" * 32)                             # 格式合法但服务端拒绝
        wait_for(app, lambda: bool(app._last_basemap_error), 120, "无效密钥异常")
        pump(app, 1.0)
        msg = app._last_basemap_error
        modal = app.grab_current()
        print(f"[14] 无效密钥 -> 状态栏异常: {msg[:88]}")
        print(f"     模态对话框: {'有(不允许!)' if modal is not None else '无 (正确)'}")
        if not ("TiandituTokenError" in msg or "TiandituQuotaError" in msg):
            print("!! 无效密钥未抛出天地图异常")
            return 1
        if modal is not None:
            print("!! 底图错误弹出了模态对话框, 会打断用户操作")
            return 1
    finally:
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
