# -*- coding: utf-8 -*-
"""地图画布 (tkinter + matplotlib).

功能:
    * 以 Web Mercator 为内部坐标显示天地图/备用瓦片底图, 右侧/上侧显示经纬度刻度;
    * 滚轮缩放、左键拖动平移、右键取消拾取、双击缩放到数据;
    * 叠加 ICESat 轨迹 (按视野自适应抽稀) 与数据点;
    * "拾取剖面"模式下依次点击两点, 实时画出连线与廊道矩形, 完成后回调
      ``on_pick(p0, p1)`` 触发剖面计算.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

from . import geo
from .basemap import PROVIDERS, TileEngine

#: 打开程序时的初始视野 (中国区域)
INITIAL_CENTER = (105.0, 35.0)
INITIAL_SPAN_DEG = 60.0
MIN_VIEW_M = 200.0            # 最小视野 (约 200 m 宽)
WORLD_HALF = geo.ORIGIN_SHIFT
#: 地图坐标轴在画布中的位置 (留出上/右侧经纬度刻度与轴标题的空间)
AXES_RECT = (0.008, 0.008, 0.925, 0.885)


def _nice_step(span: float, target: int = 6) -> float:
    """选取 1/2/2.5/5×10^n 形式的"好看"刻度间隔."""
    import math

    if not np.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(int(target), 1)
    exp = math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 2.5, 5.0):
        if raw <= mult * (10.0 ** exp):
            return mult * (10.0 ** exp)
    return 10.0 ** (exp + 1)


def _fmt_deg(value: float, step: float) -> str:
    """按刻度间隔决定小数位数."""
    import math

    if step >= 1:
        return f"{value:.0f}°"
    decimals = int(min(6, max(1, -math.floor(math.log10(step)))))
    return f"{value:.{decimals}f}°"


class MapCanvas(ttk.Frame):
    """带底图的交互式地图画布."""

    def __init__(self, master, engine: TileEngine, *, provider: str = "tianditu_img",
                 dpi: float = 100.0, on_status: Optional[Callable[[str, str], None]] = None,
                 on_pick: Optional[Callable[[Tuple[float, float], Tuple[float, float]], None]] = None,
                 on_view_change: Optional[Callable[[Tuple[float, float, float, float]], None]] = None,
                 on_basemap_error: Optional[Callable[[Exception], None]] = None):
        super().__init__(master)
        self.engine = engine
        self.provider_key = provider if provider in PROVIDERS else "esri_img"
        self.on_status = on_status or (lambda text, where="left": None)
        self.on_pick = on_pick
        self.on_view_change = on_view_change
        self.on_basemap_error = on_basemap_error

        self.mode = "pan"                     # pan | pick
        self.picks: List[Tuple[float, float]] = []
        self.pick_labels: List[Optional[str]] = []
        self.pick_raw: List[Optional[Tuple[float, float]]] = []   # 吸附前的原始点击位置
        self.corridor = None                  # CorridorGeometry
        self.half_width_m = 25.0
        self.entries: List = []               # DatasetEntry 列表
        self.show_tracks = True
        #: 选点吸附: 由 App 注入回调 (lon, lat, tol_m, prefer_label) ->
        #: (lon, lat, 说明文字, 是否吸附成功, label)
        self.snap_function: Optional[Callable] = None
        self.snap_enabled = True
        self.snap_px = 25.0                   # 吸附容差 (像素, 随缩放自适应)
        self._track_cache: Dict[str, dict] = {}

        lon0, lat0 = INITIAL_CENTER
        x0, y0 = geo.lonlat_to_mercator(lon0, lat0)
        half = geo.ORIGIN_SHIFT * INITIAL_SPAN_DEG / 180.0
        self.view = (x0 - half, x0 + half, y0 - half, y0 + half)

        # ---------------- matplotlib ----------------
        self.figure = Figure(figsize=(7.2, 5.0), dpi=dpi, facecolor="#f4f5f7")
        self.ax = self.figure.add_axes(AXES_RECT)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_facecolor("#e9eaec")
        # 经纬度刻度: 直接以度值计算刻度的 Mercator 位置 (比 secondary_axis 更稳定)
        self.ax.xaxis.set_label_position("top")
        self.ax.yaxis.set_label_position("right")
        self.ax.set_xlabel("经度 (°E)", fontsize=8, color="#37474f")
        self.ax.set_ylabel("纬度 (°N)", fontsize=8, color="#37474f")
        self.ax.tick_params(axis="x", top=True, labeltop=True, bottom=False,
                            labelbottom=False, labelsize=8, colors="#455a64")
        self.ax.tick_params(axis="y", right=True, labelright=True, left=False,
                            labelleft=False, labelsize=8, colors="#455a64")
        self.ax.set_xlim(self.view[0], self.view[1])
        self.ax.set_ylim(self.view[2], self.view[3])
        self._update_geo_ticks()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().configure(bg="#e9eaec", highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # ---------------- 状态 ----------------
        self._bg_im = None
        self._overlays: List = []
        self._tile_queue: "queue.Queue[tuple]" = queue.Queue()
        self._generation = 0
        self._last_key = None
        self._pan = None
        self._press_xy = None
        self._drag_px = 0.0
        self._resize_job = None
        self._last_size = (0, 0)
        self._busy = False
        self._last_motion = 0.0
        self.zoom_level = 4
        self.tile_info = ""

        widget = self.canvas.get_tk_widget()
        # 交互事件统一走 matplotlib 事件系统: 只有 MouseEvent 才带
        # inaxes / xdata / ydata / button / dblclick, 裸 Tk Event 没有这些属性.
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        # 注意: 必须用 add="+" 追加绑定, 否则会覆盖 matplotlib 自身的 <Configure>
        # 处理函数 (它负责同步 backing store/PhotoImage 尺寸, 覆盖后画面只显示一角)
        widget.bind("<Configure>", self._on_configure, add="+")
        widget.configure(cursor="fleur")

        self._tile_job = self.after(90, self._poll_tiles)
        self.after(200, lambda: self.refresh_tiles(force=True))

    def stop_polling(self) -> None:
        """窗口关闭前取消待执行的定时回调, 避免 Tcl 报 invalid command name."""
        job = getattr(self, "_tile_job", None)
        if job is not None:
            try:
                self.after_cancel(job)
            except Exception:
                pass
            self._tile_job = None
    # ==================================================================
    # 视图控制
    # ==================================================================
    def _widget_size(self) -> Tuple[int, int]:
        widget = self.canvas.get_tk_widget()
        return max(int(widget.winfo_width()), 60), max(int(widget.winfo_height()), 60)

    def _axes_px(self) -> Tuple[int, int]:
        """地图坐标轴的实际像素尺寸 (与 matplotlib 内部一致, 避免纵横比留白)."""
        fig_w, fig_h = self.figure.get_size_inches()
        dpi = self.figure.dpi
        return (max(int(fig_w * dpi * AXES_RECT[2]), 40),
                max(int(fig_h * dpi * AXES_RECT[3]), 40))

    def _sync_figure_size(self) -> None:
        """把 figure 尺寸对齐到控件像素尺寸 (保证清晰度与纵横比一致)."""
        w, h = self._widget_size()
        dpi = self.figure.dpi
        self.figure.set_size_inches(w / dpi, h / dpi, forward=False)

    def _fit_aspect(self, view=None):
        """让视野长宽比与坐标轴一致, 避免 matplotlib 留白."""
        view = list(self.view if view is None else view)
        w_px, h_px = self._axes_px()
        target = h_px / max(w_px, 1)
        xmin, xmax, ymin, ymax = view
        cur = (ymax - ymin) / max(xmax - xmin, 1e-9)
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        width, height = xmax - xmin, ymax - ymin
        if cur < target:
            height = width * target
        else:
            width = height / target
        return (cx - width / 2.0, cx + width / 2.0, cy - height / 2.0, cy + height / 2.0)

    def _update_geo_ticks(self) -> None:
        """按当前视野生成上/右侧的经纬度刻度 (Mercator 位置 + 度数标签)."""
        try:
            lon0 = float(geo.mercator_to_lonlat(self.view[0], 0.0)[0])
            lon1 = float(geo.mercator_to_lonlat(self.view[1], 0.0)[0])
            lat0 = float(geo.mercator_to_lonlat(0.0, self.view[2])[1])
            lat1 = float(geo.mercator_to_lonlat(0.0, self.view[3])[1])

            step_lon = _nice_step(max(lon1 - lon0, 1e-6), 6)
            step_lat = _nice_step(max(lat1 - lat0, 1e-6), 5)
            lons = np.arange(np.ceil(lon0 / step_lon) * step_lon, lon1 + step_lon * 0.5,
                             step_lon)
            lats = np.arange(np.ceil(lat0 / step_lat) * step_lat, lat1 + step_lat * 0.5,
                             step_lat)
            if lons.size:
                xs, _ = geo.lonlat_to_mercator(lons, np.zeros_like(lons))
                self.ax.set_xticks(np.atleast_1d(xs))
                self.ax.set_xticklabels([_fmt_deg(v, step_lon) for v in lons])
            if lats.size:
                _, ys = geo.lonlat_to_mercator(np.zeros_like(lats), lats)
                self.ax.set_yticks(np.atleast_1d(ys))
                self.ax.set_yticklabels([_fmt_deg(v, step_lat) for v in lats])
        except Exception:
            pass

    def set_view(self, xmin, xmax, ymin, ymax, refresh: bool = True):
        width = max(float(xmax) - float(xmin), MIN_VIEW_M)
        height = max(float(ymax) - float(ymin), MIN_VIEW_M)
        cx = (float(xmin) + float(xmax)) / 2.0
        cy = (float(ymin) + float(ymax)) / 2.0
        width = min(width, 2 * WORLD_HALF)
        height = min(height, 2 * WORLD_HALF)
        cx = float(np.clip(cx, -WORLD_HALF + width / 2, WORLD_HALF - width / 2))
        cy = float(np.clip(cy, -WORLD_HALF + height / 2, WORLD_HALF - height / 2))
        self.view = self._fit_aspect((cx - width / 2, cx + width / 2,
                                      cy - height / 2, cy + height / 2))
        self.ax.set_xlim(self.view[0], self.view[1])
        self.ax.set_ylim(self.view[2], self.view[3])
        self._update_geo_ticks()
        self._draw_overlays()
        if refresh:
            self.refresh_tiles()
            self.canvas.draw_idle()
        if self.on_view_change:
            lon_min, lat_min = geo.mercator_to_lonlat(self.view[0], self.view[2])
            lon_max, lat_max = geo.mercator_to_lonlat(self.view[1], self.view[3])
            self.on_view_change((lon_min, lon_max, lat_min, lat_max))

    def zoom_to_bounds(self, lon_min, lon_max, lat_min, lat_max, margin: float = 0.08):
        """缩放到经纬度范围."""
        if not np.isfinite([lon_min, lon_max, lat_min, lat_max]).all():
            return
        if lon_max - lon_min < 1e-6:
            lon_min, lon_max = lon_min - 0.005, lon_max + 0.005
        if lat_max - lat_min < 1e-6:
            lat_min, lat_max = lat_min - 0.005, lat_max + 0.005
        pad_x = (lon_max - lon_min) * margin
        pad_y = (lat_max - lat_min) * margin
        x0, y0 = geo.lonlat_to_mercator(lon_min - pad_x, lat_min - pad_y)
        x1, y1 = geo.lonlat_to_mercator(lon_max + pad_x, lat_max + pad_y)
        self.set_view(x0, x1, y0, y1)

    def zoom_to_data(self, margin: float = 0.05):
        """缩放到全部可见数据范围.

        若已经确定了剖切线 (点了两点), 则**优先缩放到该直线 (含廊道宽度) 的范围**,
        便于立刻在视野里看到这条剖面线; 没有剖切线时才缩放到整个数据集范围。
        """
        if self.corridor is not None:
            self.zoom_to_corridor(self.corridor, margin=0.18)
            return
        boxes = [e.dataset.bounds for e in self.entries
                 if getattr(e, "visible", True) and e.dataset.bounds]
        if not boxes:
            self.on_status("没有可见数据可供缩放", "left")
            return
        lon_min = min(b[0] for b in boxes)
        lon_max = max(b[1] for b in boxes)
        lat_min = min(b[2] for b in boxes)
        lat_max = max(b[3] for b in boxes)
        self.zoom_to_bounds(lon_min, lon_max, lat_min, lat_max, margin)

    def zoom_to_corridor(self, corridor=None, margin: float = 0.18) -> bool:
        """缩放到剖切线 A→B 的范围 (含廊道半宽与端点标记的留白).

        返回是否成功 (没有剖切线时返回 False, 不改变视野).
        """
        corridor = corridor if corridor is not None else self.corridor
        if corridor is None:
            return False
        ring = corridor.polygon_lonlat(self.half_width_m)
        lon_min, lon_max = float(ring[:, 0].min()), float(ring[:, 0].max())
        lat_min, lat_max = float(ring[:, 1].min()), float(ring[:, 1].max())
        self.zoom_to_bounds(lon_min, lon_max, lat_min, lat_max, margin=margin)
        return True

    def zoom_to_point(self, lon: float, lat: float, span_m: float = 20000.0):
        x, y = geo.lonlat_to_mercator(lon, lat)
        self.set_view(x - span_m / 2, x + span_m / 2, y - span_m / 2, y + span_m / 2)

    def zoom(self, factor: float):
        """以视野中心缩放 (供菜单/快捷键调用)."""
        self._zoom(factor, None)

    def _zoom(self, factor: float, event=None):
        if event is not None and event.inaxes is not None:
            cx, cy = event.xdata, event.ydata
        else:
            cx = (self.view[0] + self.view[1]) / 2.0
            cy = (self.view[2] + self.view[3]) / 2.0
        xmin, xmax, ymin, ymax = self.view
        new_w = (xmax - xmin) / factor
        new_h = (ymax - ymin) / factor
        if new_w < MIN_VIEW_M or new_w > 4 * WORLD_HALF:
            return
        fx = (cx - xmin) / max(xmax - xmin, 1e-9)
        fy = (cy - ymin) / max(ymax - ymin, 1e-9)
        self.set_view(cx - fx * new_w, cx + (1 - fx) * new_w,
                      cy - fy * new_h, cy + (1 - fy) * new_h)

    def set_provider(self, key: str):
        if key not in PROVIDERS:
            return
        self.provider_key = key
        if self._bg_im is not None:      # 移除旧底图 artist, 避免叠加残留
            try:
                self._bg_im.remove()
            except Exception:
                pass
            self._bg_im = None
        self.refresh_tiles(force=True)
        self.canvas.draw_idle()

    def set_mode(self, mode: str):
        self.mode = "pick" if mode == "pick" else "pan"
        cursor = "crosshair" if self.mode == "pick" else "fleur"
        self.canvas.get_tk_widget().configure(cursor=cursor)
        if self.mode == "pan":
            self.clear_pick(redraw=True)

    # ==================================================================
    # 拾取
    # ==================================================================
    def clear_pick(self, redraw: bool = True):
        self.picks = []
        self.pick_labels = []
        self.pick_raw = []
        self.corridor = None
        if redraw:
            self._draw_overlays()
            self.canvas.draw_idle()

    def set_corridor(self, corridor):
        self.corridor = corridor
        if corridor is not None:
            self.picks = [corridor.p0, corridor.p1]
            if not self.pick_labels:
                self.pick_labels = [None, None]
            if not self.pick_raw:
                self.pick_raw = [None, None]
        self._draw_overlays()
        self.canvas.draw_idle()

    def set_snap(self, enabled: bool):
        self.snap_enabled = bool(enabled)

    def _snap_tolerance_m(self) -> float:
        """吸附容差: snap_px 像素对应的地面距离 (随缩放自适应, 限幅 15 m ~ 5 km)."""
        w_px, _ = self._axes_px()
        m_per_px = (self.view[1] - self.view[0]) / max(w_px, 1)
        return float(np.clip(self.snap_px * m_per_px, 15.0, 5000.0))

    def set_half_width(self, half_width_m: float):
        self.half_width_m = float(half_width_m)
        if self.corridor is not None:
            self._draw_overlays()
            self.canvas.draw_idle()

    def _pick_point(self, x_data: float, y_data: float):
        lon, lat = geo.mercator_to_lonlat(x_data, y_data)
        lon, lat = float(lon), float(lat)
        lat = float(np.clip(lat, -85.0, 85.0))

        # ---- 选点自动吸附到最近的 ICESat 轨道点 ----
        raw_point: Optional[Tuple[float, float]] = None
        label: Optional[str] = None
        if self.snap_enabled and self.snap_function is not None and len(self.picks) < 2:
            prefer = self.pick_labels[0] if self.picks else None
            tol_m = self._snap_tolerance_m()
            snap_lon, snap_lat, info, snapped, label = self.snap_function(
                lon, lat, tol_m, prefer)
            if snapped:
                raw_point = (lon, lat)          # 记录原始点击位置, 便于画提示线
                lon, lat = float(snap_lon), float(snap_lat)
            self.on_status(info, "left")
            if snapped and prefer and label and label == prefer:
                self.on_status(f"{info} · 与第一点同波束", "left")

        if len(self.picks) >= 2:
            self.picks = []
            self.pick_labels = []
            self.pick_raw = []
        self.picks.append((lon, lat))
        self.pick_labels.append(label)
        self.pick_raw.append(raw_point)
        self._draw_overlays()
        self.canvas.draw_idle()
        if len(self.picks) == 2:
            p0, p1 = self.picks
            if self.on_pick:
                self.on_pick(p0, p1)
        else:
            self.on_status("已选第一点 (已吸附到轨道), 请点击第二点 (右键取消)…", "left")

    # ==================================================================
    # 事件 (matplotlib MouseEvent: 带 inaxes / xdata / ydata / button / dblclick)
    #   左键   : 拾取剖面两点 (拾取模式) / 双击缩放到数据
    #   中键   : 拖动平移底图
    #   右键   : 取消拾取
    #   滚轮   : 缩放
    # ==================================================================
    def _on_press(self, event):
        if event.button == 3:                    # 右键: 取消拾取
            self.clear_pick()
            return
        if event.dblclick and event.button == 1:  # 左键双击: 缩放到数据
            self._pan = None
            self._press_xy = None
            self.zoom_to_data()
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if event.button == 2:                    # 中键: 拖动平移
            self._pan = (event.x, event.y, self.view)
            self._drag_px = 0.0
            self.canvas.get_tk_widget().configure(cursor="fleur")
            return
        if event.button == 1:                    # 左键: 仅用于拾取
            self._press_xy = (event.x, event.y)

    def _on_motion(self, event):
        # 中键拖动平移
        if self._pan is not None and event.x is not None and event.y is not None:
            x0, y0, view = self._pan
            w_px, h_px = self._axes_px()
            dx_px = event.x - x0
            dy_px = event.y - y0            # matplotlib 显示坐标 y 轴向上
            self._drag_px = abs(dx_px) + abs(dy_px)
            if self._drag_px >= 3.0:        # 阈值内视为抖动, 不移动视野
                dx = dx_px * (view[1] - view[0]) / max(w_px, 1)
                dy = dy_px * (view[3] - view[2]) / max(h_px, 1)
                self.set_view(view[0] - dx, view[1] - dx, view[2] - dy, view[3] - dy,
                              refresh=False)
                self.canvas.draw_idle()
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        # 拾取模式: 第一点之后跟随光标画预览线
        if self.mode == "pick" and len(self.picks) == 1:
            self._draw_overlays(cursor=(event.xdata, event.ydata))
            self.canvas.draw_idle()
        # 经纬度读数 (限流)
        now = time.time()
        if now - self._last_motion < 0.08:
            return
        self._last_motion = now
        lon, lat = geo.mercator_to_lonlat(event.xdata, event.ydata)
        self.on_status(f"{float(lon):.6f}°E, {float(lat):.6f}°N", "cursor")

    def _on_release(self, event):
        if self._pan is not None:
            dragged = self._drag_px >= 3.0
            self._pan = None
            self.canvas.get_tk_widget().configure(
                cursor="crosshair" if self.mode == "pick" else "fleur")
            if dragged:
                self.refresh_tiles()
                self.canvas.draw_idle()
            return
        press = self._press_xy
        self._press_xy = None
        if self.mode != "pick" or press is None:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        moved = abs(event.x - press[0]) + abs(event.y - press[1])
        if moved < 5:                            # 区分"点击"与"拖动"
            self._pick_point(event.xdata, event.ydata)

    def _on_scroll(self, event):
        step = getattr(event, "step", 0) or (1 if event.button == "up" else -1)
        self._zoom(1.30 ** float(step), event)

    def _on_configure(self, event):
        if (event.width, event.height) == self._last_size:
            return
        self._last_size = (event.width, event.height)
        if event.width > 20 and event.height > 20:
            self._sync_figure_size()
        if self._resize_job is not None:
            try:
                self.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.after(260, self._after_resize)

    def _after_resize(self):
        self._resize_job = None
        self._sync_figure_size()
        self.view = self._fit_aspect(self.view)
        self.ax.set_xlim(self.view[0], self.view[1])
        self.ax.set_ylim(self.view[2], self.view[3])
        self._update_geo_ticks()
        self._draw_overlays()
        self.refresh_tiles(force=True)
        self.canvas.draw_idle()

    # ==================================================================
    # 瓦片
    # ==================================================================
    def _choose_zoom(self) -> int:
        provider = PROVIDERS[self.provider_key]
        w_px, _ = self._axes_px()
        res = (self.view[1] - self.view[0]) / max(w_px, 1)
        return geo.zoom_for_resolution(res, 1, provider.max_zoom)

    def refresh_tiles(self, force: bool = False):
        """按当前视野异步抓取并拼接瓦片 (代次号避免过期结果覆盖)."""
        z = self._choose_zoom()
        tx0, tx1, ty0, ty1 = geo.tile_index_range(self.view[0], self.view[1],
                                                  self.view[2], self.view[3], z)
        key = (self.provider_key, z, tx0, tx1, ty0, ty1)
        if key == self._last_key and not force:
            return
        self._last_key = key
        self.zoom_level = z
        self._generation += 1
        generation = self._generation
        self._busy = True

        def work():
            try:
                arr, info = self.engine.mosaic(
                    self.provider_key, z, tx0, tx1, ty0, ty1,
                    should_cancel=lambda: generation != self._generation)
                if info.get("error"):
                    self.on_status(f"底图瓦片失败: {info['error']}", "left")
            except Exception as exc:      # pragma: no cover
                arr, info = None, {"error": str(exc)}
                self.on_status(f"底图瓦片失败: {exc}", "left")
            self._tile_queue.put((generation, arr, info))

        threading.Thread(target=work, daemon=True, name="mosaic").start()

    def _poll_tiles(self):
        try:
            if not self.winfo_exists():
                return
            while True:
                generation, arr, info = self._tile_queue.get_nowait()
                if generation == self._generation and arr is not None:
                    self._apply_mosaic(arr, info)
        except queue.Empty:
            pass
        except tk.TclError:
            return
        try:
            self._tile_job = self.after(90, self._poll_tiles)
        except tk.TclError:
            pass

    def _apply_mosaic(self, arr: np.ndarray, info: dict):
        z = int(info.get("z", self.zoom_level))
        span = geo.tile_span(z)
        x0 = info["tx0"] * span - geo.ORIGIN_SHIFT
        x1 = x0 + (info["tx1"] - info["tx0"] + 1) * span
        y1 = geo.ORIGIN_SHIFT - info["ty0"] * span
        y0 = y1 - (info["ty1"] - info["ty0"] + 1) * span
        if self._bg_im is None:
            self._bg_im = self.ax.imshow(arr, extent=(x0, x1, y0, y1), origin="upper",
                                         interpolation="bilinear", zorder=0)
        else:
            self._bg_im.set_data(arr)
            self._bg_im.set_extent((x0, x1, y0, y1))
        self._bg_im.set_zorder(0)
        self._busy = False
        fails = int(info.get("fail", 0))
        provider = PROVIDERS[self.provider_key]
        error = info.get("error_exc")
        if error is not None:                     # 密钥无效等致命错误 -> 交给界面处理
            self.tile_info = f"{provider.name} z={z} (底图错误: {type(error).__name__})"
            self.on_status(self.tile_info, "tiles")
            if self.on_basemap_error is not None:
                self.on_basemap_error(error)
        else:
            blocked = getattr(self.engine, "block_reason", lambda _k: "")(
                provider.base.key) if hasattr(self.engine, "block_reason") else ""
            if blocked:
                self.tile_info = f"{provider.name} z={z} (底图不可用: {blocked})"
            else:
                self.tile_info = f"{provider.name}  z={z}"
                if fails:
                    self.tile_info += f"  ({fails} 张瓦片缺失)"
            self.on_status(self.tile_info, "tiles")
        self.ax.set_xlim(self.view[0], self.view[1])
        self.ax.set_ylim(self.view[2], self.view[3])
        self._update_geo_ticks()
        self.canvas.draw_idle()

    # ==================================================================
    # 叠加图层
    # ==================================================================
    def set_datasets(self, entries: List):
        self.entries = entries
        self._track_cache.clear()
        self._draw_overlays()
        self.canvas.draw_idle()

    def _visible_entries(self) -> List:
        return [e for e in self.entries if getattr(e, "visible", True)]

    def _draw_overlays(self, cursor: Optional[Tuple[float, float]] = None):
        for artist in self._overlays:
            try:
                artist.remove()
            except Exception:
                pass
        self._overlays = []

        if self.show_tracks:
            segments, colors = [], []
            for entry in self._visible_entries():
                for label, lon, lat, color in self._track_runs(entry):
                    pts = np.column_stack(geo.lonlat_to_mercator(lon, lat))
                    if pts.shape[0] >= 2:
                        segments.append(pts)
                        colors.append(color)
            if segments:
                lc = LineCollection(segments, colors=colors, linewidths=0.9, alpha=0.9,
                                    zorder=2)
                self.ax.add_collection(lc)
                self._overlays.append(lc)

        # 小规模数据直接画点
        for entry in self._visible_entries():
            ds = entry.dataset
            if ds.point_count <= 150000:
                try:
                    for _label, lon, lat, _color in ds.track_full_arrays():
                        x, y = geo.lonlat_to_mercator(lon, lat)
                        sc = self.ax.scatter(x, y, s=3, c=entry.color, alpha=0.75,
                                             linewidths=0, zorder=3)
                        self._overlays.append(sc)
                except Exception:
                    pass

        # 廊道
        if self.corridor is not None:
            ring = self.corridor.polygon_mercator(self.half_width_m)
            patch = MplPolygon(ring, closed=True, facecolor="#ff980033",
                               edgecolor="#ff6d00", linewidth=1.6, linestyle="--", zorder=4)
            self.ax.add_patch(patch)
            self._overlays.append(patch)
            line = self.corridor.centerline_mercator(64)
            ln, = self.ax.plot(line[:, 0], line[:, 1], color="#ff6d00", linewidth=1.8,
                               zorder=5)
            self._overlays.append(ln)

        # 拾取点 / 橡皮筋
        picks = list(self.picks)
        for i, (lon, lat) in enumerate(picks[:2]):
            x, y = geo.lonlat_to_mercator(lon, lat)
            mk, = self.ax.plot([x], [y], marker="o", markersize=9, markerfacecolor="none",
                               markeredgecolor="#d50000", markeredgewidth=2.0, zorder=6)
            label = self.pick_labels[i] if i < len(self.pick_labels) else None
            text = "AB"[i] + (f"\n{label}" if label else "")
            tx = self.ax.annotate(text, (x, y), textcoords="offset points", xytext=(9, 6),
                                  color="#d50000", fontsize=10, fontweight="bold", zorder=6)
            self._overlays.extend([mk, tx])
            # 吸附提示: 原始点击位置 + 虚线连线
            raw = self.pick_raw[i] if i < len(self.pick_raw) else None
            if raw is not None and raw != (lon, lat):
                rx, ry = geo.lonlat_to_mercator(raw[0], raw[1])
                cross, = self.ax.plot([rx], [ry], marker="x", markersize=6,
                                      color="#616161", zorder=6)
                link, = self.ax.plot([rx, x], [ry, y], color="#9e9e9e", linewidth=1.0,
                                     linestyle=(0, (2, 2)), zorder=5)
                self._overlays.extend([cross, link])
        if cursor is not None and len(picks) == 1:
            x0, y0 = geo.lonlat_to_mercator(*picks[0])
            ln, = self.ax.plot([x0, cursor[0]], [y0, cursor[1]], color="#d50000",
                               linewidth=1.2, linestyle=":", zorder=5)
            self._overlays.append(ln)

    def _track_runs(self, entry) -> List[Tuple[str, np.ndarray, np.ndarray, str]]:
        """按当前视野取轨迹点并切成连续小段 (避免跨段连线)."""
        try:
            arrays = entry.dataset.track_full_arrays()
        except Exception:
            return []
        lon_min, lat_min = geo.mercator_to_lonlat(self.view[0], self.view[2])
        lon_max, lat_max = geo.mercator_to_lonlat(self.view[1], self.view[3])
        out = []
        span_deg = max(lon_max - lon_min, 1e-4)
        max_points = int(np.clip(40000 * (10.0 / span_deg), 400, 60000))
        for label, lon, lat, color in arrays:
            mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
            idx = np.nonzero(mask)[0]
            if idx.size < 2:
                continue
            stride = max(1, int(idx.size // max_points))
            idx = idx[::stride]
            breaks = np.nonzero(np.diff(idx) > stride * 2 + 1)[0]
            for chunk in np.split(idx, breaks + 1):
                if chunk.size >= 2:
                    out.append((label, lon[chunk], lat[chunk], color))
        return out

    # ==================================================================
    def save_png(self, path: str):
        self.figure.savefig(path, dpi=170)

    def screen_center_lonlat(self):
        return geo.mercator_to_lonlat((self.view[0] + self.view[1]) / 2.0,
                                      (self.view[2] + self.view[3]) / 2.0)
