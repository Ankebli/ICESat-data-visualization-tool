# -*- coding: utf-8 -*-
"""剖面视图 (tkinter + matplotlib).

左侧: 沿剖切线距离 - 高程 的光子散点 (按波束着色), 可叠加面元中值线与
10~90% 分位带; 右侧: 同一高程区间内的光子数直方图 —— 即"该直线下的
ICESat 光子剖切面".
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .corridor import binned_statistic, elevation_statistics
from .icesat_io import BEAM_COLORS, BEAM_NAMES, ELEV_KINDS, PhotonSelection

FALLBACK_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


class ProfileCanvas(ttk.Frame):
    """剖切面绘图控件."""

    def __init__(self, master, *, dpi: float = 100.0,
                 on_status: Optional[Callable[[str, str], None]] = None,
                 on_locate: Optional[Callable[[float, float], None]] = None,
                 max_plot_points: int = 600000):
        super().__init__(master)
        self.on_status = on_status or (lambda text, where="left": None)
        self.on_locate = on_locate            # 点击剖面 -> 回调 (沿迹距离, 高程)
        self.max_plot_points = int(max_plot_points)
        self.selection: Optional[PhotonSelection] = None
        self.elev_kind = ELEV_KINDS[0]
        #: 联动高亮 (地图 <-> 剖面)
        self.marker_along: Optional[float] = None
        self.marker_elev: Optional[float] = None
        self._marker_artists: list = []
        self._plot_along: Optional[np.ndarray] = None
        self._plot_elev: Optional[np.ndarray] = None

        self.figure = Figure(figsize=(7.2, 2.9), dpi=dpi, facecolor="#fbfbfc")
        self.ax = self.figure.add_axes([0.075, 0.165, 0.695, 0.755])
        self.axh = self.figure.add_axes([0.815, 0.165, 0.165, 0.755], sharey=self.ax)
        self.ax.grid(True, color="#e2e4e8", linewidth=0.7)
        self.ax.set_axisbelow(True)
        self.axh.grid(True, axis="x", color="#e8eaee", linewidth=0.6)
        self.axh.set_axisbelow(True)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().configure(highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        # matplotlib 自带 <Configure> 处理函数, 会同步 figure 尺寸与 backing store,
        # 此处无需再绑定 (追加绑定 "add=+" 会保留其功能).
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.ax.format_coord = self._format_coord
        self._empty("请在地图上点击两点以生成 ICESat 光子剖切面")

    # ==================================================================
    # 联动: 点击剖面 -> 地图定位 / 高亮
    # ==================================================================
    def _on_click(self, event) -> None:
        """点击剖面图中的一点 -> 回调 (沿迹距离, 高程), 由主窗体把地图定位过去."""
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self.on_locate is None or self.selection is None or len(self.selection) == 0:
            return
        along = float(np.clip(event.xdata, 0.0, self.selection.corridor.length_m))
        elev = float(event.ydata) if event.ydata is not None else float("nan")
        self.on_locate(along, elev)

    def set_marker(self, along: Optional[float], elev: Optional[float] = None,
                   label: str = "") -> None:
        """在剖面上高亮一个位置 (垂直虚线 + 亮点 + 文字标注)."""
        self._clear_marker_artists()
        self.marker_along, self.marker_elev = along, elev
        if along is None or self.selection is None:
            self.canvas.draw_idle()
            return
        line = self.ax.axvline(float(along), color="#d81b60", linewidth=1.2,
                               linestyle=(0, (4, 3)), zorder=7)
        self._marker_artists.append(line)
        if elev is not None and np.isfinite(elev):
            dot, = self.ax.plot([float(along)], [float(elev)], marker="o", markersize=8,
                                markerfacecolor="none", markeredgecolor="#d81b60",
                                markeredgewidth=2.0, zorder=8)
            self._marker_artists.append(dot)
            text = label or f"{along:,.0f} m / {elev:,.2f} m"
            ann = self.ax.annotate(text, (float(along), float(elev)),
                                   textcoords="offset points", xytext=(10, 10),
                                   fontsize=8.5, color="#880e4f", zorder=8,
                                   bbox=dict(boxstyle="round,pad=0.25", fc="#fff3f8",
                                             ec="#f8bbd0", lw=0.8, alpha=0.95))
            self._marker_artists.append(ann)
        self.canvas.draw_idle()

    def clear_marker(self) -> None:
        self.marker_along = self.marker_elev = None
        self._clear_marker_artists()
        self.canvas.draw_idle()

    def _clear_marker_artists(self) -> None:
        for artist in self._marker_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._marker_artists = []

    def nearest_photon(self, along: float, elev: Optional[float] = None):
        """在剖面上找点击位置对应的光子, 返回 (along, elev) 或 None.

        策略: 先按沿迹距离圈一个窄窗 (1% 线长, 至少 20 m), 再在窗内挑高程最接近
        点击处的光子 —— 这样"沿迹位置"不会被高程差异带偏。
        """
        if self._plot_along is None or self._plot_along.size == 0:
            return None
        along_arr = self._plot_along
        elev_arr = self._plot_elev
        span = max(float(np.ptp(along_arr)), 1.0)
        window = max(span * 0.01, 20.0)
        while True:
            mask = np.abs(along_arr - float(along)) <= window
            if mask.any() or window >= span:
                break
            window *= 2.0
        if not mask.any():
            idx = int(np.argmin(np.abs(along_arr - float(along))))
        else:
            cand = np.nonzero(mask)[0]
            if elev is None or not np.isfinite(elev):
                idx = int(cand[np.argmin(np.abs(along_arr[cand] - float(along)))])
            else:
                idx = int(cand[np.argmin(np.abs(elev_arr[cand] - float(elev)))])
        return float(along_arr[idx]), float(elev_arr[idx])

    # ==================================================================
    def _empty(self, message: str):
        self.ax.clear()
        self.axh.clear()
        self._marker_artists = []
        self._plot_along = self._plot_elev = None
        self.ax.grid(True, color="#e2e4e8", linewidth=0.7)
        self.ax.set_axisbelow(True)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(0.5, 0.5, message, ha="center", va="center", color="#8a8f98",
                     fontsize=11, transform=self.ax.transAxes)
        self.axh.set_xticks([])
        self.axh.set_yticks([])
        self.canvas.draw_idle()

    def clear(self, message: str = "请在地图上点击两点以生成 ICESat 光子剖切面"):
        self.selection = None
        self.marker_along = self.marker_elev = None
        self._marker_artists = []
        self._plot_along = self._plot_elev = None
        self._empty(message)

    # ==================================================================
    def set_selection(self, selection: Optional[PhotonSelection], *,
                      elev_kind: str = ELEV_KINDS[0], point_size: float = 2.0,
                      show_surface: bool = True, bin_size_m: float = 20.0,
                      hist_bins: int = 200, show_histogram: bool = True):
        """绘制剖面."""
        self.selection = selection
        self.elev_kind = elev_kind
        if selection is None or len(selection) == 0:
            self._empty("该直线范围内未找到 ICESat 光子\n"
                        "请确认两点落在 ICESat 覆盖区内, 或加大廊道宽度")
            return

        elev_all = selection.display_elev(elev_kind)
        along_all = np.asarray(selection.along, "f8")
        labels = np.asarray(selection.beam, dtype=str)
        finite = np.isfinite(elev_all) & np.isfinite(along_all)
        elev_all, along_all, labels = elev_all[finite], along_all[finite], labels[finite]
        if elev_all.size == 0:
            self._empty("剖面内无有效高程值")
            return

        # 抽稀绘制 (统计仍用全量)
        stride_note = ""
        if elev_all.size > self.max_plot_points:
            keep = np.random.default_rng(0).choice(elev_all.size,
                                                   size=self.max_plot_points, replace=False)
            stride_note = f" (显示抽稀至 {self.max_plot_points:,})"
        else:
            keep = slice(None)

        self.ax.clear()
        self.axh.clear()
        self._marker_artists = []          # ax.clear() 已移除旧的标记 artist
        self.marker_along = self.marker_elev = None
        self._plot_along, self._plot_elev = along_all, elev_all
        self.ax.grid(True, color="#e2e4e8", linewidth=0.7)
        self.ax.set_axisbelow(True)

        # ---- 光子散点 (按波束着色) ----
        handles = []
        unique = list(dict.fromkeys(labels.tolist()))
        for i, label in enumerate(unique):
            mask = labels == label
            color = BEAM_COLORS.get(label, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
            if isinstance(keep, np.ndarray):
                sub = mask[keep]
                xs, ys = along_all[keep][sub], elev_all[keep][sub]
            else:
                xs, ys = along_all[mask], elev_all[mask]
            if xs.size == 0:
                continue
            self.ax.scatter(xs, ys, s=float(point_size), c=color, alpha=0.6,
                            linewidths=0, rasterized=True,
                            label=BEAM_NAMES.get(label, label))
            handles.append(label)

        # ---- 面元中值 + 分位带 ----
        stats_text = ""
        if show_surface:
            bin_m = max(float(bin_size_m), 0.5)
            centers, med, counts = binned_statistic(along_all, elev_all, bin_m,
                                                    statistic="median",
                                                    length_m=selection.corridor.length_m)
            _, p10, _ = binned_statistic(along_all, elev_all, bin_m, statistic="p10",
                                         length_m=selection.corridor.length_m)
            _, p90, _ = binned_statistic(along_all, elev_all, bin_m, statistic="p90",
                                         length_m=selection.corridor.length_m)
            ok = np.isfinite(med)
            if np.any(ok):
                self.ax.fill_between(centers[ok], p10[ok], p90[ok], color="#ff9800",
                                     alpha=0.13, linewidth=0, zorder=2,
                                     label="10~90% 分位带")
                self.ax.plot(centers[ok], med[ok], color="#e65100", linewidth=1.7,
                             zorder=3, label=f"面元中值 ({bin_m:g} m)")

        # ---- 统计信息 ----
        stats = elevation_statistics(elev_all)
        self.ax.set_xlim(0, max(float(selection.corridor.length_m), 1.0))
        ymin, ymax = float(np.min(elev_all)), float(np.max(elev_all))
        pad = max((ymax - ymin) * 0.06, 1.0)
        self.ax.set_ylim(ymin - pad, ymax + pad)
        self.ax.set_xlabel("沿剖切线距离 (m)", fontsize=9)
        self.ax.set_ylabel(selection.elev_label(elev_kind), fontsize=9)
        self.ax.tick_params(labelsize=8)
        title = (f"光子数 {elev_all.size:,}    中线长 {selection.corridor.length_m:,.0f} m    "
                 f"廊道 ±{selection.corridor.half_width_m:g} m    "
                 f"高程 {stats['min']:.2f}~{stats['max']:.2f} m, 中值 {stats['median']:.2f} m")
        self.ax.set_title(title + stride_note, fontsize=9, color="#333333", loc="left")
        if handles:
            self.ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85,
                           ncol=1, handletextpad=0.3, borderpad=0.3)

        # ---- 右侧高程直方图 ----
        if show_histogram:
            self.axh.hist(elev_all, bins=int(max(hist_bins, 10)), orientation="horizontal",
                          color="#5c7c8a", alpha=0.75)
            self.axh.set_xlabel("光子数", fontsize=8)
            self.axh.tick_params(labelsize=7)
            self.axh.tick_params(labelleft=False)
            self.axh.set_ylim(self.ax.get_ylim())
        else:
            self.axh.set_visible(False)

        self.canvas.draw_idle()
        self.on_status(selection.describe().replace("\n", "  |  "), "left")

    # ==================================================================
    def _format_coord(self, x, y):
        if self.selection is None:
            return f"距离 {x:,.0f} m, 高程 {y:,.1f} m"
        return f"沿迹距离 {x:,.1f} m, 高程 {y:,.2f} m"

    def toggle_histogram(self, visible: bool):
        self.axh.set_visible(bool(visible))
        self.canvas.draw_idle()

    def export_csv(self, path: str):
        if self.selection is None or len(self.selection) == 0:
            raise ValueError("当前没有剖面数据可导出")
        return self.selection.to_csv(path, self.elev_kind)

    def save_png(self, path: str):
        self.figure.savefig(path, dpi=170)
