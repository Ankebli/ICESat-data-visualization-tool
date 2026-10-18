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
                 max_plot_points: int = 600000):
        super().__init__(master)
        self.on_status = on_status or (lambda text, where="left": None)
        self.max_plot_points = int(max_plot_points)
        self.selection: Optional[PhotonSelection] = None
        self.elev_kind = ELEV_KINDS[0]

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
        self.ax.format_coord = self._format_coord
        self._empty("请在地图上点击两点以生成 ICESat 光子剖切面")

    # ==================================================================
    def _empty(self, message: str):
        self.ax.clear()
        self.axh.clear()
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
