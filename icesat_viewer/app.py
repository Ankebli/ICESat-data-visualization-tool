# -*- coding: utf-8 -*-
"""主窗体.

布局::

    ┌──────────────── 菜单 / 工具栏 ────────────────┐
    │ 数据面板 │        地图画布 (天地图底图)        │
    │ 波束选择 │──────────────────────────────────── │
    │ 信息     │  剖面视图 (高程-距离 + 直方图)       │
    └──────────────── 状态栏 ──────────────────────┘

交互: 工具栏切到"拾取剖面"后在地图上点击两点, 即按该直线与廊道半宽提取
ICESat 光子并绘制剖切面.
"""

from __future__ import annotations

import queue
import threading
import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from . import __version__, geo
from .basemap import (PROVIDERS, TIANDITU_TOKEN_PATTERN, TiandituQuotaError,
                      TiandituTokenError, TileDownloadError, TileEngine,
                      is_valid_token_format, mask_token)
from .corridor import CorridorGeometry
from .dem import (DEM_DATUMS, DemComparison, DemError, OpenTopographyDem, RasterDem,
                  TerrariumDem, adjust_dem_to_elev_kind, compare_with_profile,
                  corridor_bounds, open_dem_file, sample_along_corridor)
from .icesat_io import (BEAM_NAMES, BEAM_ORDER, CONF_CHANNELS, CONF_MODES,
                        ELEV_KINDS, OPEN_FILETYPES, PhotonSelection, ReadOptions,
                        compact_count, merge_selections, open_dataset,
                        short_source_name)
from .demo_data import generate_demo_atl03
from .mapview import MapCanvas
from .ml import (REFERENCE_CHOICES, REFERENCE_NONE, REF_DISPLAY_MODES,
                 model_from_selection, reference_model_for)
from .profileview import ProfileCanvas
from .settings import INDEX_CACHE_DIR, TILE_CACHE_DIR, Settings

DATASET_COLORS = ["#e91e63", "#3f51b5", "#009688", "#ff5722", "#795548",
                  "#607d8b", "#9c27b0", "#00bcd4"]
TDT_CONSOLE = "https://console.tianditu.gov.cn/api/key"


@dataclass
class DatasetEntry:
    """已加载的数据集 (含显示状态)."""

    dataset: object
    name: str
    color: str
    visible: bool = True


class IcesatViewerApp(tk.Tk):
    """ICESat 光子剖面查看器主窗口."""

    def __init__(self, settings: Optional[Settings] = None, dpi_scale: Optional[float] = None,
                 provider: Optional[str] = None, token: Optional[str] = None,
                 files: Optional[List[str]] = None):
        super().__init__()
        self.settings = settings or Settings()
        if provider:
            self.settings["provider"] = provider
        if token:
            self.settings.token = token

        if dpi_scale is None:
            # 高分屏适配: Tk 侧按屏幕 DPI 放大字体/控件; matplotlib 侧不需要手工放大
            # (matplotlib 会依据 Tk 的设备像素比自行调整 figure.dpi, 手工再乘会重复放大)
            try:
                screen_dpi = float(self.winfo_fpixels("1i"))
                if screen_dpi / 96.0 > 1.05:
                    self.tk.call("tk", "scaling", screen_dpi / 72.0)
            except Exception:
                pass
            dpi_scale = 1.0
        self.dpi = float(int(round(96 * max(0.8, min(dpi_scale, 2.2)))))
        self.title(f"ICESat 光子剖面查看器  v{__version__}  —  天地图底图 · 两点连线剖切")
        self.geometry("1500x950")
        self.minsize(1180, 720)

        self.entries: List[DatasetEntry] = []
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.tile_engine = TileEngine(TILE_CACHE_DIR, token=self.settings.token)
        self._profile_id = 0
        self._recompute_job = None
        self._busy_count = 0
        self._last_basemap_error = ""
        self.last_locate: dict = {}          # 最近一次 地图↔剖面 联动定位结果
        self.dem = None                      # 当前 DEM 数据源
        self.last_dem_compare: Optional[DemComparison] = None
        self.goto_lon_var = tk.StringVar(value="")
        self.goto_lat_var = tk.StringVar(value="")
        #: 界面状态变量 (菜单与工具栏共用, 必须在建菜单之前创建)
        self.pick_var = tk.BooleanVar(value=False)
        self.snap_var = tk.BooleanVar(value=bool(self.settings["snap_to_track"]))
        self.auto_zoom_var = tk.BooleanVar(
            value=bool(self.settings["auto_zoom_to_corridor"]))
        self.link_var = tk.BooleanVar(value=bool(self.settings["link_highlight"]))

        self._init_style()
        self._build_menu()
        self._build_toolbar()
        # 状态栏要先于可扩展的主体 pack, 否则窗口变小时会被主体挤没
        self._build_statusbar()
        self._build_body(dpi_scale)
        self._bind_keys()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_job = self.after(80, self._poll_events)
        self.after(300, self._startup_hint)
        if files:
            self.after(200, lambda: [self.load_file(f) for f in files])

    # ==================================================================
    # 界面搭建
    # ==================================================================
    def _init_style(self) -> None:
        style = ttk.Style(self)
        for theme in ("vista", "winnative", "clam", "default"):
            if theme in style.theme_names():
                try:
                    style.theme_use(theme)
                    break
                except Exception:
                    continue
        try:
            default_font = ("Microsoft YaHei UI", 9)
            self.option_add("*Font", default_font)
            style.configure("Treeview", rowheight=int(20 * self.dpi / 96))
            style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))
            style.configure("TLabelframe.Label", foreground="#37474f")
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="打开 ICESat 数据…", accelerator="Ctrl+O",
                           command=self.open_files)
        m_file.add_command(label="打开 DEM 数据 / 剖切对比…", command=self.open_dem_dialog)
        self.m_recent = tk.Menu(m_file, tearoff=0)
        m_file.add_cascade(label="最近打开", menu=self.m_recent)
        m_file.add_separator()
        m_file.add_command(label="导出剖面数据 (CSV)…", accelerator="Ctrl+S",
                           command=self.export_profile_csv)
        m_file.add_command(label="导出剖面图 (PNG)…", command=self.export_profile_png)
        m_file.add_command(label="导出地图视图 (PNG)…", command=self.export_map_png)
        m_file.add_separator()
        m_file.add_command(label="移除全部数据", command=self.remove_all)
        m_file.add_command(label="退出", command=self._on_close)
        menubar.add_cascade(label="文件", menu=m_file)

        m_view = tk.Menu(menubar, tearoff=0)
        self.provider_var = tk.StringVar(value=self.settings["provider"])
        for key, prov in PROVIDERS.items():
            m_view.add_radiobutton(label=prov.name, value=key, variable=self.provider_var,
                                   command=lambda k=key: self.set_provider(k))
        m_view.add_separator()
        self.show_tracks_var = tk.BooleanVar(value=bool(self.settings["show_tracks"]))
        m_view.add_checkbutton(label="显示 ICESat 轨迹", variable=self.show_tracks_var,
                               command=self._toggle_tracks)
        self.show_hist_var = tk.BooleanVar(value=True)
        m_view.add_checkbutton(label="显示高程直方图", variable=self.show_hist_var,
                               command=self._toggle_histogram)
        m_view.add_separator()
        m_view.add_command(label="放大", accelerator="+", command=lambda: self.map.zoom(1.4))
        m_view.add_command(label="缩小", accelerator="-", command=lambda: self.map.zoom(1 / 1.4))
        m_view.add_command(label="缩放到数据 / 剖面线", accelerator="Ctrl+D",
                           command=self.zoom_current_target)
        m_view.add_command(label="跳转到经纬度…", command=self.goto_dialog)
        menubar.add_cascade(label="视图", menu=m_view)

        m_profile = tk.Menu(menubar, tearoff=0)
        m_profile.add_checkbutton(label="拾取剖面模式", variable=self.pick_var,
                                  accelerator="Ctrl+P", command=self._toggle_pick_mode)
        m_profile.add_checkbutton(label="选点自动吸附到轨道", variable=self.snap_var,
                                  command=self._toggle_snap)
        m_profile.add_checkbutton(label="确定直线后自动缩放到该直线",
                                  variable=self.auto_zoom_var,
                                  command=self._toggle_auto_zoom)
        m_profile.add_checkbutton(label="地图 ↔ 剖面联动 (互点互定位)",
                                  variable=self.link_var, command=self._toggle_link)
        m_profile.add_command(label="计算/刷新剖面", accelerator="F5",
                              command=lambda: self.compute_profile(reason="手动刷新"))
        m_profile.add_command(label="清除拾取", accelerator="Esc", command=self.clear_pick)
        m_profile.add_separator()
        for width in (10, 25, 50, 100, 200):
            m_profile.add_command(label=f"廊道半宽 ±{width} m",
                                  command=lambda w=width: self.set_half_width(w))
        m_profile.add_command(label="自定义廊道半宽…", command=self.half_width_dialog)
        menubar.add_cascade(label="剖面", menu=m_profile)

        m_tool = tk.Menu(menubar, tearoff=0)
        m_tool.add_command(label="天地图密钥 / 底图设置…", command=self.open_settings)
        m_tool.add_command(label="打开瓦片缓存目录", command=lambda: self._open_dir(TILE_CACHE_DIR))
        m_tool.add_command(label="打开索引缓存目录", command=lambda: self._open_dir(INDEX_CACHE_DIR))
        m_tool.add_command(label="清空瓦片缓存", command=self.clear_tile_cache)
        m_tool.add_command(label="底图错误详情…", command=self.show_basemap_error)
        m_tool.add_command(label="DEM 剖切对比…", command=self.open_dem_dialog)
        m_tool.add_separator()
        m_tool.add_command(label="数据详情…", command=self.show_dataset_info)
        m_tool.add_command(label="生成演示数据 (合成 ATL03)…", command=self.make_demo_data)
        menubar.add_cascade(label="工具", menu=m_tool)

        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="使用说明", accelerator="F1", command=self.show_help)
        m_help.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=m_help)

        self.config(menu=menubar)
        self._refresh_recent_menu()

    # ------------------------------------------------------------------
    @staticmethod
    def _group_label(parent, text: str) -> ttk.Label:
        """统一的小组标题样式."""
        return ttk.Label(parent, text=text, foreground="#546e7a",
                         font=("Microsoft YaHei UI", 9, "bold"))

    @staticmethod
    def _sep(parent) -> None:
        ttk.Separator(parent, orient="vertical").pack(side="left", fill="y", padx=8)

    def _build_toolbar(self) -> None:
        """单行工具栏: 只放高频操作; 低频功能(定位/说明/密钥)收进菜单."""
        bar = ttk.Frame(self, padding=(6, 5, 6, 4))
        bar.pack(side="top", fill="x")

        ttk.Button(bar, text="打开数据", width=9, command=self.open_files).pack(side="left")
        ttk.Button(bar, text="打开 DEM", width=9,
                   command=self.open_dem_dialog).pack(side="left", padx=(4, 0))
        self._sep(bar)

        self._group_label(bar, "底图").pack(side="left")
        self.provider_combo = ttk.Combobox(bar, state="readonly", width=15,
                                           values=[PROVIDERS[k].name for k in PROVIDERS])
        self.provider_combo.set(PROVIDERS[self.provider_var.get()].name
                                if self.provider_var.get() in PROVIDERS
                                else list(PROVIDERS.values())[0].name)
        self.provider_combo.pack(side="left", padx=(4, 2))
        self.provider_combo.bind("<<ComboboxSelected>>", self._on_provider_combo)
        self.key_button = ttk.Button(bar, text="密钥…", width=6,
                                     command=self.open_settings)
        self.key_button.pack(side="left")
        self._sep(bar)

        self.pick_button = ttk.Checkbutton(bar, text="拾取剖面", variable=self.pick_var,
                                           command=self._toggle_pick_mode)
        self.pick_button.pack(side="left")
        ttk.Checkbutton(bar, text="吸附轨道", variable=self.snap_var,
                        command=self._toggle_snap).pack(side="left", padx=(6, 0))
        ttk.Label(bar, text="半宽 ±").pack(side="left", padx=(8, 0))
        self.half_width_var = tk.DoubleVar(value=float(self.settings["corridor_half_width_m"]))
        spin = ttk.Spinbox(bar, from_=1, to=2000, increment=5, width=5,
                           textvariable=self.half_width_var, command=self._on_half_width_change)
        spin.pack(side="left", padx=(2, 0))
        spin.bind("<Return>", lambda e: self._on_half_width_change())
        ttk.Label(bar, text="m").pack(side="left", padx=(0, 2))
        self._sep(bar)

        ttk.Button(bar, text="计算", width=5,
                   command=lambda: self.compute_profile("手动")).pack(side="left")
        ttk.Button(bar, text="清除", width=5, command=self.clear_pick).pack(
            side="left", padx=(4, 0))
        self.zoom_button = ttk.Button(bar, text="缩放到数据", width=10,
                                      command=self.zoom_current_target)
        self.zoom_button.pack(side="left", padx=(4, 0))

    # ------------------------------------------------------------------
    def _build_body(self, dpi_scale: float) -> None:
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(side="top", fill="both", expand=True, padx=6, pady=(0, 4))

        # ---------------- 左侧数据面板 ----------------
        left = ttk.Frame(main, width=276)
        main.add(left, weight=0)

        self._group_label(left, "数据").pack(anchor="w", pady=(2, 2))
        columns = ("vis", "name", "count")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=6,
                                 selectmode="browse")
        for col, text, width, anchor in (("vis", "显", 24, "center"),
                                         ("name", "数据", 78, "w"),
                                         ("count", "点数", 62, "e")):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor, stretch=(col == "name"))
        self.tree.pack(fill="x", expand=False)
        self.tree.bind("<Double-1>", lambda e: self.toggle_visible())
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._show_selected_info())

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(3, 6))
        ttk.Button(btns, text="显示/隐藏", width=9, command=self.toggle_visible).pack(side="left")
        ttk.Button(btns, text="缩放到", width=7,
                   command=self.zoom_current_target).pack(side="left", padx=2)
        ttk.Button(btns, text="移除", width=6, command=self.remove_selected).pack(side="left")

        # 波束过滤 (ATL03)
        self._group_label(left, "波束").pack(anchor="w", pady=(0, 2))
        self.beam_vars: Dict[str, tk.BooleanVar] = {}
        grid = ttk.Frame(left)
        grid.pack(fill="x", pady=(0, 6))
        for i, beam in enumerate(BEAM_ORDER):
            var = tk.BooleanVar(value=True)
            self.beam_vars[beam] = var
            ttk.Checkbutton(grid, text=BEAM_NAMES.get(beam, beam), variable=var,
                            command=self._schedule_recompute).grid(
                row=i // 2, column=i % 2, sticky="w", padx=(0, 6))

        self._group_label(left, "信息").pack(anchor="w", pady=(0, 2))
        self.info_text = ScrolledText(left, height=10, width=32, wrap="word",
                                      font=("Consolas", 8), state="disabled")
        self.info_text.pack(fill="both", expand=True)

        # ---------------- 右侧: 地图 + 剖面 ----------------
        right = ttk.PanedWindow(main, orient="vertical")
        main.add(right, weight=1)

        self.map = MapCanvas(right, self.tile_engine, provider=self.provider_var.get(),
                             dpi=self.dpi, on_status=self._on_map_status,
                             on_pick=self._on_pick, on_basemap_error=self._on_basemap_error,
                             on_locate=self._on_map_locate)
        self.map.snap_function = self._snap_lookup
        self.map.snap_enabled = bool(self.settings["snap_to_track"])
        self.map.snap_px = float(self.settings["snap_px"])
        right.add(self.map, weight=3)

        profile_frame = ttk.Frame(right)
        right.add(profile_frame, weight=2)
        self._build_profile_panel(profile_frame)
        self._main_paned = right
        self.after(500, self._layout_panes)

    def _layout_panes(self) -> None:
        """初次布局: 地图约占 6 成高, 剖面保留至少 300 px (绘图区不被挤压)."""
        try:
            self.update_idletasks()
            height = self._main_paned.winfo_height()
            if height > 200:
                pos = int(height * 0.58)
                pos = min(pos, max(int(height - 300), int(height * 0.45)))
                self._main_paned.sashpos(0, max(pos, 120))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _build_profile_panel(self, parent: ttk.Frame) -> None:
        """剖面工具条: 三行按功能分组, 每组以彩色小标题开头, 行宽均衡.

        第 1 行  剖面: 高程基准 / 置信度 / 通道 / 剔除噪声
        第 2 行  参考: 参考模型 / 显示方式   ┃  DEM: ☐剖切对比 + 设置…
        第 3 行  显示: ☐面元中值 / 分箱 [/] m / 点大小
        """
        bar = ttk.Frame(parent, padding=(6, 4, 6, 2))
        bar.pack(side="top", fill="x")

        # ---------------- 第 1 行 ----------------
        row1 = ttk.Frame(bar)
        row1.pack(side="top", fill="x")
        self._group_label(row1, "剖面").pack(side="left", padx=(0, 6))
        ttk.Label(row1, text="高程基准").pack(side="left")
        self.elev_var = tk.StringVar(value=str(self.settings["elev_kind"]))
        cb = ttk.Combobox(row1, textvariable=self.elev_var, values=list(ELEV_KINDS),
                          state="readonly", width=13)
        cb.pack(side="left", padx=(3, 8))
        cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_profile_display())

        ttk.Label(row1, text="置信度").pack(side="left")
        self.conf_var = tk.StringVar(value=str(self.settings["conf_mode"]))
        cb2 = ttk.Combobox(row1, textvariable=self.conf_var, values=list(CONF_MODES),
                           state="readonly", width=10)
        cb2.pack(side="left", padx=(3, 8))
        cb2.bind("<<ComboboxSelected>>", lambda e: self._schedule_recompute())

        ttk.Label(row1, text="通道").pack(side="left")
        self.channel_var = tk.StringVar(value=str(self.settings["conf_channel"]))
        cb3 = ttk.Combobox(row1, textvariable=self.channel_var,
                           values=["自动"] + list(CONF_CHANNELS), state="readonly", width=8)
        cb3.pack(side="left", padx=(3, 6))
        cb3.bind("<<ComboboxSelected>>", lambda e: self._schedule_recompute())
        self.afterpulse_var = tk.BooleanVar(value=bool(self.settings["remove_afterpulse"]))
        ttk.Checkbutton(row1, text="剔除噪声", variable=self.afterpulse_var,
                        command=self._schedule_recompute).pack(side="left", padx=(2, 0))

        # ---------------- 第 2 行: 参考模型 + DEM 对比 ----------------
        row2 = ttk.Frame(bar)
        row2.pack(side="top", fill="x", pady=(3, 0))
        self._group_label(row2, "参考").pack(side="left", padx=(0, 6))
        self.reference_var = tk.StringVar(
            value=str(self.settings["reference_model"] or REFERENCE_NONE))
        cb_ref = ttk.Combobox(row2, textvariable=self.reference_var,
                              values=[REFERENCE_NONE] + list(REFERENCE_CHOICES),
                              state="readonly", width=17)
        cb_ref.pack(side="left", padx=(0, 8))
        cb_ref.bind("<<ComboboxSelected>>", lambda e: self._refresh_profile_display())
        self.reference_combo = cb_ref          # 供界面回归测试检查可见性

        ttk.Label(row2, text="方式").pack(side="left")
        self.ref_mode_var = tk.StringVar(
            value=str(self.settings["reference_display"] or "高亮参考信号"))
        cb_mode = ttk.Combobox(row2, textvariable=self.ref_mode_var,
                               values=list(REF_DISPLAY_MODES), state="readonly", width=9)
        cb_mode.pack(side="left", padx=(3, 6))
        cb_mode.bind("<<ComboboxSelected>>", lambda e: self._refresh_profile_display())
        self.ref_mode_combo = cb_mode

        self._sep(row2)
        self._group_label(row2, "DEM").pack(side="left", padx=(0, 6))
        self.dem_var = tk.BooleanVar(value=bool(self.settings["dem_enabled"]))
        self.dem_check = ttk.Checkbutton(
            row2, text="剖切对比", variable=self.dem_var,
            command=self._on_dem_toggle)
        self.dem_check.pack(side="left", padx=(0, 4))
        ttk.Button(row2, text="设置…", width=6,
                   command=self.open_dem_dialog).pack(side="left")

        # ---------------- 第 3 行: 显示选项 ----------------
        row3 = ttk.Frame(bar)
        row3.pack(side="top", fill="x", pady=(3, 3))
        self._group_label(row3, "显示").pack(side="left", padx=(0, 6))
        self.surface_var = tk.BooleanVar(value=bool(self.settings["show_surface"]))
        ttk.Checkbutton(row3, text="面元中值", variable=self.surface_var,
                        command=self._refresh_profile_display).pack(side="left", padx=(0, 10))
        ttk.Label(row3, text="分箱").pack(side="left")
        self.bin_var = tk.DoubleVar(value=float(self.settings["bin_size_m"]))
        ttk.Spinbox(row3, from_=1, to=500, increment=5, width=5, textvariable=self.bin_var,
                    command=self._refresh_profile_display).pack(side="left", padx=(3, 0))
        ttk.Label(row3, text="m").pack(side="left", padx=(0, 12))
        ttk.Label(row3, text="点大小").pack(side="left")
        self.point_size_var = tk.DoubleVar(value=float(self.settings["point_size"]))
        ttk.Spinbox(row3, from_=0.3, to=10, increment=0.5, width=4,
                    textvariable=self.point_size_var,
                    command=self._refresh_profile_display).pack(side="left", padx=(3, 0))

        self.profile = ProfileCanvas(parent, dpi=self.dpi, on_status=self._on_map_status,
                                     on_locate=self._on_profile_locate)
        self.profile.pack(side="top", fill="both", expand=True)

    # ------------------------------------------------------------------
    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self, padding=(6, 3))
        bar.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="就绪 — 请先打开 ICESat 数据 (文件 → 打开)")
        ttk.Style(self).configure("Error.TLabel", foreground="#c62828")
        self.status_label = ttk.Label(bar, textvariable=self.status_var, anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(bar, mode="determinate", length=150, maximum=100)
        self.progress.pack(side="right", padx=6)
        self.tile_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.tile_var, width=44, anchor="e").pack(side="right")
        self.cursor_var = tk.StringVar(value="—")
        ttk.Label(bar, textvariable=self.cursor_var, width=30, anchor="e").pack(side="right")

    def _bind_keys(self) -> None:
        self.bind("<Control-o>", lambda e: self.open_files())
        self.bind("<Control-s>", lambda e: self.export_profile_csv())
        self.bind("<Control-d>", lambda e: self.zoom_current_target())
        self.bind("<Control-p>", lambda e: self._toggle_pick_mode())
        self.bind("<F5>", lambda e: self.compute_profile("F5"))
        self.bind("<F1>", lambda e: self.show_help())
        self.bind("<Escape>", lambda e: self.clear_pick())
        self.bind("<plus>", lambda e: self.map.zoom(1.4))
        self.bind("<minus>", lambda e: self.map.zoom(1 / 1.4))
        self.bind("<Delete>", lambda e: self.remove_selected())

    # ==================================================================
    # 状态 / 进度
    # ==================================================================
    def _on_map_status(self, text: str, where: str = "left") -> None:
        if where == "cursor":
            self.cursor_var.set(text)
        elif where == "tiles":
            self.tile_var.set(text)
        else:
            self.status_var.set(text)

    def _status(self, text: str, error: bool = False) -> None:
        self.status_var.set(("⚠ " + text) if error else text)
        try:
            self.status_label.configure(style="Error.TLabel" if error else "TLabel")
        except Exception:
            pass

    def _progress_begin(self, text: str) -> None:
        self._busy_count += 1
        self.progress.configure(mode="indeterminate")
        self.progress.start(60)
        self._status(text)

    def _progress_set(self, frac: float, text: str) -> None:
        self.status_var.set(f"{text}  ({frac * 100:.0f}%)")

    def _progress_end(self, text: Optional[str] = None) -> None:
        self._busy_count = max(0, self._busy_count - 1)
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress["value"] = 0
        if text:
            self._status(text)

    # ==================================================================
    # 事件循环 (worker 线程 -> 主线程)
    # ==================================================================
    def _poll_events(self) -> None:
        try:
            if not self.winfo_exists():
                return
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    frac, text, _name = payload
                    self._progress_set(frac, text)
                elif kind == "loaded":
                    self._on_dataset_loaded(payload)
                elif kind == "profile":
                    rid, sel, corridor = payload
                    if rid == self._profile_id:
                        self._progress_end()
                        self._on_profile_ready(sel, corridor)
                elif kind == "error":
                    message, detail = payload
                    self._progress_end()
                    self._status("出错: " + message.splitlines()[0])
                    messagebox.showerror("错误", f"{message}\n\n{detail[-1500:]}", parent=self)
                elif kind == "basemap_error":
                    self._on_basemap_error(payload)
                elif kind == "dem_loaded":
                    self._on_dem_loaded(payload)
                elif kind == "dem_error":
                    self._progress_end()
                    self._status(f"DEM 加载失败 — {payload}", error=True)
                elif kind == "status":
                    self._status(payload)
        except queue.Empty:
            pass
        except tk.TclError:
            return
        try:
            self._poll_job = self.after(80, self._poll_events)
        except tk.TclError:
            pass

    # ==================================================================
    # 数据加载
    # ==================================================================
    def open_files(self) -> None:
        initial = self.settings.get("last_dir") or ""
        paths = filedialog.askopenfilenames(parent=self, title="打开 ICESat 数据",
                                            filetypes=OPEN_FILETYPES,
                                            initialdir=initial or None)
        for path in paths:
            self.load_file(path)

    def load_file(self, path: str) -> None:
        path = str(Path(path))
        self._progress_begin(f"正在打开 {Path(path).name} …")
        self.settings["last_dir"] = str(Path(path).parent)
        self._push_recent(path)

        def work():
            try:
                def progress(frac, msg):
                    self.events.put(("progress", (frac, f"{Path(path).name}: {msg}", path)))

                dataset = open_dataset(path, cache_dir=INDEX_CACHE_DIR, progress=progress)
                self.events.put(("progress", (1.0, f"{dataset.name}: 建立覆盖索引 …", path)))
                dataset.coverage_index()
                self.events.put(("loaded", dataset))
            except Exception as exc:
                self.events.put(("error", (f"加载失败: {Path(path).name}\n{exc}",
                                           traceback.format_exc())))

        threading.Thread(target=work, daemon=True, name="load").start()

    def _on_dataset_loaded(self, dataset) -> None:
        color = DATASET_COLORS[len(self.entries) % len(DATASET_COLORS)]
        entry = DatasetEntry(dataset=dataset, name=dataset.name, color=color)
        self.entries.append(entry)
        self.tree.insert("", "end", iid=str(id(entry)),
                         values=("✔", short_source_name(entry.name),
                                 compact_count(dataset.point_count)))
        self.map.set_datasets(self.entries)
        if len(self.entries) == 1:
            self.map.zoom_to_data()
        self.tree.selection_set(str(id(entry)))
        self._show_selected_info()
        self._progress_end(f"已加载 {dataset.name} — {dataset.kind}, "
                           f"{dataset.point_count:,} 个点/光子")
        self.settings.save()

    # ------------------------------------------------------------------
    def _selected_entry(self) -> Optional[DatasetEntry]:
        sel = self.tree.selection()
        if not sel:
            return None
        target = sel[0]
        for entry in self.entries:
            if str(id(entry)) == target:
                return entry
        return None

    def _show_selected_info(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        lines = [f"{k}: {v}" for k, v in entry.dataset.summary()]
        lines.append(f"路径: {entry.dataset.path}")
        self._set_info("\n".join(lines))

    def _set_info(self, text: str) -> None:
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")

    def toggle_visible(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        entry.visible = not entry.visible
        self.tree.set(str(id(entry)), "vis", "✔" if entry.visible else "✘")
        self.map.set_datasets(self.entries)

    def remove_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        try:
            entry.dataset.close()
        except Exception:
            pass
        self.entries = [e for e in self.entries if e is not entry]
        try:
            self.tree.delete(str(id(entry)))
        except Exception:
            pass
        self.map.set_datasets(self.entries)
        self.profile.clear()
        self._status(f"已移除 {entry.name}")

    def remove_all(self) -> None:
        for entry in list(self.entries):
            try:
                entry.dataset.close()
            except Exception:
                pass
        self.entries = []
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.map.set_datasets(self.entries)
        self.profile.clear()
        self._set_info("")
        self._status("已移除全部数据")

    def show_dataset_info(self) -> None:
        if not self.entries:
            messagebox.showinfo("数据详情", "尚未加载任何数据。", parent=self)
            return
        blocks = []
        for i, entry in enumerate(self.entries, 1):
            blocks.append(f"【{i}】{entry.name}   ({'显示' if entry.visible else '隐藏'})")
            blocks.extend(f"    {k}: {v}" for k, v in entry.dataset.summary())
            blocks.append("")
        self._text_window("数据详情", "\n".join(blocks))

    # ==================================================================
    # 剖面
    # ==================================================================
    def _toggle_pick_mode(self) -> None:
        mode = "pick" if self.pick_var.get() else "pan"
        self.map.set_mode(mode)
        if mode == "pick":
            self._status("剖面拾取模式: 左键点击第 1 个点 (会自动吸附到最近轨道); "
                         "中键拖动平移, 右键/Esc 取消")
            try:
                self.map.canvas.get_tk_widget().focus_set()
            except Exception:
                pass
        else:
            self._status("浏览模式: 中键拖动平移, 滚轮缩放, 左键双击缩放到数据")

    def set_half_width(self, value: float) -> None:
        self.half_width_var.set(float(value))
        self._on_half_width_change()

    def _on_half_width_change(self) -> None:
        try:
            value = float(self.half_width_var.get())
        except Exception:
            return
        value = max(0.5, min(value, 5000.0))
        self.map.set_half_width(value)
        self.settings["corridor_half_width_m"] = value
        self._schedule_recompute()

    def half_width_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("自定义廊道半宽")
        win.transient(self)
        ttk.Label(win, text="廊道半宽 ± (m):").pack(side="left", padx=8, pady=12)
        var = tk.DoubleVar(value=float(self.half_width_var.get()))
        entry = ttk.Entry(win, textvariable=var, width=10)
        entry.pack(side="left")
        entry.focus_set()

        def apply():
            self.set_half_width(var.get())
            win.destroy()

        ttk.Button(win, text="确定", command=apply).pack(side="left", padx=8)
        entry.bind("<Return>", lambda e: apply())

    def _on_pick(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> None:
        corridor = CorridorGeometry(p0, p1, float(self.half_width_var.get()))
        self.map.set_corridor(corridor)
        self._update_zoom_button()
        # 确定直线后: 立即把视野框到这条剖切线上 (可在 剖面 菜单里关闭)
        if self.auto_zoom_var.get():
            self.map.zoom_to_corridor(corridor, margin=0.20)
        warn = "" if self.snap_var.get() else self._coverage_warning(p0, p1)
        self.compute_profile(reason="拾取", corridor=corridor, prefix=warn)

    # ------------------------------------------------------------------
    def zoom_current_target(self) -> None:
        """缩放到数据范围; 已确定剖切线时缩放到该直线位置."""
        if self.map.corridor is not None:
            self.map.zoom_to_corridor(self.map.corridor, margin=0.20)
            p0, p1 = self.map.corridor.p0, self.map.corridor.p1
            self._status(f"已缩放到剖面线范围 — 长 {self.map.corridor.length_m:,.0f} m, "
                         f"A({p0[0]:.5f}, {p0[1]:.5f}) → B({p1[0]:.5f}, {p1[1]:.5f})")
        else:
            self.map.zoom_to_data()
            self._status("已缩放到数据范围 (尚未确定剖切线)")

    def _update_zoom_button(self) -> None:
        """按钮文字随当前状态切换: 有剖切线时提示可缩放到该直线."""
        try:
            self.zoom_button.configure(
                text="缩放到剖面线" if self.map.corridor is not None else "缩放到数据")
        except Exception:
            pass

    def _toggle_auto_zoom(self) -> None:
        enabled = bool(self.auto_zoom_var.get())
        self.settings["auto_zoom_to_corridor"] = enabled
        self._status("已开启: 确定剖切线后自动缩放到该直线"
                     if enabled else "已关闭: 确定剖切线后保持当前视野")

    # ------------------------------------------------------------------
    # 地图 <-> 剖面 联动
    # ------------------------------------------------------------------
    def _on_map_locate(self, lon: float, lat: float) -> None:
        """浏览模式下点击地图轨迹 -> 在剖面上高亮对应位置."""
        corridor = self.map.corridor
        selection = self.profile.selection
        if corridor is None or selection is None or len(selection) == 0:
            self._status("请先用两点生成剖切面, 再点击轨迹查看对应剖面位置")
            return
        if not self.link_var.get():
            return
        along, cross = corridor.project(lon, lat)
        along = float(np.clip(along, 0.0, corridor.length_m))
        hw = float(corridor.half_width_m)
        cross = float(np.clip(cross, -hw, hw))
        # 在光子集合里找 (沿迹, 横向) 最接近点击位置的光子
        dist2 = (np.asarray(selection.along) - along) ** 2 \
            + (np.asarray(selection.cross) - cross) ** 2
        idx = int(np.argmin(dist2))
        idx = int(np.clip(idx, 0, len(selection) - 1))
        sel_along = float(selection.along[idx])
        elev = float(selection.display_elev(self.elev_var.get())[idx])
        self.profile.set_marker(sel_along, elev,
                                label=f"{sel_along:,.0f} m / {elev:,.2f} m")
        self.map.set_link_marker(float(selection.lon[idx]), float(selection.lat[idx]),
                                 label=f"沿迹 {sel_along:,.0f} m")
        self.last_locate = {"source": "map", "along": sel_along, "elev": elev,
                            "lon": float(selection.lon[idx]),
                            "lat": float(selection.lat[idx])}
        self._status(f"地图点击 → 剖面定位: 沿迹 {sel_along:,.0f} m, 高程 {elev:,.2f} m "
                     f"(横向偏距 {float(selection.cross[idx]):+.1f} m)")

    def _on_profile_locate(self, along: float, elev: float) -> None:
        """点击剖面 -> 地图 zoom 到该位置."""
        corridor = self.map.corridor
        selection = self.profile.selection
        if corridor is None or selection is None or len(selection) == 0:
            return
        if not self.link_var.get():
            return
        sel = selection
        # 先把点击位置吸附到实际光子 (更符合"点光子看位置")
        photon = self.profile.nearest_photon(along, elev)
        if photon is not None:
            along = photon[0]
        along = float(np.clip(along, 0.0, corridor.length_m))
        lon, lat = geo.GEOD.forward(corridor.p0[0], corridor.p0[1], corridor.azimuth, along)
        idx = int(np.argmin(np.abs(np.asarray(sel.along) - along)))
        self.profile.set_marker(along, photon[1] if photon else None,
                                label=f"{along:,.0f} m")
        self.map.set_link_marker(float(sel.lon[idx]), float(sel.lat[idx]),
                                 label=f"沿迹 {along:,.0f} m")
        span = float(np.clip(corridor.length_m * 0.10, 1500.0, 20000.0))
        self.map.zoom_to_point(float(lon), float(lat), span_m=span)
        self.last_locate = {"source": "profile", "along": along,
                            "elev": photon[1] if photon else elev,
                            "lon": float(lon), "lat": float(lat)}
        self._status(f"剖面点击 → 地图定位: 沿迹 {along:,.0f} m → "
                     f"{float(lon):.5f}°E, {float(lat):.5f}°N (视野 {span / 1000:.1f} km)")

    def _toggle_link(self) -> None:
        enabled = bool(self.link_var.get())
        self.settings["link_highlight"] = enabled
        if not enabled:
            self.profile.clear_marker()
            self.map.clear_link_marker()
        self._status("已开启地图↔剖面联动 (互点互定位)" if enabled else "已关闭联动高亮")

    # ------------------------------------------------------------------
    def _toggle_snap(self) -> None:
        enabled = bool(self.snap_var.get())
        self.map.set_snap(enabled)
        self.settings["snap_to_track"] = enabled
        self._status("选点已开启自动吸附: 点击处会落到最近的 ICESat 轨道点上"
                     if enabled else "选点吸附已关闭: 使用鼠标原始点击位置")

    def _snap_lookup(self, lon: float, lat: float, tol_m: float,
                     prefer_label: Optional[str] = None
                     ) -> Tuple[float, float, str, bool, Optional[str]]:
        """把点击位置吸附到最近的 ICESat 轨道点.

        返回 ``(经度, 纬度, 说明文字, 是否吸附成功, 轨迹标签)``;
        ``prefer_label`` 非空时优先吸附到同一波束 (容差放宽 3 倍), 便于两点落在同一条轨道上.
        """
        def nearest(label: Optional[str] = None):
            best = None
            for entry in self.entries:
                if not entry.visible:
                    continue
                index = entry.dataset.coverage_index()
                if index is None:
                    continue
                dist, slon, slat, lab = index.nearest_point(lon, lat, label=label)
                if np.isfinite(dist) and (best is None or dist < best[0]):
                    best = (dist, slon, slat, lab, entry)
            return best

        if prefer_label:
            same = nearest(prefer_label)
            if same is not None and same[0] <= tol_m * 3.0:
                return (same[1], same[2],
                        f"已吸附到 {same[3]} 轨道 (与第一点同波束, 偏移 {same[0]:.1f} m)",
                        True, same[3])
        best = nearest()
        if best is None:
            return lon, lat, "没有可用于吸附的 ICESat 轨道 (请先打开数据)", False, None
        dist, slon, slat, lab, _entry = best
        if dist <= tol_m:
            return (slon, slat,
                    f"已吸附到 {lab or '轨道'} (偏移 {dist:.1f} m ≤ 容差 {tol_m:.0f} m)",
                    True, lab)
        return (lon, lat,
                f"未吸附: 距最近轨迹 {dist:.0f} m > 容差 {tol_m:.0f} m "
                f"(请靠近轨迹点击, 或放大/关闭吸附)", False, None)

    def _coverage_warning(self, p0, p1) -> str:
        tol = float(self.settings["coverage_tolerance_m"])
        problems = []
        for name, pt in (("A", p0), ("B", p1)):
            best = float("inf")
            for entry in self.entries:
                if not entry.visible:
                    continue
                idx = entry.dataset.coverage_index()
                if idx is not None:
                    best = min(best, idx.nearest_distance_m(pt[0], pt[1]))
            if np.isfinite(best) and best > tol:
                problems.append(f"点{name}距最近 ICESat 轨迹约 {best / 1000:.1f} km")
        if problems:
            return "⚠ " + "; ".join(problems) + f" (超出 {tol / 1000:.1f} km 覆盖容差), 结果可能为空"
        return ""

    def _read_options(self) -> ReadOptions:
        beams = [b for b, var in self.beam_vars.items() if var.get()] or list(BEAM_ORDER)
        return ReadOptions(half_width_m=float(self.half_width_var.get()),
                           beams=tuple(beams),
                           conf_mode=self.conf_var.get(),
                           conf_channel=self.channel_var.get(),
                           remove_afterpulse=bool(self.afterpulse_var.get()))

    def _schedule_recompute(self) -> None:
        """选项变化后延迟重算 (防抖)."""
        if self.map.corridor is None:
            return
        if self._recompute_job is not None:
            try:
                self.after_cancel(self._recompute_job)
            except Exception:
                pass
        self._recompute_job = self.after(280, lambda: self.compute_profile("选项变化"))

    def compute_profile(self, reason: str = "", corridor: Optional[CorridorGeometry] = None,
                        prefix: str = "") -> None:
        self._recompute_job = None
        corridor = corridor or self.map.corridor
        if corridor is None or len(self.map.picks) < 2:
            self._status("请先在拾取模式下点击两个点")
            return
        entries = [e for e in self.entries if e.visible]
        if not entries:
            self._status("没有可见的数据集")
            return
        options = self._read_options()

        self._profile_id += 1
        rid = self._profile_id
        self._progress_begin(f"正在提取廊道光子 ({reason}) …")

        def work():
            try:
                selections = []
                for i, entry in enumerate(entries):
                    self.events.put(("progress", (i / max(len(entries), 1),
                                                   f"读取 {entry.name} …", entry.name)))
                    selections.append(entry.dataset.select(corridor, options))
                merged = merge_selections(selections, corridor)
                self.events.put(("profile", (rid, merged, corridor)))
            except Exception as exc:
                self.events.put(("error", (f"剖面计算失败: {exc}", traceback.format_exc())))

        threading.Thread(target=work, daemon=True, name="profile").start()
        if prefix:
            self._status(prefix)

    def _on_profile_ready(self, selection: PhotonSelection, corridor: CorridorGeometry) -> None:
        self.settings["conf_mode"] = self.conf_var.get()
        self.settings["conf_channel"] = self.channel_var.get()
        self.settings["elev_kind"] = self.elev_var.get()
        self._refresh_profile_display(selection)
        self.map.clear_link_marker()          # 新剖面 -> 清空旧的联动标记
        if len(selection) == 0:
            self._status("该直线/廊道内没有 ICESat 光子 — 请检查两点是否位于覆盖区, "
                         "或加大廊道半宽 / 放宽置信度筛选")
        else:
            self._set_info("\n".join(selection.info_lines()))

    def _refresh_profile_display(self, selection: Optional[PhotonSelection] = None) -> None:
        selection = selection if selection is not None else self.profile.selection
        if selection is None:
            return
        highlight, note = self._reference_overlay(selection)
        mode = REF_DISPLAY_MODES.get(self.ref_mode_var.get(), "highlight")
        dem_curve = self._dem_display_curve(selection)
        self.profile.set_selection(
            selection, elev_kind=self.elev_var.get(),
            point_size=float(self.point_size_var.get()),
            show_surface=bool(self.surface_var.get()),
            bin_size_m=float(self.bin_var.get()),
            hist_bins=int(self.settings["hist_bins"]),
            show_histogram=bool(self.show_hist_var.get()),
            highlight_mask=highlight, ref_mode=mode, note=note,
            dem_curve=dem_curve)
        if dem_curve is not None and self.last_dem_compare is not None:
            cmp = self.last_dem_compare
            self._status((cmp.summary() + (f" — {cmp.note}" if cmp.note else ""))[:200])
        elif note:
            self._status(f"{note}  —  " + self.status_var.get()[:130])
        self.settings.update({"elev_kind": self.elev_var.get(),
                              "point_size": float(self.point_size_var.get()),
                              "show_surface": bool(self.surface_var.get()),
                              "bin_size_m": float(self.bin_var.get()),
                              "reference_model": self.reference_var.get(),
                              "reference_display": self.ref_mode_var.get()})

    # ------------------------------------------------------------------
    # M0: 官方参考模型 (ATL03 signal_conf) —— 作为"已训练好的基线"
    # ------------------------------------------------------------------
    def _reference_overlay(self, selection: Optional[PhotonSelection]
                           ) -> tuple:
        """计算参考模型的信号掩膜与说明文字.

        返回 ``(highlight_mask, note)``; 参考模型被关闭或产品缺少 ``signal_conf``
        时返回 ``(None, "")`` / ``(None, 说明)``, 剖面保持原样。
        """
        self.last_reference = {}
        if selection is None or len(selection) == 0:
            return None, ""
        choice = self.reference_var.get()
        model = reference_model_for(choice)
        if model is None:
            return None, ""
        try:
            frame = model_from_selection(selection, self.elev_var.get())
            result = model.result(frame)
        except Exception as exc:                      # 模型异常不影响主流程
            self._status(f"参考模型计算失败 — {type(exc).__name__}: {exc}", error=True)
            return None, ""
        if result is None:
            note = "该数据无 ATL03 signal_conf, 参考模型不可用"
            self.last_reference = {"choice": choice, "available": False}
            return None, note
        mask = np.asarray(result.label, dtype=bool)
        n_sig = int(mask.sum())
        total = max(len(selection), 1)
        ratio = 100.0 * n_sig / total
        note = (f"参考 {choice.replace('ATL03 ', '')}: {n_sig:,}/{total:,} ({ratio:.1f}%)")
        if n_sig >= total:
            # 用户当前的置信度筛选已经把所有光子都筛成"信号", 参考模型自然看不出差异
            note += ("  ⚠ 当前筛选已全部命中参考口径, 参考模型无可淡化 —— "
                     "把「置信度」改为「全部」即可看到原始噪声被淡化")
        self.last_reference = {
            "choice": choice, "available": True, "n_signal": n_sig,
            "n_total": int(len(selection)), "ratio": ratio,
            "all_signal": bool(n_sig >= total),
            "channel": result.meta.get("channel"), "min_conf": result.meta.get("min_conf"),
        }
        return mask, note

    def clear_pick(self) -> None:
        self.map.clear_pick()
        self.map.clear_link_marker()
        self.profile.clear()
        self._profile_id += 1
        self._update_zoom_button()
        self._status("已清除拾取, 请重新点击两点")

    # ==================================================================
    # 导出
    # ==================================================================
    def export_profile_csv(self) -> None:
        if self.profile.selection is None or len(self.profile.selection) == 0:
            messagebox.showinfo("导出剖面", "当前没有剖面数据。请先拾取两点生成剖面。", parent=self)
            return
        path = filedialog.asksaveasfilename(parent=self, title="导出剖面数据",
                                            defaultextension=".csv",
                                            initialfile="icesat_profile.csv",
                                            filetypes=[("CSV 文件", "*.csv")])
        if not path:
            return
        try:
            out = self.profile.export_csv(path)
            self._status(f"剖面数据已导出: {out}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc), parent=self)

    def export_profile_png(self) -> None:
        if self.profile.selection is None:
            messagebox.showinfo("保存剖面图", "当前没有剖面图。", parent=self)
            return
        path = filedialog.asksaveasfilename(parent=self, title="保存剖面图",
                                            defaultextension=".png",
                                            initialfile="icesat_profile.png",
                                            filetypes=[("PNG 图片", "*.png")])
        if path:
            self.profile.save_png(path)
            self._status(f"剖面图已保存: {path}")

    def export_map_png(self) -> None:
        path = filedialog.asksaveasfilename(parent=self, title="保存地图视图",
                                            defaultextension=".png",
                                            initialfile="icesat_map.png",
                                            filetypes=[("PNG 图片", "*.png")])
        if path:
            self.map.save_png(path)
            self._status(f"地图视图已保存: {path}")

    # ==================================================================
    # 视图 / 底图
    # ==================================================================
    def set_provider(self, key: str, persist: bool = True) -> None:
        if key not in PROVIDERS:
            return
        provider = PROVIDERS[key]
        if provider.requires_token and not self.settings.has_token:
            # 不弹模态框, 只在状态栏提示 (用户主动点工具栏"天地图密钥…"即可配置)
            self._status("天地图底图需要 tk 密钥 — 点工具栏「天地图密钥…」填写后即可使用",
                         error=True)
        self.provider_var.set(key)
        self.provider_combo.set(provider.name)
        self.map.set_provider(key)
        if persist:                    # persist=False 用于临时降级, 不写入配置
            self.settings["provider"] = key
            self.settings.save()

    def _on_provider_combo(self, event=None) -> None:
        name = self.provider_combo.get()
        for key, prov in PROVIDERS.items():
            if prov.name == name:
                self.set_provider(key)
                return

    def _toggle_tracks(self) -> None:
        self.map.show_tracks = bool(self.show_tracks_var.get())
        self.settings["show_tracks"] = self.map.show_tracks
        self.map._draw_overlays()
        self.map.canvas.draw_idle()

    def _toggle_histogram(self) -> None:
        self.profile.toggle_histogram(bool(self.show_hist_var.get()))
        self._refresh_profile_display()

    def goto_dialog(self) -> None:
        """跳转到经纬度 (低频功能, 收进对话框)."""
        win = tk.Toplevel(self)
        win.title("跳转到经纬度")
        win.transient(self)
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="经度 (°E)").grid(row=0, column=0, sticky="w")
        lon_var = tk.StringVar(value=self.goto_lon_var.get() or "120.20")
        ttk.Entry(frame, textvariable=lon_var, width=14).grid(row=0, column=1, padx=4, pady=3)
        ttk.Label(frame, text="纬度 (°N)").grid(row=1, column=0, sticky="w")
        lat_var = tk.StringVar(value=self.goto_lat_var.get() or "31.20")
        ttk.Entry(frame, textvariable=lat_var, width=14).grid(row=1, column=1, padx=4, pady=3)
        ttk.Label(frame, text="视野宽度 (m)").grid(row=2, column=0, sticky="w")
        span_var = tk.DoubleVar(value=30000.0)
        ttk.Entry(frame, textvariable=span_var, width=14).grid(row=2, column=1, padx=4, pady=3)

        def apply():
            try:
                lon = float(lon_var.get())
                lat = float(lat_var.get())
            except Exception:
                messagebox.showinfo("跳转", "请输入有效的经纬度, 例如 120.20 / 31.20",
                                    parent=win)
                return
            self.goto_lon_var.set(lon_var.get())
            self.goto_lat_var.set(lat_var.get())
            self.map.zoom_to_point(lon, lat, span_m=float(span_var.get()))
            self._status(f"已定位到 {lon:.5f}°E, {lat:.5f}°N")
            win.destroy()

        ttk.Button(frame, text="前往", command=apply).grid(row=3, column=1, sticky="e", pady=6)

    # ==================================================================
    # 工具 / 对话框
    # ==================================================================
    def open_settings(self) -> None:
        """天地图密钥配置窗口 (自己配置自己的 key, 无效密钥直接抛异常)."""
        win = tk.Toplevel(self)
        win.title("天地图密钥配置")
        win.transient(self)
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="天地图 tk 密钥:", font=("Microsoft YaHei UI", 9, "bold")).grid(
            row=0, column=0, sticky="w")
        token_var = tk.StringVar(value=self.settings.token)
        token_entry = ttk.Entry(frame, textvariable=token_var, width=46, show="•")
        token_entry.grid(row=0, column=1, columnspan=2, sticky="we", padx=6, pady=3)
        show_var = tk.BooleanVar(value=False)

        def toggle_show():
            token_entry.configure(show="" if show_var.get() else "•")

        ttk.Checkbutton(frame, text="显示", variable=show_var,
                        command=toggle_show).grid(row=0, column=3, sticky="w")

        fmt_label = ttk.Label(frame, text="", foreground="#546e7a")
        fmt_label.grid(row=1, column=0, columnspan=4, sticky="w")

        def refresh_format_hint(*_a):
            token = token_var.get().strip()
            if not token:
                fmt_label.configure(text="当前未配置密钥 (未配置时自动使用免密钥备用底图)",
                                    foreground="#546e7a")
            elif is_valid_token_format(token):
                fmt_label.configure(text=f"格式正确 (32 位十六进制) · tk={mask_token(token)}",
                                    foreground="#2e7d32")
            else:
                fmt_label.configure(text="格式不正确: 天地图 tk 应为 32 位十六进制字符",
                                    foreground="#c62828")

        token_var.trace_add("write", refresh_format_hint)
        refresh_format_hint()

        ttk.Label(frame, foreground="#546e7a", wraplength=600, justify="left",
                  text=("• 免费申请: 天地图开发者控制台 → 创建“浏览器端”应用 → 复制 tk 密钥; "
                        "也可设置环境变量 TIANDITU_TK (优先级最高), 或命令行 --token;\n"
                        "• 保存后会立即用新密钥刷新底图; 若密钥无效, 程序会抛出 "
                        "TiandituTokenError 并提示具体原因。")).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(4, 2))
        link = ttk.Label(frame, text=TDT_CONSOLE, foreground="#1565c0", cursor="hand2")
        link.grid(row=3, column=0, columnspan=4, sticky="w")
        link.bind("<Button-1>", lambda e: webbrowser.open(TDT_CONSOLE))

        ttk.Label(frame, text="默认底图:").grid(row=4, column=0, sticky="w", pady=(8, 2))
        prov_var = tk.StringVar(value=PROVIDERS[self.provider_var.get()].name
                                if self.provider_var.get() in PROVIDERS
                                else list(PROVIDERS.values())[0].name)
        ttk.Combobox(frame, textvariable=prov_var, state="readonly", width=30,
                     values=[p.name for p in PROVIDERS.values()]).grid(
            row=4, column=1, columnspan=3, sticky="w", pady=(8, 2))

        ttk.Label(frame, text="瓦片缓存:").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Label(frame, text=str(TILE_CACHE_DIR), foreground="#546e7a").grid(
            row=5, column=1, columnspan=3, sticky="w", pady=2)

        status = ttk.Label(frame, text="", foreground="#546e7a", wraplength=600,
                           justify="left")
        status.grid(row=6, column=0, columnspan=4, sticky="w", pady=(6, 2))

        def run_check(token: str, on_ok=None, on_fail=None):
            """在线校验密钥: 失败时把异常原样抛出并显示."""
            status.configure(text="正在连接天地图校验密钥 …", foreground="#546e7a")
            win.update_idletasks()

            def work():
                engine = TileEngine(TILE_CACHE_DIR, token=token, timeout=12)
                try:
                    engine.validate_provider("tianditu_vec")   # 失败会抛异常
                    msg = f"密钥可用 ✓  tk={mask_token(token)}"
                    self.after(0, lambda: (status.configure(text=msg, foreground="#2e7d32"),
                                           on_ok and on_ok()))
                except Exception as exc:                       # 原样抛出并展示
                    detail = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                    self.after(0, lambda: (
                        status.configure(text="密钥校验失败 → " + detail, foreground="#c62828"),
                        messagebox.showerror(
                            "天地图密钥无效",
                            f"密钥校验失败, 已抛出异常:\n\n{detail}\n\n"
                            f"• 请确认密钥完整 (32 位十六进制) 且已开通“Web 端服务”;\n"
                            f"• 申请地址: {TDT_CONSOLE}",
                            parent=win),
                        on_fail and on_fail()))
                finally:
                    engine.close()

            threading.Thread(target=work, daemon=True).start()

        def current_provider_key() -> str:
            for key, prov in PROVIDERS.items():
                if prov.name == prov_var.get():
                    return key
            return self.settings["provider"]

        def apply_settings() -> None:
            self.settings.token = token_var.get().strip()
            self.settings["provider"] = current_provider_key()
            self.settings.save()
            self._apply_token(self.settings.token)     # 换 key 即刻联网校验
            self.provider_var.set(self.settings["provider"])
            self.provider_combo.set(PROVIDERS[self.settings["provider"]].name)
            self.map.set_provider(self.settings["provider"])

        def test_token():
            run_check(token_var.get().strip())

        def save(verify: bool = False):
            token = token_var.get().strip()
            if token and not is_valid_token_format(token):
                exc = TiandituTokenError(
                    f"天地图 tk 密钥格式不正确: {token!r} (应为 32 位十六进制字符)")
                status.configure(text=f"密钥格式错误 → {type(exc).__name__}: {exc}",
                                 foreground="#c62828")
                messagebox.showerror("密钥格式错误",
                                     f"已抛出异常:\n\n{type(exc).__name__}: {exc}", parent=win)
                token_entry.focus_set()
                return
            apply_settings()
            if verify and token:
                run_check(token, on_ok=win.destroy)
            else:
                win.destroy()
                self._status("天地图密钥已保存" if token else "已清除密钥 — 使用免密钥备用底图")

        def clear_token():
            token_var.set("")
            apply_settings()
            win.destroy()
            self._status("已清除天地图密钥 — 自动使用免密钥备用底图")

        btns = ttk.Frame(frame)
        btns.grid(row=7, column=0, columnspan=4, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="校验密钥", command=test_token).pack(side="left", padx=3)
        ttk.Button(btns, text="清除", command=clear_token).pack(side="left", padx=3)
        ttk.Button(btns, text="仅保存", command=lambda: save(False)).pack(side="left", padx=3)
        ttk.Button(btns, text="校验并保存", command=lambda: save(True)).pack(side="left", padx=3)
        ttk.Button(btns, text="取消", command=win.destroy).pack(side="left")
        token_entry.focus_set()

    # ------------------------------------------------------------------
    def _on_basemap_error(self, exc: Exception) -> None:
        """底图错误 (密钥无效 / 限流): **只写状态栏, 绝不弹模态对话框**.

        自动弹出的模态框会打断用户操作 (启动时尤其恼人), 因此这里仅记录并提示;
        需要看完整异常时可点 工具 → 底图错误详情, 或主动点
        "天地图密钥…→ 校验密钥" (那条路径才弹框, 因为是你主动点的)。

        * 限流 (HTTP 418): 天地图临时不可用, 自动降级到免密钥底图, 保证还能看图;
        * 密钥无效/过期: 只提示, 不动底图选择 (由用户决定换 key 还是换底图)。
        """
        detail = f"{type(exc).__name__}: {exc}"
        self._last_basemap_error = detail
        if isinstance(exc, TiandituQuotaError):
            if self.provider_var.get().startswith("tianditu"):
                # 仅本次会话降级 (不写入配置): 下次启动仍优先尝试天地图
                self.set_provider("esri_img", persist=False)
                self._status("天地图访问受限 (HTTP 418 限流/配额), 已临时改用 Esri 影像底图 — "
                             "稍后可在「底图」下拉框切回天地图", error=True)
                return
            self._status(f"底图限流 — {detail}", error=True)
            return
        self._status(f"底图不可用 — {detail} (工具→底图错误详情 可查看)", error=True)

    def show_basemap_error(self) -> None:
        """用户主动查看底图错误详情 (这里才用对话框)."""
        if not self._last_basemap_error:
            messagebox.showinfo("底图错误详情", "当前没有底图错误。", parent=self)
            return
        if messagebox.askyesno("底图错误详情",
                               f"{self._last_basemap_error}\n\n是否现在重新配置天地图密钥?",
                               parent=self):
            self.open_settings()

    # ------------------------------------------------------------------
    def _apply_token(self, token: str) -> None:
        """应用(切换)天地图密钥: 更新引擎并**立即在线校验**, 无效则抛异常提示.

        仅仅刷新底图是不够的 —— 视野内的瓦片可能已在缓存里, 会掩盖失效的密钥,
        因此这里显式走一次联网校验 (``validate_provider`` 会绕过缓存).
        """
        self.tile_engine.set_token(token)
        self._last_basemap_error = ""
        if len((token or "").strip()) >= 8:
            self._validate_token(token)
        else:
            self._status("已清除天地图密钥 — 自动使用免密钥备用底图")

    def _validate_token(self, token: str) -> None:
        """后台在线校验密钥; 失败时把异常原样送到界面上."""
        def work():
            engine = TileEngine(TILE_CACHE_DIR, token=token, timeout=12)
            try:
                engine.validate_provider("tianditu_vec")
                self.events.put(("status",
                                 f"天地图密钥校验通过 (tk={mask_token(token)})"))
            except Exception as exc:
                self.events.put(("basemap_error", exc))
            finally:
                engine.close()

        threading.Thread(target=work, daemon=True, name="token-check").start()

    def _validate_token_on_startup(self) -> None:
        """启动后在后台校验一次密钥 (静默: 只写状态栏, 绝不弹模态框)."""
        if self.settings.has_token:
            self._validate_token(self.settings.token)

    # ------------------------------------------------------------------
    # DEM 剖切对比
    # ------------------------------------------------------------------
    def _on_dem_toggle(self) -> None:
        enabled = bool(self.dem_var.get())
        self.settings["dem_enabled"] = enabled
        if enabled and self.dem is None:
            self._status("DEM 对比已开启, 但尚未加载 DEM — 点「设置…」(或 文件 → 打开 DEM) "
                         "选择文件或在线获取", error=True)
        elif enabled:
            self._status(f"DEM 对比已开启: {self.dem.describe()}")
        else:
            self._status("DEM 对比已关闭 — 仅显示 ICESat 剖面")
        self._refresh_profile_display()

    def _dem_display_curve(self, selection: Optional[PhotonSelection]) -> Optional[tuple]:
        """按当前剖面与 DEM 生成对比曲线 ``(along, z, label, stats_text)``."""
        self.last_dem_compare = None
        if (self.dem is None or selection is None or len(selection) == 0
                or not self.dem_var.get()):
            return None
        corridor = selection.corridor
        step = float(self.settings["dem_step_m"] or 2.0)
        try:
            along, z = sample_along_corridor(self.dem, corridor, step_m=step)
        except Exception as exc:
            self._status(f"DEM 采样失败 — {type(exc).__name__}: {exc}", error=True)
            return None
        if along.size == 0:
            return None
        if not self.dem.covers(*corridor_bounds(corridor)) and self.dem.bounds:
            self._status("注意: DEM 未完全覆盖该剖切线, 部分位置无 DEM 值", error=True)
        z_adj, datum_note = adjust_dem_to_elev_kind(
            along, z, selection, self.elev_var.get(),
            dem_is_orthometric=self.dem.is_orthometric,
            offset_m=float(self.settings["dem_offset_m"] or 0.0))

        # ICESat 侧: 参考模式开启时用信号子集, 否则用全部光子
        kind = self.elev_var.get()
        elev_i = selection.display_elev(kind)
        along_i = np.asarray(selection.along, "f8")
        mask = self.profile._highlight_mask
        if mask is not None and mask.size == elev_i.size and bool(np.any(mask)):
            along_i, elev_i = along_i[mask], elev_i[mask]
        try:
            cmp = compare_with_profile(along_i, elev_i, along, z_adj,
                                       bin_size_m=float(self.bin_var.get() or 20.0))
        except Exception as exc:
            cmp = DemComparison(note=f"对比统计失败: {exc}")
        self.last_dem_compare = cmp
        parts = [cmp.summary()]
        if datum_note:
            parts.append(datum_note)
        if cmp.note and cmp.n_points > 0:
            parts.append(cmp.note)
        stats_text = "  |  ".join(p for p in parts if p)
        label = f"DEM: {self.dem.name if hasattr(self.dem, 'name') else 'DEM'}"
        if float(self.settings["dem_offset_m"] or 0.0):
            label += f" (偏移 {float(self.settings['dem_offset_m']):+.2f} m)"
        return (along, z_adj, label, stats_text)

    # ------------------------------------------------------------------
    def open_dem_dialog(self) -> None:
        """DEM 加载/获取对话框."""
        win = tk.Toplevel(self)
        win.title("DEM 剖切对比")
        win.transient(self)
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)

        status = ttk.Label(frame, text="", foreground="#37474f", wraplength=560,
                           justify="left")
        status.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        def refresh_status(msg: str = "") -> None:
            cur = self.dem.describe() if self.dem is not None else "未加载 DEM"
            status.configure(text=(msg + "\n" if msg else "") + "当前: " + cur)

        # ---- 本地文件 ----
        ttk.Label(frame, text="① 本地 DEM 文件", font=("Microsoft YaHei UI", 9, "bold")
                  ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 2))

        def pick_file() -> None:
            path = filedialog.askopenfilename(
                parent=win, title="打开 DEM 文件",
                filetypes=[("DEM 栅格", "*.tif *.tiff *.img *.vrt *.asc *.hgt *.grd"),
                           ("全部文件", "*.*")],
                initialdir=str(Path(self.settings["dem_last_file"]).parent)
                if self.settings["dem_last_file"] else None)
            if not path:
                return
            self._load_dem(lambda: open_dem_file(
                path, datum=self._dem_datum(), bounds_lonlat=self._corridor_bounds()))
            self.settings["dem_last_file"] = path

        ttk.Button(frame, text="打开文件…", command=pick_file).grid(
            row=2, column=0, sticky="w", pady=2)
        ttk.Label(frame, foreground="#78909c",
                  text="GeoTIFF / IMG / VRT / ASCII Grid 等 (rasterio 读取)").grid(
            row=2, column=1, columnspan=2, sticky="w", padx=6)

        # ---- 在线获取 ----
        ttk.Label(frame, text="② 在线获取 (需先在图上拾取两点确定剖切线范围)",
                  font=("Microsoft YaHei UI", 9, "bold")).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(10, 2))

        def fetch_terrarium() -> None:
            b = self._corridor_bounds()
            if b is None:
                messagebox.showinfo("需要剖切线",
                                    "请先在地图上拾取两点生成剖切线, 再在线获取 DEM。",
                                    parent=win)
                return
            self._load_dem(lambda: TerrariumDem(
                b, cache_dir=TILE_CACHE_DIR.parent / "dem", progress=None))

        ttk.Button(frame, text="AWS Terrain Tiles (免密钥)",
                   command=fetch_terrarium).grid(row=4, column=0, sticky="w", pady=2)
        ttk.Label(frame, foreground="#78909c",
                  text="SRTM/ASTER 融合, 免注册, 按走廊范围拼瓦片").grid(
            row=4, column=1, columnspan=2, sticky="w", padx=6)

        ttk.Label(frame, text="OpenTopography API key").grid(row=5, column=0, sticky="w",
                                                            pady=2)
        key_var = tk.StringVar(value=str(self.settings["opentopo_api_key"] or ""))
        ttk.Entry(frame, textvariable=key_var, width=34).grid(row=5, column=1, sticky="w",
                                                              padx=6)
        demtype_var = tk.StringVar(value=str(self.settings["opentopo_demtype"] or "SRTMGL1"))
        ttk.Combobox(frame, textvariable=demtype_var, width=12, state="readonly",
                     values=["SRTMGL1", "SRTMGL3", "AW3D30", "COP30", "NASADEM"]).grid(
            row=5, column=2, sticky="w")
        ttk.Label(frame, foreground="#78909c",
                  text="免费申请: portal.opentopography.org (可留空, 用上面的免密钥源)").grid(
            row=6, column=0, columnspan=3, sticky="w")

        def fetch_opentopo() -> None:
            b = self._corridor_bounds()
            if b is None:
                messagebox.showinfo("需要剖切线",
                                    "请先在地图上拾取两点生成剖切线, 再在线获取 DEM。",
                                    parent=win)
                return
            self.settings["opentopo_api_key"] = key_var.get().strip()
            self.settings["opentopo_demtype"] = demtype_var.get()
            self._load_dem(lambda: OpenTopographyDem(
                b, api_key=key_var.get().strip(), demtype=demtype_var.get(),
                datum=self._dem_datum(), cache_dir=TILE_CACHE_DIR.parent / "dem"))

        ttk.Button(frame, text="下载 DEM", command=fetch_opentopo).grid(
            row=7, column=0, sticky="w", pady=2)

        # ---- 选项 ----
        opts = ttk.LabelFrame(frame, text="垂直基准与采样", padding=6)
        opts.grid(row=8, column=0, columnspan=3, sticky="we", pady=(10, 4))
        ttk.Label(opts, text="DEM 视为").grid(row=0, column=0, sticky="w")
        self.dem_datum_var = tk.StringVar(value=str(self.settings["dem_datum"]))
        cb = ttk.Combobox(opts, textvariable=self.dem_datum_var, values=list(DEM_DATUMS),
                          state="readonly", width=18)
        cb.grid(row=0, column=1, sticky="w", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_dem_option_change())
        ttk.Label(opts, text="垂直偏移 (m)").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.dem_offset_var = tk.DoubleVar(value=float(self.settings["dem_offset_m"] or 0.0))
        ttk.Spinbox(opts, from_=-100, to=100, increment=0.5, width=7,
                    textvariable=self.dem_offset_var,
                    command=self._on_dem_option_change).grid(row=0, column=3, sticky="w",
                                                            padx=4)
        ttk.Label(opts, text="采样间距 (m)").grid(row=0, column=4, sticky="w", padx=(10, 0))
        self.dem_step_var = tk.DoubleVar(value=float(self.settings["dem_step_m"] or 2.0))
        ttk.Spinbox(opts, from_=0.5, to=200, increment=1, width=6,
                    textvariable=self.dem_step_var,
                    command=self._on_dem_option_change).grid(row=0, column=5, sticky="w",
                                                            padx=4)
        ttk.Label(opts, foreground="#78909c",
                  text="ATL03 为椭球高; SRTM 类 DEM 为正交高, 程序会按 geoid 自动换算 "
                       "(太湖处 N≈+7.1 m)").grid(row=1, column=0, columnspan=6, sticky="w",
                                                 pady=(4, 0))

        # ---- 底部按钮 ----
        btns = ttk.Frame(frame)
        btns.grid(row=9, column=0, columnspan=3, sticky="e", pady=(8, 0))

        def export_curve() -> None:
            if self.last_dem_compare is None or self.profile._dem_curve is None:
                messagebox.showinfo("导出对比曲线", "当前没有 DEM 对比结果。", parent=win)
                return
            path = filedialog.asksaveasfilename(parent=win, title="导出 DEM 对比曲线",
                                                defaultextension=".csv",
                                                initialfile="dem_compare.csv",
                                                filetypes=[("CSV 文件", "*.csv")])
            if not path:
                return
            along, z = self.profile._dem_curve[0], self.profile._dem_curve[1]
            with open(path, "w", encoding="utf-8-sig", newline="") as fh:
                fh.write("along_m,dem_elev_m\n")
                for a, e in zip(along, z):
                    fh.write(f"{a:.2f},{'' if not np.isfinite(e) else f'{e:.3f}'}\n")
            self._status(f"DEM 对比曲线已导出: {path}")

        def clear_dem() -> None:
            self.dem = None
            self.last_dem_compare = None
            refresh_status("已清除 DEM")
            self._refresh_profile_display()

        ttk.Button(btns, text="导出对比曲线 CSV", command=export_curve).pack(side="left",
                                                                            padx=3)
        ttk.Button(btns, text="清除 DEM", command=clear_dem).pack(side="left", padx=3)
        ttk.Button(btns, text="关闭", command=win.destroy).pack(side="left")
        refresh_status()

    # ------------------------------------------------------------------
    def _dem_datum(self) -> str:
        return str(self.settings["dem_datum"] or DEM_DATUMS[0])

    def _corridor_bounds(self):
        corridor = self.map.corridor
        if corridor is None:
            return None
        return corridor_bounds(corridor, half_width_m=max(self.map.half_width_m, 25.0))

    def _on_dem_option_change(self) -> None:
        self.settings["dem_datum"] = self.dem_datum_var.get()
        try:
            self.settings["dem_offset_m"] = float(self.dem_offset_var.get())
            self.settings["dem_step_m"] = float(self.dem_step_var.get())
        except Exception:
            pass
        if self.dem is not None:
            self.dem.datum = self.dem_datum_var.get()
        self._refresh_profile_display()

    def _load_dem(self, factory) -> None:
        """在线程里加载 DEM (下载可能较慢), 完成后刷新对比曲线."""
        def work():
            try:
                dem = factory()
                self.events.put(("dem_loaded", dem))
            except DemError as exc:
                self.events.put(("dem_error", str(exc)))
            except Exception as exc:
                self.events.put(("dem_error", f"{type(exc).__name__}: {exc}"))

        self._progress_begin("正在加载 DEM …")
        threading.Thread(target=work, daemon=True, name="dem-load").start()

    def _on_dem_loaded(self, dem) -> None:
        self.dem = dem
        self.dem_var.set(True)
        self.settings["dem_enabled"] = True
        self._progress_end(f"DEM 已加载: {dem.describe()}")
        self._refresh_profile_display()

    def clear_tile_cache(self) -> None:
        if not messagebox.askyesno("清空瓦片缓存", "确定要删除磁盘上的全部地图瓦片缓存吗?",
                                   parent=self):
            return
        removed = self.tile_engine.clear_cache()
        self.map.refresh_tiles(force=True)
        self._status(f"已清空瓦片缓存 ({removed} 个文件)")

    def make_demo_data(self) -> None:
        path = filedialog.asksaveasfilename(parent=self, title="生成演示数据",
                                            defaultextension=".h5",
                                            initialfile="demo_ATL03.h5",
                                            filetypes=[("HDF5", "*.h5")])
        if not path:
            return
        self._progress_begin("正在生成演示数据 …")
        try:
            generate_demo_atl03(path)
            self._progress_end(f"演示数据已生成: {path}")
            if messagebox.askyesno("演示数据", "已生成合成 ATL03 演示文件。\n是否立即打开?",
                                   parent=self):
                self.load_file(path)
        except Exception as exc:
            self._progress_end()
            messagebox.showerror("生成失败", str(exc), parent=self)

    def _open_dir(self, path: Path) -> None:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            import os
            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showinfo("打开目录", f"{path}\n({exc})", parent=self)

    def _text_window(self, title: str, text: str) -> None:
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("880x600")
        box = ScrolledText(win, wrap="word", font=("Microsoft YaHei UI", 9))
        box.pack(fill="both", expand=True)
        box.insert("1.0", text)
        box.configure(state="disabled")

    def show_help(self) -> None:
        self._text_window("使用说明", HELP_TEXT)

    def show_about(self) -> None:
        messagebox.showinfo(
            "关于",
            f"ICESat 光子剖面查看器 v{__version__}\n\n"
            "• 天地图 (tianditu.gov.cn) WMTS 底图, 支持免密钥备用底图\n"
            "• ICESat-2 ATL03/ATL06/ATL08 与 ICESat/GLAS GLAH14 读取\n"
            "• 两点连线剖切: 任点两点即得该直线廊道内的光子剖面\n\n"
            f"投影: Web Mercator (EPSG:3857)   大地线: {geo.GEOD.backend}\n"
            f"运行环境: Python {__import__('sys').version.split()[0]}",
            parent=self)

    # ==================================================================
    # 启动 / 关闭
    # ==================================================================
    def _startup_hint(self) -> None:
        """启动提示: 全部静默, 只写状态栏 (绝不弹模态框打断用户)."""
        # DEM 开关上次是开着的, 但 DEM 本身不随配置保存 -> 复位为关闭, 避免"勾着却没曲线"
        if self.dem_var.get() and self.dem is None:
            self.dem_var.set(False)
            self.settings["dem_enabled"] = False
        provider = self.provider_var.get()
        if PROVIDERS[provider].requires_token and not self.settings.has_token:
            self.set_provider("esri_img")
            self._status("未配置天地图 tk 密钥, 已临时使用免密钥备用底图 — "
                         "可在 工具栏「密钥…」 中填写后切换")
            return
        if not self.settings.has_token:
            self._status("就绪 — 请打开 ICESat 数据 (Ctrl+O)")
            return
        self._status(f"就绪 — 底图: {PROVIDERS[provider].name} "
                     f"(tk={mask_token(self.settings.token)})")
        if PROVIDERS[provider].requires_token:
            # 仅在真正使用天地图时才后台校验密钥; 失败只写状态栏
            self._validate_token_on_startup()

    def _push_recent(self, path: str) -> None:
        recent = [p for p in self.settings.get("recent_files", []) if p != path]
        recent.insert(0, path)
        self.settings["recent_files"] = recent[:8]
        self._refresh_recent_menu()

    def _refresh_recent_menu(self) -> None:
        self.m_recent.delete(0, "end")
        recent = self.settings.get("recent_files", []) or []
        if not recent:
            self.m_recent.add_command(label="(空)", state="disabled")
            return
        for path in recent:
            self.m_recent.add_command(label=str(path), command=lambda p=path: self.load_file(p))

    def _on_close(self) -> None:
        try:
            self.settings.save()
        except Exception:
            pass
        for entry in self.entries:
            try:
                entry.dataset.close()
            except Exception:
                pass
        try:
            self.tile_engine.close()
        except Exception:
            pass
        for job in (getattr(self, "_poll_job", None), self._recompute_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
        try:
            self.map.stop_polling()
        except Exception:
            pass
        self.destroy()


HELP_TEXT = """ICESat 光子剖面查看器 — 使用说明
================================================================

一、打开 ICESat 数据
    文件 → 打开 ICESat 数据 (Ctrl+O), 可多选。
    支持:
      • ICESat-2  ATL03 (.h5)          —— 光子产品, 惰性分块读取, 2 GB 文件秒级打开
      • ICESat-2  ATL06 / ATL08 (.h5)  —— 自动识别经纬度与高程字段
      • ICESat/GLAS GLAH14 (.h5)
      • CSV / TXT                       —— 需含 lat/lon/elev 或 lon,lat,elev 列
      • LAS / LAZ 点云
    打开后地图上会显示该文件 6 个波束的轨迹, 可据此找到覆盖区域。

二、底图
    视图 → 或工具栏"底图"下拉框切换:
      • 天地图·影像/矢量/地形 (含注记)  —— 需要 tk 密钥
      • Esri 影像/地形                  —— 免密钥, 无偏移, 未配置密钥时自动启用
    天地图密钥: 工具 → 天地图密钥 / 底图设置, 或设置环境变量 TIANDITU_TK。
    免费申请地址: https://console.tianditu.gov.cn/api/key

三、两点连线剖切 (核心功能)
    1) 工具栏勾选"拾取剖面 (点两点)" (或按 Ctrl+P);
    2) 在 ICESat 覆盖区内单击第 1 个点 —— 默认开启"吸附轨道": 点击处会自动落到
       最近的 ICESat 轨道点上 (状态栏显示吸附到的波束与偏移量), 并画出 A 标记;
       灰色 × 是鼠标原始点击位置, 虚线连着吸附后的真实取点;
    3) 移动鼠标会显示预览连线, 单击第 2 个点 —— 若第 1 点已吸附到某波束,
       第 2 点会优先吸附到同一条轨道 (容差放宽 3 倍), 使两点尽量落在同一条轨道的
       沿迹方向上; 随后立即提取该直线下方廊道内的全部 ICESat 光子并绘制剖切面:
          左图: 横轴 = 沿剖切线距离 (m), 纵轴 = 高程 (m), 按波束着色
          右图: 该高程区间内的光子数直方图
       若两点离轨道太远 (超出吸附容差), 状态栏会提示"距最近轨迹 xx m", 并沿用原始点击位置。
    吸附容差 = 25 像素对应的地面距离 (随缩放自适应, 15 m ~ 5 km), 可在工具栏
    "吸附轨道" 复选框随时开关。

四、剖面参数
    • 廊道半宽 ±m : 剖切线两侧的采样带宽。水面/平坦地表 10~25 m 即可;
      地形起伏或想纳入更多光子时可加大到 50~200 m。
    • 置信度      : 全部 / 中等及以上 (signal_conf ≥ 3) / 仅高置信 (== 4)。
    • 通道        : 置信度所属地表类型 (陆地/海洋/海冰/陆地冰/内陆水体), "自动"取各通道最大值。
    • 剔除噪声标记: 剔除 ATL03 quality_ph ≠ 0 的光子 (afterpulse / 脉冲响应等)。
    • 高程基准    : 椭球高 (WGS84) / 正交高 (减 geophys_corr/geoid) / 相对高 (减中值)。
    • 面元中值    : 按沿迹分箱 (默认 20 m) 的中值曲线与 10~90% 分位带, 便于看清水面/地表起伏。
    • 参考模型    : 用 ATL03 产品自带的 signal_conf_ph (NASA 官方信号识别) 作为"已训练好的
      基线": 参考判定为非信号的光子会**淡化成灰色**, 标题显示"参考信号数/总数(占比)"。
      口径可选 高置信(≥4) / 中等及以上(≥3) / 指定通道(陆地·海洋·内陆水体) 等;
      选"无 (不评估)"则恢复普通显示。CSV/LAS 等无 signal_conf 的产品会自动提示不可用。
    • DEM 剖切对比: 勾选后, 剖面图上叠加 DEM 沿该直线的地形曲线 (蓝色), 并给出
      Δ = ICESat 中值 − DEM 的均值/标准差、RMSE 与采样点数; 点「设置…」可选择本地
      栅格 (tif/img/asc/hgt 等)、免密钥地形瓦片或 OpenTopography API。
    • 点大小      : 剖面散点的像素大小 (0.3~10), 光子密集时调小更易看清结构。

五、地图操作
    中键拖动   : 平移底图
    滚轮       : 以光标为中心缩放
    左键单击   : 拾取剖面点 (拾取模式) / 联动定位 (浏览模式, 见下)
    左键双击   : 缩放 (有剖切线时缩放到该直线, 否则缩放到数据范围)
    右键 / Esc : 清除拾取
    上/右侧刻度为经纬度; 状态栏实时显示光标经纬度、底图级别与瓦片状态。

六、地图 ↔ 剖面 联动 (默认开启, 可在 剖面 菜单关闭)
    • 在浏览模式 (不勾选"拾取剖面") 下, 点击地图轨迹上的任意一点:
      剖面图上会出现红色竖虚线 + 亮点, 标出该位置对应的光子
      (状态栏给出沿迹距离、高程与横向偏距);
      地图上同时用红色圆环标出该光子的实际位置。
    • 反过来, 点击剖面图上的任意一点 (或某个光子):
      地图立即 zoom 到该点对应的经纬度 (沿迹距离换算成经纬度),
      并在剖面与地图上同时高亮该位置, 便于核对"剖面上的起伏对应地图上哪里"。

    关于"缩放到数据": 选完两点、确定剖切线之后, 该按钮会变为"缩放到剖面线",
    点击 (或 Ctrl+D / 双击地图) 即把视野精确框到这条 A→B 直线上 (含廊道宽度);
    清除拾取后自动恢复为"缩放到数据", 即整个数据集范围。
    确定两点时也会自动执行一次该缩放 (可在 剖面 菜单里关闭)。

七、导出
    文件 → 导出剖面数据 (CSV, 含 lon/lat/高程/沿迹距离/横向偏距/波束)
    文件 → 导出剖面图 (PNG) / 导出地图视图 (PNG)

八、性能提示
    • ATL03 首次打开会建立段索引 (约 1~3 s) 并缓存到 ~/.icesat_tdt_viewer/index,
      再次打开同一文件瞬时完成;
    • 瓦片缓存位于 ~/.icesat_tdt_viewer/tiles, 可在 工具 菜单中清理;
    • 单次剖面默认最多读取 400 万光子, 超出会随机抽稀并在剖面标题中注明。
"""
