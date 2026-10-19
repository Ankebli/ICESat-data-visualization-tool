# -*- coding: utf-8 -*-
"""配置管理: 天地图 tk 密钥与显示参数持久化.

密钥优先级: 环境变量 ``TIANDITU_TK`` > ``config.json`` > 空.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .basemap import is_valid_token_format

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.json"
DEFAULT_CACHE_DIR = Path.home() / ".icesat_tdt_viewer"
TILE_CACHE_DIR = DEFAULT_CACHE_DIR / "tiles"
INDEX_CACHE_DIR = DEFAULT_CACHE_DIR / "index"

#: 可选: 在这里写死一枚天地图 tk 密钥给所有机器共用 (留空即可)。
#: 推荐把密钥放在本地 ``config.json`` (已被 .gitignore 忽略) 或环境变量
#: ``TIANDITU_TK`` 中, 避免密钥被提交到公开仓库。
PROJECT_TIANDITU_TOKEN = ""

#: 默认配置 (写入 config.json 的全部字段)
DEFAULTS = {
    "tianditu_token": PROJECT_TIANDITU_TOKEN,
    "provider": "tianditu_img",      # 底图: 见 basemap.PROVIDERS
    "corridor_half_width_m": 25.0,   # 剖面廊道半宽 (米)
    "elev_kind": "椭球高",            # 椭球高 / 正交高(减geoid) / 相对高(减中值)
    "conf_mode": "全部",              # 全部 / 中等及以上 / 仅高置信
    "conf_channel": "自动",           # 自动 / 陆地 / 海洋 / 海冰 / 陆地冰 / 内陆水体
    "remove_afterpulse": False,      # 是否剔除 quality_ph != 0 的光子
    "point_size": 2.0,               # 剖面散点大小
    "show_surface": True,            # 是否叠加面元中值线
    "bin_size_m": 20.0,              # 面元中值 / 直方图沿迹分箱 (米)
    "hist_bins": 200,                # 高程直方图分箱数
    "show_tracks": True,             # 显示 ICESat 轨迹
    "snap_to_track": True,           # 选点自动吸附到最近轨道点
    "snap_px": 25.0,                 # 吸附容差 (像素, 随缩放自适应)
    "auto_zoom_to_corridor": True,   # 确定剖切线后自动缩放到该直线范围
    "link_highlight": True,          # 地图 ↔ 剖面 联动高亮 (互点互定位)
    "reference_model": "ATL03 高置信 (≥4)",   # M0: 参考模型口径 (官方 signal_conf)
    "reference_display": "高亮参考信号",       # M0: 显示方式 (高亮/仅信号/全部)
    "dem_enabled": False,            # 是否开启 DEM 剖切对比
    "dem_datum": "正交高 (EGM96 等)",  # DEM 垂直基准
    "dem_offset_m": 0.0,             # DEM 手调垂直偏移 (米)
    "dem_step_m": 2.0,               # DEM 沿剖切线采样间距 (米)
    "dem_last_file": "",             # 上次打开的 DEM 文件
    "opentopo_api_key": "",          # OpenTopography API key (可选)
    "opentopo_demtype": "SRTMGL1",   # OpenTopography DEM 产品
    "max_plot_points": 600000,       # 单次绘制的最大光子数 (超出则抽稀显示)
    "coverage_tolerance_m": 2000.0,  # 判定"位于覆盖区"的距离阈值 (米)
    "last_dir": "",                  # 上次打开数据的目录
    "recent_files": [],              # 最近打开的数据文件
}


class Settings:
    """极简配置对象 (dict + json 落盘)."""

    def __init__(self, path: Path = CONFIG_PATH):
        self.path = Path(path)
        self.data = dict(DEFAULTS)
        self.load()

    # ------------------------------------------------------------------
    def load(self) -> None:
        try:
            if self.path.exists():
                with open(self.path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    for key, value in loaded.items():
                        if key in DEFAULTS:
                            self.data[key] = value
                        else:
                            self.data[key] = value
        except Exception:
            pass  # 配置损坏时静默使用默认值

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def __getitem__(self, key):
        return self.data.get(key, DEFAULTS.get(key))

    def __setitem__(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, DEFAULTS.get(key, default))

    def update(self, mapping: dict) -> None:
        self.data.update(mapping)

    # ------------------------------------------------------------------
    @property
    def token(self) -> str:
        """天地图 tk 密钥 (环境变量优先)."""
        env = os.environ.get("TIANDITU_TK", "").strip()
        if env:
            return env
        return str(self.data.get("tianditu_token", "") or "").strip()

    @token.setter
    def token(self, value: str) -> None:
        self.data["tianditu_token"] = (value or "").strip()

    @property
    def has_token(self) -> bool:
        """是否已提供密钥 (宽松判断, 供界面决定是否回退到免密钥底图)."""
        return len(self.token) >= 8

    def token_format_ok(self) -> bool:
        """密钥格式是否合规 (天地图 tk 为 32 位十六进制)."""
        return is_valid_token_format(self.token)
