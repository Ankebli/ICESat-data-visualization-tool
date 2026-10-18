# -*- coding: utf-8 -*-
"""两点连线廊道 (corridor) 的几何与剖面统计.

"任点击两点给出该直线下的 ICESat 光子剖切面" 的核心数学:
    1. 以起点 p0 为中心做方位等距投影, 把每个光子投影到 p0->p1 的直线坐标系,
       得到沿迹距离 ``along`` (米) 与横向偏距 ``cross`` (米);
    2. 保留 ``0 <= along <= L`` 且 ``|cross| <= 半宽`` 的光子 —— 即"直线下方"的
       一条矩形剖切带;
    3. 以 along 为横轴、高程为纵轴绘制剖面, 并可做面元统计 (中值/分位数) 与直方图.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .geo import GEOD, corridor_polygon_lonlat, lonlat_to_mercator, project_on_line

__all__ = ["CorridorGeometry", "binned_statistic", "along_elev_histogram",
           "densify_line", "elevation_statistics"]


@dataclass
class CorridorGeometry:
    """一条剖切线: 起点、终点、廊道半宽."""

    p0: Tuple[float, float]
    p1: Tuple[float, float]
    half_width_m: float = 25.0

    length_m: float = field(init=False, default=0.0)
    azimuth: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.p0 = (float(self.p0[0]), float(self.p0[1]))
        self.p1 = (float(self.p1[0]), float(self.p1[1]))
        self.half_width_m = float(self.half_width_m)
        az, dist = GEOD.inverse(self.p0[0], self.p0[1], self.p1[0], self.p1[1])
        self.azimuth = float(np.asarray(az).reshape(-1)[0])
        self.length_m = float(np.asarray(dist).reshape(-1)[0])

    # ------------------------------------------------------------------
    def project(self, lon, lat) -> Tuple[np.ndarray, np.ndarray]:
        """光子 -> (沿迹距离, 横向偏距) 米."""
        return project_on_line(self.p0, self.p1, lon, lat)

    def mask(self, lon, lat, half_width_m: Optional[float] = None) -> np.ndarray:
        """返回落在廊道内的布尔掩膜."""
        along, cross = self.project(lon, lat)
        hw = self.half_width_m if half_width_m is None else float(half_width_m)
        return (along >= 0.0) & (along <= self.length_m) & (np.abs(cross) <= hw)

    def polygon_lonlat(self, half_width_m: Optional[float] = None) -> np.ndarray:
        """廊道矩形 (闭合环, Nx2 经纬度)."""
        hw = self.half_width_m if half_width_m is None else float(half_width_m)
        return corridor_polygon_lonlat(self.p0, self.p1, hw)

    def polygon_mercator(self, half_width_m: Optional[float] = None) -> np.ndarray:
        ring = self.polygon_lonlat(half_width_m)
        x, y = lonlat_to_mercator(ring[:, 0], ring[:, 1])
        return np.column_stack([x, y])

    def centerline_mercator(self, n: int = 2) -> np.ndarray:
        lon, lat = densify_line(self.p0, self.p1, n)
        x, y = lonlat_to_mercator(lon, lat)
        return np.column_stack([x, y])

    def describe(self) -> str:
        return (f"起点 ({self.p0[0]:.5f}, {self.p0[1]:.5f}) → "
                f"终点 ({self.p1[0]:.5f}, {self.p1[1]:.5f})\n"
                f"长度 {self.length_m:,.1f} m    方位角 {self.azimuth:.2f}°    "
                f"廊道半宽 ±{self.half_width_m:.1f} m")


def densify_line(p0, p1, n: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """在 p0-p1 之间等距取点 (用于绘制剖切线)."""
    n = max(int(n), 2)
    lon = np.linspace(float(p0[0]), float(p1[0]), n)
    lat = np.linspace(float(p0[1]), float(p1[1]), n)
    return lon, lat


# ----------------------------------------------------------------------------
# 剖面统计
# ----------------------------------------------------------------------------
def binned_statistic(along: np.ndarray, values: np.ndarray, bin_size_m: float = 20.0,
                     statistic: str = "median", length_m: Optional[float] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿迹分箱统计.

    返回 ``(bin 中心, 统计值, 每箱点数)``; 空箱为 NaN / 0.
    """
    along = np.asarray(along, "f8")
    values = np.asarray(values, "f8")
    if along.size == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)
    bin_size_m = max(float(bin_size_m), 0.1)
    lo = 0.0 if length_m is None else 0.0
    hi = float(length_m) if length_m is not None else float(np.nanmax(along))
    if hi <= lo:
        hi = lo + bin_size_m
    nbin = max(int(np.ceil((hi - lo) / bin_size_m)), 1)
    idx = np.clip(((along - lo) / bin_size_m).astype(np.int64), 0, nbin - 1)
    valid = np.isfinite(values) & (along >= lo) & (along <= hi)

    centers = lo + (np.arange(nbin) + 0.5) * bin_size_m
    out = np.full(nbin, np.nan)
    counts = np.zeros(nbin, dtype=int)

    idx_v = idx[valid]
    val_v = values[valid]
    counts = np.bincount(idx_v, minlength=nbin)[:nbin]
    order = np.argsort(idx_v, kind="stable")
    idx_s = idx_v[order]
    val_s = val_v[order]
    bounds = np.searchsorted(idx_s, np.arange(nbin + 1))
    for b in range(nbin):
        s, e = bounds[b], bounds[b + 1]
        if e <= s:
            continue
        seg = val_s[s:e]
        if statistic == "mean":
            out[b] = float(np.mean(seg))
        elif statistic == "min":
            out[b] = float(np.min(seg))
        elif statistic == "max":
            out[b] = float(np.max(seg))
        elif statistic == "p10":
            out[b] = float(np.percentile(seg, 10))
        elif statistic == "p90":
            out[b] = float(np.percentile(seg, 90))
        else:  # median (默认, 抗噪)
            out[b] = float(np.median(seg))
    return centers, out, counts


def along_elev_histogram(along: np.ndarray, elev: np.ndarray,
                         along_bins: int = 200, elev_bins: int = 200):
    """沿迹-高程二维直方图 (剖面密度图备用)."""
    along = np.asarray(along, "f8")
    elev = np.asarray(elev, "f8")
    ok = np.isfinite(along) & np.isfinite(elev)
    if not np.any(ok):
        return np.zeros((elev_bins, along_bins)), np.array([0, 1.0]), np.array([0, 1.0])
    return np.histogram2d(along[ok], elev[ok], bins=(along_bins, elev_bins))


def elevation_statistics(elev: np.ndarray) -> Dict[str, float]:
    """高程基本统计量."""
    arr = np.asarray(elev, "f8")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: float("nan") for k in
                ("count", "min", "max", "mean", "median", "p10", "p90", "std", "range")}
    return {
        "count": float(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "std": float(arr.std()),
        "range": float(arr.max() - arr.min()),
    }
