# -*- coding: utf-8 -*-
"""经纬度 / Web Mercator 投影 / 大地线计算工具.

设计约定
--------
* 地图画布内部统一使用 **Web Mercator (EPSG:3857) 米制坐标**, 与瓦片底图一致,
  这样底图 ``imshow`` 与矢量图层可以严格套合; 经纬度只在交互与数据层出现.
* 函数尽量向量化 (支持 numpy 数组), 标量输入返回标量.
* 大地线计算优先使用 ``pyproj.Geod`` (WGS84 椭球), 若环境无 pyproj 则退化为球面近似.
"""

from __future__ import annotations

import math

import numpy as np

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
R_EARTH = 6378137.0                       # WGS84 长半轴, Web Mercator 采用球体近似
ORIGIN_SHIFT = math.pi * R_EARTH          # 20037508.342789244 (半周长)
MAX_LAT = 85.05112877980659               # Web Mercator 最大纬度
TILE_SIZE = 256                           # 瓦片边长 (像素)

__all__ = [
    "R_EARTH", "ORIGIN_SHIFT", "MAX_LAT", "TILE_SIZE",
    "lonlat_to_mercator", "mercator_to_lonlat",
    "resolution", "zoom_for_resolution", "tile_span",
    "tile_index_range", "tile_bounds_mercator", "wrap_tile_x",
    "Geodesic", "GEOD", "project_on_line", "corridor_polygon_lonlat",
]


def _scalarize(value, scalar: bool):
    """若输入是标量则以 float 返回, 否则原样返回数组."""
    if scalar:
        return float(np.asarray(value).reshape(-1)[0]) if np.asarray(value).size else float("nan")
    return value


# ----------------------------------------------------------------------------
# Web Mercator 投影
# ----------------------------------------------------------------------------
def lonlat_to_mercator(lon, lat):
    """经纬度(度) -> Web Mercator 米."""
    lon_arr = np.asarray(lon, dtype="f8")
    lat_arr = np.asarray(lat, dtype="f8")
    scalar = lon_arr.ndim == 0 and lat_arr.ndim == 0
    lat_clip = np.clip(lat_arr, -MAX_LAT, MAX_LAT)
    x = np.radians(lon_arr) * R_EARTH
    y = np.log(np.tan(np.pi / 4.0 + np.radians(lat_clip) / 2.0)) * R_EARTH
    return _scalarize(x, scalar), _scalarize(y, scalar)


def mercator_to_lonlat(x, y):
    """Web Mercator 米 -> 经纬度(度)."""
    x_arr = np.asarray(x, dtype="f8")
    y_arr = np.asarray(y, dtype="f8")
    scalar = x_arr.ndim == 0 and y_arr.ndim == 0
    lon = np.degrees(x_arr / R_EARTH)
    lat = np.degrees(2.0 * np.arctan(np.exp(y_arr / R_EARTH)) - np.pi / 2.0)
    return _scalarize(lon, scalar), _scalarize(lat, scalar)


def resolution(z: float) -> float:
    """第 z 级瓦片的地面分辨率 (米/像素)."""
    return 2.0 * ORIGIN_SHIFT / (TILE_SIZE * (2.0 ** float(z)))


def tile_span(z: float) -> float:
    """第 z 级单张瓦片在 Mercator 平面上的边长 (米)."""
    return 2.0 * ORIGIN_SHIFT / (2.0 ** float(z))


def zoom_for_resolution(res_m_per_px: float, min_zoom: int = 1, max_zoom: int = 19) -> int:
    """根据所需分辨率选择瓦片层级 (取不小于需求的最小整数级)."""
    if not np.isfinite(res_m_per_px) or res_m_per_px <= 0:
        return min_zoom
    z = math.log2(2.0 * ORIGIN_SHIFT / (TILE_SIZE * res_m_per_px))
    return int(max(min_zoom, min(max_zoom, math.ceil(z - 1e-9))))


def tile_index_range(x_min, x_max, y_min, y_max, z: int):
    """Mercator 范围 -> 需覆盖的瓦片索引范围 (含端点, y 轴向下)."""
    span = tile_span(z)
    n = 2 ** int(z)
    tx0 = int(math.floor((x_min + ORIGIN_SHIFT) / span))
    tx1 = int(math.floor((x_max + ORIGIN_SHIFT) / span))
    ty0 = int(math.floor((ORIGIN_SHIFT - y_max) / span))
    ty1 = int(math.floor((ORIGIN_SHIFT - y_min) / span))
    # 允许越界, 由 wrap_tile_x / 调用方处理世界重复
    return tx0, min(tx1, tx0 + n * 2), max(ty0, 0), min(ty1, n - 1)


def tile_bounds_mercator(tx: int, ty: int, z: int):
    """瓦片索引 -> Mercator 边界 (x_min, x_max, y_min, y_max)."""
    span = tile_span(z)
    x_min = tx * span - ORIGIN_SHIFT
    x_max = x_min + span
    y_max = ORIGIN_SHIFT - ty * span
    y_min = y_max - span
    return x_min, x_max, y_min, y_max


def wrap_tile_x(tx: int, z: int) -> int:
    """经度方向循环 (世界重复), 便于跨 180° 拖动."""
    n = 2 ** int(z)
    return int(tx) % n


# ----------------------------------------------------------------------------
# 大地线计算
# ----------------------------------------------------------------------------
class Geodesic:
    """WGS84 椭球大地线计算; 无 pyproj 时使用球面近似."""

    def __init__(self):
        self._geod = None
        try:  # pyproj 在 waterenv 中可用, 精度更高
            from pyproj import Geod  # type: ignore

            self._geod = Geod(ellps="WGS84")
        except Exception:
            self._geod = None

    @property
    def backend(self) -> str:
        return "pyproj(WGS84)" if self._geod is not None else "spherical"

    def inverse(self, lon1, lat1, lon2, lat2):
        """反算: 返回 (起点方位角(度, 正北顺时针), 距离(米))."""
        if self._geod is not None:
            az, _, dist = self._geod.inv(lon1, lat1, lon2, lat2)
            return az, dist
        return _sphere_inverse(lon1, lat1, lon2, lat2)

    def forward(self, lon, lat, azimuth, distance):
        """正算: 由起点、方位角(度)、距离(米) 推算终点经纬度."""
        if self._geod is not None:
            lon2, lat2, _ = self._geod.fwd(lon, lat, azimuth, distance)
            return lon2, lat2
        return _sphere_forward(lon, lat, azimuth, distance)

    def distance(self, lon1, lat1, lon2, lat2) -> float:
        _, dist = self.inverse(lon1, lat1, lon2, lat2)
        return float(np.asarray(dist).reshape(-1)[0])


def _sphere_inverse(lon1, lat1, lon2, lat2):
    """球面反算 (Haversine + 初始方位角), 支持数组."""
    lon1a, lat1a = np.radians(np.asarray(lon1, "f8")), np.radians(np.asarray(lat1, "f8"))
    lon2a, lat2a = np.radians(np.asarray(lon2, "f8")), np.radians(np.asarray(lat2, "f8"))
    dlon = lon2a - lon1a
    cos_lat2 = np.cos(lat2a)
    y = np.sin(dlon) * cos_lat2
    x = np.cos(lat1a) * np.sin(lat2a) - np.sin(lat1a) * cos_lat2 * np.cos(dlon)
    az = np.degrees(np.arctan2(y, x))
    a = np.sin((lat2a - lat1a) / 2.0) ** 2 + np.cos(lat1a) * cos_lat2 * np.sin(dlon / 2.0) ** 2
    dist = 2.0 * R_EARTH * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return az, dist


def _sphere_forward(lon, lat, azimuth, distance):
    """球面正算, 支持数组."""
    lon_a, lat_a = np.radians(np.asarray(lon, "f8")), np.radians(np.asarray(lat, "f8"))
    az = np.radians(np.asarray(azimuth, "f8"))
    d = np.asarray(distance, "f8") / R_EARTH
    lat2 = np.arcsin(np.sin(lat_a) * np.cos(d) + np.cos(lat_a) * np.sin(d) * np.cos(az))
    lon2 = lon_a + np.arctan2(np.sin(az) * np.sin(d) * np.cos(lat_a),
                              np.cos(d) - np.sin(lat_a) * np.sin(lat2))
    return np.degrees(lon2), np.degrees(lat2)


GEOD = Geodesic()


def project_on_line(p0, p1, lon, lat):
    """把 (lon, lat) 投影到 p0->p1 的直线坐标系.

    实现方式等价于以 p0 为中心的方位等距投影 (Azimuthal Equidistant):
    返回 ``(along, cross)`` —— 沿迹距离 (米, 从 p0 起算, p1 处等于线长)
    与横向偏距 (米, 右正左负). 数百公里以内精度优于 0.1%.

    注: 基准线取 p0->p1 的 **大地线 (测地线)**, 与地图上按大地线加密绘制的剖切线
    严格一致. 它与"墨卡托图面上的直线"在 10 km 尺度上相差约 4 m (31°N),
    远小于常用廊道宽度, 不影响剖切结果.
    """
    lon0, lat0 = float(p0[0]), float(p0[1])
    lon1, lat1 = float(p1[0]), float(p1[1])
    az0, _ = GEOD.inverse(lon0, lat0, lon1, lat1)
    lon_arr = np.asarray(lon, "f8")
    lat_arr = np.asarray(lat, "f8")
    az, dist = GEOD.inverse(np.full(lon_arr.shape, lon0, "f8"),
                            np.full(lat_arr.shape, lat0, "f8"),
                            lon_arr, lat_arr)
    daz = np.radians(np.asarray(az, "f8") - float(np.asarray(az0).reshape(-1)[0]))
    d = np.asarray(dist, "f8")
    return d * np.cos(daz), d * np.sin(daz)


def corridor_polygon_lonlat(p0, p1, half_width_m: float):
    """廊道矩形四角经纬度 (闭合环, 5x2), 用于地图上绘制剖切范围."""
    lon0, lat0 = float(p0[0]), float(p0[1])
    lon1, lat1 = float(p1[0]), float(p1[1])
    az, _ = GEOD.inverse(lon0, lat0, lon1, lat1)
    az = float(np.asarray(az).reshape(-1)[0])
    left0 = GEOD.forward(lon0, lat0, az - 90.0, half_width_m)
    right0 = GEOD.forward(lon0, lat0, az + 90.0, half_width_m)
    left1 = GEOD.forward(lon1, lat1, az - 90.0, half_width_m)
    right1 = GEOD.forward(lon1, lat1, az + 90.0, half_width_m)
    ring = np.array([
        [left0[0], left0[1]],
        [left1[0], left1[1]],
        [right1[0], right1[1]],
        [right0[0], right0[1]],
        [left0[0], left0[1]],
    ], dtype="f8")
    return ring
