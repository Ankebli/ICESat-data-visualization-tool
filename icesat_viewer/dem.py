# -*- coding: utf-8 -*-
"""DEM 数据源与剖面对比.

支持三类来源:

1. **本地文件**: GeoTIFF / IMG / VRT / ASCII Grid 等, 由 ``rasterio`` 读取
   (缺失 rasterio 时给出明确提示);
2. **AWS Terrain Tiles (terrarium)**: ``s3.amazonaws.com/elevation-tiles-prod``,
   **免密钥**, 无需注册; 高程编码 ``(R*256 + G + B/256) - 32768``;
3. **OpenTopography Global DEM API**: 需要免费 API key, 直接下载走廊范围的 GeoTIFF
   (SRTMGL1 / SRTMGL3 / AW3D30 / COP30 等)。

垂直基准
--------
ATL03 的 ``h_ph`` 是**椭球高 (WGS84)**; SRTM 类 DEM 一般是**正交高 (EGM96)**。
两者差一个大地水准面差距 N (ATL03 里就有 ``geophys_corr/geoid``), 太湖实测
N = +7.13 m, 所以在湖面上"SRAH 读 0 m、ICESat 读 8.38 m" 是正常的,
程序会按当前"高程基准"自动换算, 并把整体高差单独报出来, 便于区分
"基准/偏移问题" 与 "实际地形差异"。

注意: SRTM 在水面上没有水面高程 (填充为 0 或空值), 因此湖面/河面区域
DEM 曲线会贴着 0 —— 这不是 bug, 程序中会给出提示。
"""

from __future__ import annotations

import io
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from . import geo

DEM_DATUMS = ("正交高 (EGM96 等)", "椭球高 (WGS84)")
TDT_DEM_CACHE = Path.home() / ".icesat_tdt_viewer" / "dem"
OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ICESatViewer/1.0"


class DemError(RuntimeError):
    """DEM 读取/下载失败."""


# ----------------------------------------------------------------------------
# 基类
# ----------------------------------------------------------------------------
class BaseDem:
    """DEM 数据源基类."""

    kind = "dem"

    def __init__(self, name: str, datum: str = DEM_DATUMS[0], note: str = ""):
        self.name = name
        self.datum = datum if datum in DEM_DATUMS else DEM_DATUMS[0]
        self.note = note
        self.bounds: Optional[Tuple[float, float, float, float]] = None  # 经度/纬度
        self.resolution_m: Optional[float] = None

    # --- 子类实现 ---------------------------------------------------
    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # --- 公共 -------------------------------------------------------
    @property
    def is_orthometric(self) -> bool:
        return self.datum == DEM_DATUMS[0]

    def describe(self) -> str:
        parts = [self.name]
        if self.bounds:
            b = self.bounds
            parts.append(f"范围 {b[0]:.4f}~{b[1]:.4f}°E, {b[2]:.4f}~{b[3]:.4f}°N")
        if self.resolution_m:
            parts.append(f"分辨率 ~{self.resolution_m:.0f} m")
        parts.append(f"基准: {self.datum}")
        if self.note:
            parts.append(self.note)
        return " · ".join(parts)

    def covers(self, lon_min, lon_max, lat_min, lat_max) -> bool:
        if not self.bounds:
            return True
        b = self.bounds
        return not (lon_max < b[0] or lon_min > b[1] or lat_max < b[2] or lat_min > b[3])


# ----------------------------------------------------------------------------
# 1) 本地栅格文件
# ----------------------------------------------------------------------------
class RasterDem(BaseDem):
    """本地 DEM 文件 (GeoTIFF 等), 由 rasterio 读取; 采样为双线性."""

    kind = "file"

    def __init__(self, path: str | Path, datum: str = DEM_DATUMS[0],
                 bounds_lonlat: Optional[Sequence[float]] = None):
        self.path = Path(path)
        super().__init__(self.path.name, datum=datum)
        try:
            import rasterio                                    # noqa: F401
            from rasterio.windows import from_bounds as _window_from_bounds  # noqa: F401
        except Exception as exc:   # pragma: no cover
            raise DemError(f"读取 DEM 文件需要 rasterio: {exc}") from exc
        self._read(bounds_lonlat)

    def _read(self, bounds_lonlat: Optional[Sequence[float]]) -> None:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(self.path) as src:
            if src.crs is None:
                raise DemError(f"{self.path.name}: 文件没有坐标系定义, 无法定位")
            self.crs = src.crs
            name_bounds = bounds_lonlat or transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            lons = [name_bounds[0], name_bounds[2]]
            lats = [name_bounds[1], name_bounds[3]]
            lon_min, lon_max = min(lons), max(lons)
            lat_min, lat_max = min(lats), max(lats)
            self.bounds = (lon_min, lon_max, lat_min, lat_max)

            window = None
            if bounds_lonlat is not None:
                # 只读走廊附近窗口, 大文件也能秒开
                west, south, east, north = transform_bounds(
                    "EPSG:4326", src.crs, bounds_lonlat[0], bounds_lonlat[2],
                    bounds_lonlat[1], bounds_lonlat[3])
                try:
                    from rasterio.windows import from_bounds

                    window = from_bounds(west, south, east, north, src.transform)
                    window = window.round_offsets().round_lengths()
                except Exception:
                    window = None
            if window is not None:
                data = src.read(1, window=window, masked=True)
                self._transform = src.window_transform(window)
            else:
                data = src.read(1, masked=True)
                self._transform = src.transform
            self._arr = data.filled(np.nan).astype("f8")
            if self._arr.size == 0:
                raise DemError(f"{self.path.name}: 窗口内没有数据")
            px = abs(self._transform.a)
            py = abs(self._transform.e)
            # 估算地面分辨率 (米)
            lat_mid = (lat_min + lat_max) / 2.0
            if str(self.crs).upper().endswith("4326") or "4326" in str(self.crs):
                self.resolution_m = float(max(px * 111320.0 * math.cos(math.radians(lat_mid)),
                                              py * 110540.0))
            else:
                self.resolution_m = float(max(px, py))

    # ------------------------------------------------------------------
    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        lon = np.atleast_1d(np.asarray(lon, "f8"))
        lat = np.atleast_1d(np.asarray(lat, "f8"))
        if lon.size == 0:
            return np.array([], "f8")
        from rasterio.warp import transform as warp_transform

        xs, ys = warp_transform("EPSG:4326", self.crs, lon.tolist(), lat.tolist())
        xs = np.asarray(xs, "f8")
        ys = np.asarray(ys, "f8")
        inv = ~self._transform
        cols, rows = inv * (xs, ys)
        cols = np.asarray(cols, "f8")
        rows = np.asarray(rows, "f8")
        return _bilinear(self._arr, cols, rows)


def open_dem_file(path: str | Path, datum: str = DEM_DATUMS[0],
                  bounds_lonlat: Optional[Sequence[float]] = None) -> BaseDem:
    """打开本地 DEM 文件."""
    path = Path(path)
    if not path.exists():
        raise DemError(f"文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff", ".img", ".vrt", ".asc", ".dt2", ".grd", ".hgt"}:
        return RasterDem(path, datum=datum, bounds_lonlat=bounds_lonlat)
    # 未知后缀也试着按栅格打开
    return RasterDem(path, datum=datum, bounds_lonlat=bounds_lonlat)


# ----------------------------------------------------------------------------
# 2) AWS Terrain Tiles (terrarium, 免密钥)
# ----------------------------------------------------------------------------
def _terrarium_zoom(bounds: Sequence[float], max_tiles: int = 16) -> int:
    """按走廊范围选缩放级, 使瓦片数量可控 (默认 <= 4x4)."""
    lon_min, lon_max, lat_min, lat_max = bounds
    best = 5
    for z in range(5, 15):
        span_lon = 360.0 / (2 ** z)
        span_lat = span_lon * math.cos(math.radians((lat_min + lat_max) / 2.0))
        nx = max(1, int(math.ceil((lon_max - lon_min) / span_lon)) + 1)
        ny = max(1, int(math.ceil((lat_max - lat_min) / max(span_lat, 1e-6))) + 1)
        if nx * ny > max_tiles:
            break
        best = z
    return best


class TerrariumDem(BaseDem):
    """AWS Terrain Tiles (terrarium) 拼接成 DEM, 沿走廊双线性采样."""

    kind = "terrarium"

    def __init__(self, bounds_lonlat: Sequence[float], zoom: Optional[int] = None,
                 cache_dir: Path = TDT_DEM_CACHE, timeout: float = 20.0,
                 progress: Optional[Callable[[float, str], None]] = None):
        lon_min, lon_max, lat_min, lat_max = bounds_lonlat
        super().__init__("AWS Terrain Tiles (SRTM/ASTER 融合)",
                         datum=DEM_DATUMS[0],
                         note="免密钥; 水面按 0 m 填充")
        self.bounds = (lon_min, lon_max, lat_min, lat_max)
        self.zoom = int(zoom or _terrarium_zoom(self.bounds))
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self._session = None
        self._lock = threading.Lock()
        self._build(progress)

    # ------------------------------------------------------------------
    @property
    def _requests(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _tile_path(self, z: int, x: int, y: int) -> Path:
        return self.cache_dir / f"z{z}" / str(x) / f"{y}.png"

    def _load_tile(self, z: int, x: int, y: int) -> Optional[np.ndarray]:
        path = self._tile_path(z, x, y)
        if path.exists():
            try:
                from PIL import Image

                with Image.open(path) as im:
                    return _decode_terrarium(np.asarray(im.convert("RGB")))
            except Exception:
                try:
                    path.unlink()
                except Exception:
                    pass
        try:
            import requests
            from PIL import Image

            resp = self._requests.get(TERRARIUM_URL.format(z=z, x=x, y=y),
                                      timeout=self.timeout,
                                      headers={"User-Agent": USER_AGENT})
            if resp.status_code != 200:
                return None
            with Image.open(io.BytesIO(resp.content)) as im:
                im = im.convert("RGB")
                arr = np.asarray(im)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with Image.fromarray(arr) as out:
                    out.save(path)
            except Exception:
                pass
            return _decode_terrarium(arr)
        except Exception:
            return None

    def _build(self, progress=None) -> None:
        z = self.zoom
        n = 2 ** z
        span = 360.0 / n
        tx0 = int(math.floor((self.bounds[0] + 180.0) / span))
        tx1 = int(math.floor((self.bounds[1] + 180.0) / span))
        lat_lo, lat_hi = self.bounds[2], self.bounds[3]
        ty0 = int(_lat_to_tiley(lat_hi, z))
        ty1 = int(_lat_to_tiley(lat_lo, z))
        tx0, tx1 = max(tx0, 0), min(tx1, n - 1)
        ty0, ty1 = max(ty0, 0), min(ty1, n - 1)
        cols, rows = tx1 - tx0 + 1, ty1 - ty0 + 1
        if cols <= 0 or rows <= 0:
            raise DemError("走廊范围无效")
        mosaic = np.full((rows * 256, cols * 256), np.nan)
        jobs = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
        got = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(self._load_tile, z, tx, ty): (tx, ty) for tx, ty in jobs}
            for fut, (tx, ty) in futures.items():
                arr = fut.result()
                if arr is None:
                    continue
                r = (ty - ty0) * 256
                c = (tx - tx0) * 256
                mosaic[r:r + 256, c:c + 256] = arr
                got += 1
                if progress:
                    progress(got / max(len(jobs), 1),
                             f"下载 DEM 瓦片 {got}/{len(jobs)}")
        if got == 0:
            raise DemError("DEM 瓦片下载失败 (网络不可用?)")
        self._mosaic = mosaic
        self._z = z
        self._tx0, self._ty0 = tx0, ty0
        # 像素分辨率 (米)
        span_m = 2 * geo.ORIGIN_SHIFT / (256.0 * n)
        self.resolution_m = float(span_m * math.cos(math.radians((lat_lo + lat_hi) / 2.0)))

    # ------------------------------------------------------------------
    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        lon = np.atleast_1d(np.asarray(lon, "f8"))
        lat = np.atleast_1d(np.asarray(lat, "f8"))
        if lon.size == 0:
            return np.array([], "f8")
        n = 2 ** self._z
        xf = (lon + 180.0) / 360.0 * n
        yf = _lat_to_tiley(lat, self._z)
        col = (xf - self._tx0) * 256.0
        row = (yf - self._ty0) * 256.0
        return _bilinear(self._mosaic, col, row)


# ----------------------------------------------------------------------------
# 3) OpenTopography API (需要免费 key)
# ----------------------------------------------------------------------------
class OpenTopographyDem(RasterDem):
    """通过 OpenTopography API 下载走廊范围 DEM 后按文件采样."""

    kind = "opentopography"

    def __init__(self, bounds_lonlat: Sequence[float], api_key: str,
                 demtype: str = "SRTMGL1", datum: str = DEM_DATUMS[0],
                 cache_dir: Path = TDT_DEM_CACHE, timeout: float = 120.0,
                 progress: Optional[Callable[[float, str], None]] = None):
        key = (api_key or "").strip()
        if not key:
            raise DemError("OpenTopography 需要 API key (免费注册: "
                           "https://portal.opentopography.org/myopentopo )")
        lon_min, lon_max, lat_min, lat_max = bounds_lonlat
        if progress:
            progress(0.1, f"向 OpenTopography 请求 {demtype} …")
        import requests

        params = {
            "demtype": demtype,
            "south": f"{lat_min:.6f}", "north": f"{lat_max:.6f}",
            "west": f"{lon_min:.6f}", "east": f"{lon_max:.6f}",
            "outputFormat": "GTiff", "API_Key": key,
        }
        try:
            resp = requests.get(OPENTOPO_URL, params=params, timeout=timeout,
                                headers={"User-Agent": USER_AGENT})
        except Exception as exc:
            raise DemError(f"OpenTopography 请求失败: {exc}") from exc
        ctype = resp.headers.get("Content-Type", "")
        if resp.status_code != 200 or "image" not in ctype:
            text = ""
            try:
                text = resp.text[:200]
            except Exception:
                pass
            raise DemError(f"OpenTopography 返回错误 (HTTP {resp.status_code}): {text}")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        out = cache_dir / (f"opentopo_{demtype}_{lat_min:.4f}_{lon_min:.4f}_"
                           f"{lat_max:.4f}_{lon_max:.4f}.tif")
        out.write_bytes(resp.content)
        if progress:
            progress(0.8, "解析 DEM …")
        super().__init__(out, datum=datum, bounds_lonlat=bounds_lonlat)
        self.name = f"OpenTopography {demtype}"
        self.note = "在线 API 下载"


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def _decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    """terrarium 编码 -> 高程 (米)."""
    arr = np.asarray(rgb, dtype="f8")
    return arr[..., 0] * 256.0 + arr[..., 1] + arr[..., 2] / 256.0 - 32768.0


def _lat_to_tiley(lat, z: int):
    """纬度 -> 瓦片浮点行号 (Web Mercator)."""
    lat = np.clip(np.asarray(lat, dtype="f8"), -85.05112878, 85.05112878)
    n = 2.0 ** z
    rad = np.radians(lat)
    return (1.0 - np.log(np.tan(rad) + 1.0 / np.cos(rad)) / math.pi) / 2.0 * n


def _bilinear(arr: np.ndarray, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """在数组上做双线性采样; 越界返回 NaN."""
    out = np.full(cols.shape, np.nan, dtype="f8")
    h, w = arr.shape
    ok = (np.isfinite(cols) & np.isfinite(rows)
          & (cols >= 0) & (rows >= 0) & (cols <= w - 1) & (rows <= h - 1))
    if not np.any(ok):
        return out
    c = cols[ok]
    r = rows[ok]
    c0 = np.floor(c).astype(np.int64)
    r0 = np.floor(r).astype(np.int64)
    c1 = np.minimum(c0 + 1, w - 1)
    r1 = np.minimum(r0 + 1, h - 1)
    dc = c - c0
    dr = r - r0
    vals = (arr[r0, c0] * (1 - dc) * (1 - dr) + arr[r0, c1] * dc * (1 - dr)
            + arr[r1, c0] * (1 - dc) * dr + arr[r1, c1] * dc * dr)
    out[ok] = vals
    return out


# ----------------------------------------------------------------------------
# 沿剖切线采样 + 对比统计
# ----------------------------------------------------------------------------
def corridor_bounds(corridor, half_width_m: float = 50.0,
                    margin_m: float = 200.0) -> Tuple[float, float, float, float]:
    """剖切线 (含缓冲) 的经纬度范围 -> (lon_min, lon_max, lat_min, lat_max)."""
    ring = corridor.polygon_lonlat(max(half_width_m, 1.0))
    lon_min, lon_max = float(ring[:, 0].min()), float(ring[:, 0].max())
    lat_min, lat_max = float(ring[:, 1].min()), float(ring[:, 1].max())
    dlon = margin_m / (111320.0 * max(math.cos(math.radians((lat_min + lat_max) / 2.0)), 0.1))
    dlat = margin_m / 110540.0
    return (lon_min - dlon, lon_max + dlon, lat_min - dlat, lat_max + dlat)


def sample_along_corridor(dem: BaseDem, corridor, step_m: float = 2.0,
                          max_points: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """沿剖切线中心线等间距采样 DEM, 返回 ``(along_m, z_dem)``."""
    length = float(corridor.length_m)
    if length <= 0:
        return np.array([]), np.array([])
    step = max(float(step_m), 0.1)
    n = int(min(max(math.ceil(length / step) + 1, 2), max_points))
    along = np.linspace(0.0, length, n)
    lon, lat = geo.GEOD.forward(np.full(n, corridor.p0[0]), np.full(n, corridor.p0[1]),
                                np.full(n, corridor.azimuth), along)
    z = dem.sample(np.asarray(lon, "f8"), np.asarray(lat, "f8"))
    return along, z


def adjust_dem_to_elev_kind(along_dem: np.ndarray, z_dem: np.ndarray,
                            selection, elev_kind: str,
                            dem_is_orthometric: bool, offset_m: float = 0.0
                            ) -> Tuple[np.ndarray, str]:
    """把 DEM 高程换算到与剖面相同的垂直基准.

    返回 ``(z_adjusted, 说明)``。规则:

    * 剖面"椭球高" + DEM 正交高: ``h = H + N``, N 由 ATL03 geoid 沿迹插值;
    * 剖面"正交高" + DEM 正交高: 直接用;
    * 剖面"椭球高" + DEM 椭球高: 直接用;
    * 剖面"正交高" + DEM 椭球高: ``H = h - N``;
    * 相对高: 各自减去自身中值 (只比较地形起伏, 不看绝对高程);
    * 最后统一加 ``offset_m`` (手调).
    """
    z = np.asarray(z_dem, dtype="f8").copy()
    if z.size == 0:
        return z, ""
    geoid_along = _geoid_along(along_dem, selection)
    note_bits = []
    rel = elev_kind.startswith("相对")
    want_ortho = elev_kind.startswith("正交")

    if rel:
        finite = np.isfinite(z)
        if np.any(finite):
            z = z - float(np.nanmedian(z[finite]))
        if offset_m:
            z = z + float(offset_m)
        return z, "相对高: DEM 减自身中值"

    if dem_is_orthometric and not want_ortho:
        if geoid_along is not None:
            z = z + geoid_along
            note_bits.append(f"兼容椭球高: DEM + geoid (N≈{float(np.nanmedian(geoid_along)):.2f} m)")
        else:
            note_bits.append("缺少 geoid, 未做椭球高换算")
    elif (not dem_is_orthometric) and want_ortho:
        if geoid_along is not None:
            z = z - geoid_along
            note_bits.append(f"转正交高: DEM - geoid (N≈{float(np.nanmedian(geoid_along)):.2f} m)")
        else:
            note_bits.append("缺少 geoid, 未做正交高换算")

    if offset_m:
        z = z + float(offset_m)
        note_bits.append(f"手调偏移 {offset_m:+.2f} m")
    return z, "; ".join(note_bits)


def _geoid_along(along_dem: np.ndarray, selection) -> Optional[np.ndarray]:
    """把 ATL03 的 geoid 沿迹插值到 DEM 采样位置 (默认 0)."""
    if selection is None or getattr(selection, "geoid", None) is None:
        return None
    try:
        a = np.asarray(selection.along, "f8")
        g = np.asarray(selection.geoid, "f8")
        ok = np.isfinite(a) & np.isfinite(g)
        if not np.any(ok):
            return None
        a, g = a[ok], g[ok]
        order = np.argsort(a)
        a, g = a[order], g[order]
        uniq, idx = np.unique(a, return_index=True)
        g = g[idx]
        if uniq.size == 1:
            return np.full(np.asarray(along_dem).shape, float(g[0]))
        return np.interp(np.asarray(along_dem, "f8"), uniq, g)
    except Exception:
        return None


@dataclass
class DemComparison:
    """DEM 与 ICESat 剖面的一致性统计."""

    n_points: int = 0
    mean_m: float = float("nan")
    std_m: float = float("nan")
    rmse_m: float = float("nan")
    median_m: float = float("nan")
    coverage: float = float("nan")     # DEM 有效采样比例
    note: str = ""
    dem_flat_ratio: float = float("nan")   # DEM 段为常数的比例 (识别 SRTM 水面)

    def summary(self) -> str:
        if self.n_points <= 0:
            return self.note or "无有效对比点"
        return (f"DEM 对比: Δ {self.mean_m:+.2f} ± {self.std_m:.2f} m "
                f"(RMSE {self.rmse_m:.2f} m, n={self.n_points:,})")


def compare_with_profile(along_icesat: np.ndarray, elev_icesat: np.ndarray,
                         along_dem: np.ndarray, z_dem: np.ndarray,
                         bin_size_m: float = 20.0) -> DemComparison:
    """ICESat 面元中值线与 DEM 曲线的高差统计."""
    along_icesat = np.asarray(along_icesat, "f8")
    elev_icesat = np.asarray(elev_icesat, "f8")
    along_dem = np.asarray(along_dem, "f8")
    z_dem = np.asarray(z_dem, "f8")
    ok = np.isfinite(along_icesat) & np.isfinite(elev_icesat)
    along_icesat, elev_icesat = along_icesat[ok], elev_icesat[ok]
    ok2 = np.isfinite(along_dem) & np.isfinite(z_dem)
    if along_dem.size == 0 or not np.any(ok2):
        return DemComparison(note="DEM 在该范围无有效值")
    coverage = float(np.count_nonzero(ok2)) / float(max(along_dem.size, 1))
    if along_icesat.size < 5:
        return DemComparison(coverage=coverage, note="ICESat 光子太少, 无法对比")

    # ICESat 面元中值
    from .corridor import binned_statistic

    length = float(max(along_icesat.max(), 1.0))
    centers, med, counts = binned_statistic(along_icesat, elev_icesat,
                                            max(float(bin_size_m), 1.0),
                                            statistic="median", length_m=length)
    good = np.isfinite(med) & (counts >= 3)
    if np.count_nonzero(good) < 3:
        return DemComparison(coverage=coverage, note="ICESat 分箱有效值太少")
    dem_at_bins = np.interp(centers[good], along_dem[ok2], z_dem[ok2])
    diff = dem_at_bins - med[good]
    finite = np.isfinite(diff)
    diff = diff[finite]
    if diff.size == 0:
        return DemComparison(coverage=coverage, note="重合段无有效对比")
    # DEM 在该段是否几乎是常数 (SRTM 水面填 0 的典型特征)
    z_seg = z_dem[ok2]
    flat_ratio = float(np.mean(np.abs(np.diff(z_seg)) < 1e-6)) if z_seg.size > 1 else float("nan")
    cmp = DemComparison(
        n_points=int(diff.size),
        mean_m=float(np.mean(diff)),
        std_m=float(np.std(diff)),
        median_m=float(np.median(diff)),
        rmse_m=float(np.sqrt(np.mean(diff ** 2))),
        coverage=coverage,
        dem_flat_ratio=flat_ratio)
    if flat_ratio > 0.9:
        cmp.note = "DEM 在该剖面几乎是常数 (SRTM 类产品对水面按 0 m 填充), 对比结果仅供参考"
    return cmp
