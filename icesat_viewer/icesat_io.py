# -*- coding: utf-8 -*-
"""ICESat / ICESat-2 数据读取.

支持的产品与格式
----------------
======================  ==================================================
ATL03 (ICESat-2)        HDF5, **惰性读取**: 先用 geolocation 段的
                        ``reference_photon_lat/lon`` + ``ph_index_beg`` +
                        ``segment_ph_cnt`` 建立索引, 剖面计算时只读取落在
                        廊道内的那几段光子 (2 GB 的文件也能秒级响应)
ATL06 / ATL08 / ATL13   通用 HDF5 读取 (自动识别 latitude/longitude/高程路径)
GLAH14 (ICESat/GLAS)    ``/Data_40HZ/Geolocation/d_lat`` 等
CSV / TXT               自动识别分隔符与 lat/lon/elev 列名
LAS / LAZ               内置纯 Python LAS 读取 (LAZ 需 laspy)
======================  ==================================================

统一输出 :class:`PhotonSelection`: 廊道内的光子经纬度、高程、沿迹距离、横向偏距。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .corridor import CorridorGeometry
from .settings import INDEX_CACHE_DIR

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
BEAM_ORDER = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")
BEAM_NAMES = {"gt1l": "gt1l 弱", "gt1r": "gt1r 强", "gt2l": "gt2l 弱",
              "gt2r": "gt2r 强", "gt3l": "gt3l 弱", "gt3r": "gt3r 强"}
BEAM_COLORS = {"gt1l": "#1f77b4", "gt1r": "#17becf", "gt2l": "#2ca02c",
               "gt2r": "#98df8a", "gt3l": "#d62728", "gt3r": "#ff9896",
               "ATL06": "#ff7f0e", "ATL08": "#9467bd", "GLAH14": "#8c564b",
               "CSV": "#e377c2", "LAS": "#7f7f7f", "data": "#e377c2"}

#: ATL03 signal_conf_ph 的 5 个地表类型通道
CONF_CHANNELS = {"陆地": 0, "海洋": 1, "海冰": 2, "陆地冰": 3, "内陆水体": 4}
CONF_MODES = ("全部", "中等及以上", "仅高置信")
ELEV_KINDS = ("椭球高", "正交高(减geoid)", "相对高(减中值)")

MAX_READ_PHOTONS = 12_000_000   # 单次剖面最多读取的光子数 (安全阀)


# ----------------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------------
@dataclass
class ReadOptions:
    """剖面读取/筛选参数."""

    half_width_m: float = 25.0
    beams: Sequence[str] = BEAM_ORDER
    conf_mode: str = "全部"          # 见 CONF_MODES
    conf_channel: str = "自动"       # 见 CONF_CHANNELS 的键, 或 "自动"
    remove_afterpulse: bool = False  # 剔除 quality_ph != 0
    max_photons: int = 4_000_000


@dataclass
class PhotonSelection:
    """廊道内光子集合 (剖面计算的结果)."""

    lon: np.ndarray
    lat: np.ndarray
    elev: np.ndarray                  # 椭球高 (m)
    along: np.ndarray                 # 沿迹距离 (m)
    cross: np.ndarray                 # 横向偏距 (m)
    beam: np.ndarray                  # 每个光子的来源 (gt1l/... 或产品名)
    corridor: CorridorGeometry
    product: str = ""
    source: str = ""
    geoid: Optional[np.ndarray] = None
    truncated: int = 0                # 因超限被丢弃的光子数
    note: str = ""
    extras: Dict[str, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.elev.size)

    @property
    def empty(self) -> bool:
        return self.elev.size == 0

    def display_elev(self, kind: str = ELEV_KINDS[0]) -> np.ndarray:
        """按高程基准返回用于绘图的高程."""
        if kind.startswith("正交") and self.geoid is not None:
            return self.elev - self.geoid
        if kind.startswith("相对"):
            arr = np.asarray(self.elev, "f8")
            finite = arr[np.isfinite(arr)]
            if finite.size:
                return arr - float(np.median(finite))
            return arr
        return self.elev

    def elev_label(self, kind: str = ELEV_KINDS[0]) -> str:
        if kind.startswith("正交"):
            return "正交高 (m, 减 geoid)" if self.geoid is not None else "椭球高 (m)"
        if kind.startswith("相对"):
            return "相对高 (m, 减中值)"
        return "椭球高 (m, WGS84)"

    def beam_counts(self) -> Dict[str, int]:
        labels, counts = np.unique(self.beam, return_counts=True)
        return {str(k): int(v) for k, v in zip(labels, counts)}

    def present_beams(self) -> List[str]:
        return sorted(set(str(b) for b in np.unique(self.beam)))

    def describe(self) -> str:
        from .corridor import elevation_statistics

        stats = elevation_statistics(self.display_elev("椭球高"))
        lines = [
            f"数据源: {self.source} ({self.product})",
            self.corridor.describe(),
            f"廊道内光子数: {len(self):,}"
            + (f"  (另有 {self.truncated:,} 个因超限未载入)" if self.truncated else ""),
        ]
        if len(self):
            lines.append(f"高程: {stats['min']:.2f} ~ {stats['max']:.2f} m, "
                         f"中值 {stats['median']:.2f} m (跨度 {stats['range']:.2f} m)")
            bc = self.beam_counts()
            lines.append("各波束光子数: " + ", ".join(f"{k}={v:,}" for k, v in sorted(bc.items())))
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def to_csv(self, path: str | Path, elev_kind: str = ELEV_KINDS[0]) -> Path:
        """导出剖面点云为 CSV."""
        path = Path(path)
        elev = self.display_elev(elev_kind)
        header = ("lon,lat,elev_m,along_m,cross_m,beam,elev_kind,"
                  "longitude_start,latitude_start,longitude_end,latitude_end")
        p0, p1 = self.corridor.p0, self.corridor.p1
        arr = np.column_stack([
            self.lon, self.lat, elev, self.along, self.cross,
        ])
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write(header + "\n")
            fh.write(f"# 剖切线: ({p0[0]:.8f},{p0[1]:.8f}) -> ({p1[0]:.8f},{p1[1]:.8f}), "
                     f"半宽 {self.corridor.half_width_m:g} m, "
                     f"长度 {self.corridor.length_m:.2f} m\n")
            for row, beam in zip(arr, self.beam):
                fh.write(f"{row[0]:.8f},{row[1]:.8f},{row[2]:.3f},"
                         f"{row[3]:.2f},{row[4]:.2f},{beam},{elev_kind},"
                         f"{p0[0]:.8f},{p0[1]:.8f},{p1[0]:.8f},{p1[1]:.8f}\n")
        return path


# ----------------------------------------------------------------------------
# 覆盖判定 (KD 树 / 网格)
# ----------------------------------------------------------------------------
class CoverageIndex:
    """轨迹点索引: 判断经纬度是否落在 ICESat 覆盖范围内, 并支持"吸附到最近轨道点"."""

    def __init__(self, lon: np.ndarray, lat: np.ndarray,
                 labels: Optional[Sequence[str]] = None):
        lon = np.asarray(lon, "f8")
        lat = np.asarray(lat, "f8")
        ok = np.isfinite(lon) & np.isfinite(lat)
        lon, lat = lon[ok], lat[ok]
        if labels is not None and len(labels) == ok.size:
            labels = np.asarray(labels, dtype="U16")[ok]
        else:
            labels = None
        self.tree = None
        self._grid = None
        self._cell = 0.05
        self.lon = lon
        self.lat = lat
        self.labels = labels
        if lon.size == 0:
            return
        try:
            from scipy.spatial import cKDTree  # type: ignore

            lam = np.radians(lon)
            phi = np.radians(lat)
            xyz = np.column_stack([np.cos(phi) * np.cos(lam),
                                   np.cos(phi) * np.sin(lam),
                                   np.sin(phi)]) * 6371008.8
            # 点太密时抽稀, 控制建树时间 (抽稀后仍保持 ~几十米间隔, 足以吸附)
            stride = max(1, int(xyz.shape[0] // 1_500_000))
            if stride > 1:
                xyz = xyz[::stride]
                self.lon = lon[::stride]
                self.lat = lat[::stride]
                self.labels = None if labels is None else labels[::stride]
            self.tree = cKDTree(xyz)
        except Exception:
            self._grid = set(zip(np.round(lon / self._cell).astype(np.int64),
                                 np.round(lat / self._cell).astype(np.int64)))

    # ------------------------------------------------------------------
    def _ecef(self, lon: float, lat: float) -> np.ndarray:
        lam, phi = np.radians(lon), np.radians(lat)
        return np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam),
                         np.sin(phi)]) * 6371008.8

    def nearest_distance_m(self, lon: float, lat: float) -> float:
        """到最近轨迹点的近似地面距离 (米); 无数据返回 inf."""
        return self.nearest_point(lon, lat)[0]

    def nearest_point(self, lon: float, lat: float, label: Optional[str] = None,
                      k: int = 200) -> Tuple[float, float, float, Optional[str]]:
        """最近轨迹点.

        返回 ``(距离m, 经度, 纬度, 波束/标签)``; 无数据返回 ``(inf, lon, lat, None)``.
        ``label`` 非空时, 在最近的 k 个点中优先取该标签的点 (用于"两点吸附到同一波束"),
        找不到则返回 inf.
        """
        if self.tree is not None:
            pt = self._ecef(lon, lat)
            if label is None:
                dist, idx = self.tree.query(pt, k=1)
                idx = int(np.asarray(idx).reshape(-1)[0])
                lab = None if self.labels is None else str(self.labels[idx])
                return (float(dist), float(self.lon[idx]), float(self.lat[idx]), lab)
            kk = int(min(max(k, 1), self.tree.n))
            dist, idx = self.tree.query(pt, k=kk)
            for d, i in zip(np.atleast_1d(dist), np.atleast_1d(idx)):
                lab = None if self.labels is None else str(self.labels[int(i)])
                if lab == label:
                    return (float(d), float(self.lon[int(i)]), float(self.lat[int(i)]), lab)
            return float("inf"), float(lon), float(lat), None
        if self._grid:
            cell = self._cell
            ix, iy = int(round(lon / cell)), int(round(lat / cell))
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if (ix + dx, iy + dy) in self._grid:
                        # 无 scipy 时只能给粗略结果, 不再精确吸附
                        return 0.0, float(lon), float(lat), label
            return float(cell * 111320.0), float(lon), float(lat), None
        return float("inf"), float(lon), float(lat), None


# ----------------------------------------------------------------------------
# 数据集基类
# ----------------------------------------------------------------------------
class BaseDataset:
    """所有数据集的公共接口."""

    kind = "data"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.name = self.path.name
        self.product = self.kind
        self.beams: List[str] = []

    # -- 元信息 --------------------------------------------------------
    @property
    def point_count(self) -> int:
        raise NotImplementedError

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """(lon_min, lon_max, lat_min, lat_max)."""
        raise NotImplementedError

    def tracks(self, max_points: int = 6000) -> List[dict]:
        """返回用于地图绘制的折线: [{'label','lon','lat','color','stride'}]."""
        return []

    def track_full_arrays(self) -> List[Tuple[str, np.ndarray, np.ndarray, str]]:
        """返回完整轨迹点 ``[(label, lon, lat, color)]`` (供地图按视野裁剪)."""
        out = []
        for tr in self.tracks(max_points=200000):
            out.append((tr.get("label", self.name), np.asarray(tr["lon"], "f8"),
                        np.asarray(tr["lat"], "f8"),
                        tr.get("color", BEAM_COLORS.get(self.kind, "#444444"))))
        return out

    def summary(self) -> List[Tuple[str, str]]:
        return [("文件", self.name), ("产品", self.product), ("类型", self.kind)]

    # -- 覆盖 ----------------------------------------------------------
    def coverage_index(self) -> Optional[CoverageIndex]:
        return None

    # -- 剖面 ----------------------------------------------------------
    def select(self, corridor: CorridorGeometry, options: ReadOptions) -> PhotonSelection:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------
# ATL03 —— 惰性读取
# ----------------------------------------------------------------------------
def _open_h5(path: Path):
    if h5py is None:
        raise RuntimeError("缺少 h5py, 请先安装: pip install h5py")
    try:
        return h5py.File(str(path), "r", locking=False)
    except TypeError:  # 老版本 h5py 不支持 locking
        return h5py.File(str(path), "r")


def _h5_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    if isinstance(value, np.ndarray) and value.size:
        return _h5_text(value.reshape(-1)[0])
    return str(value)


def _product_name(h5, default: str = "") -> str:
    for key in ("short_name", "identifier_product_type", "granule_type"):
        try:
            if key in h5.attrs:
                text = _h5_text(h5.attrs[key]).strip()
                if text:
                    return text
        except Exception:
            pass
    return default


def _cache_path(cache_dir: Path, source: Path, tag: str) -> Path:
    try:
        st = os.stat(source)
        sig = f"{source.resolve()}|{st.st_size}|{int(st.st_mtime)}|{tag}|v1"
    except OSError:
        sig = f"{source}|{tag}|v1"
    name = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:20]
    return Path(cache_dir) / "index" / f"{name}.npz"


class ATL03Dataset(BaseDataset):
    """ICESat-2 ATL03 光子数据 (惰性分块读取)."""

    kind = "ATL03"

    def __init__(self, path: str | Path, cache_dir: Optional[Path] = None,
                 progress: Optional[Callable[[float, str], None]] = None):
        super().__init__(path)
        self.cache_dir = Path(cache_dir or INDEX_CACHE_DIR)
        self.h5 = _open_h5(self.path)
        self.product = _product_name(self.h5, "ATL03")
        self.beams = [b for b in BEAM_ORDER
                      if b in self.h5 and "geolocation" in self.h5[b]]
        if not self.beams:
            raise ValueError("文件中未找到 gt1l/gt1r/... 波束组, 不是标准 ATL03 文件")
        self._index: Dict[str, dict] = {}
        self._geoid: Dict[str, np.ndarray] = {}
        self._coverage: Optional[CoverageIndex] = None
        self._photon_total = 0
        self._build(progress)

    # ------------------------------------------------------------------
    def _build(self, progress) -> None:
        n = len(self.beams)
        for i, beam in enumerate(self.beams):
            if progress:
                progress(i / max(n, 1), f"建立 {beam} 段索引 …")
            self._index[beam] = self._load_index(beam)
            self._photon_total += int(self._index[beam]["cnt"].sum())
        if progress:
            progress(1.0, "索引就绪")

    def _load_index(self, beam: str) -> dict:
        cache = _cache_path(self.cache_dir, self.path, f"atl03idx_{beam}")
        if cache.exists():
            try:
                with np.load(cache, allow_pickle=False) as npz:
                    idx = {k: npz[k] for k in npz.files}
                if idx.get("lat", np.array([])).size:
                    return self._finalize_index(idx)
            except Exception:
                pass
        grp = self.h5[f"{beam}/geolocation"]
        lat = lon = None
        for lname, oname in (("segment_lat", "segment_lon"),
                             ("reference_photon_lat", "reference_photon_lon")):
            if lname in grp and oname in grp:
                lat = grp[lname][:].astype("f8")
                lon = grp[oname][:].astype("f8")
                break
        if lat is None:
            raise ValueError(f"{beam}: geolocation 组中未找到经纬度数据集")
        cnt = grp["segment_ph_cnt"][:].astype("i8")
        beg = grp["ph_index_beg"][:].astype("i8")
        seg_len = (grp["segment_length"][:].astype("f8")
                   if "segment_length" in grp else np.full(cnt.shape, 20.0))
        seg_id = (grp["segment_id"][:].astype("i8")
                  if "segment_id" in grp else np.arange(cnt.size, dtype="i8"))
        # ph_index_beg 在 ATL03 中为 1-based; 若最小有效值已是 0 则认为 0-based
        valid = cnt > 0
        beg0 = np.where(valid, beg - 1, -1)
        if valid.any() and np.nanmin(beg0[valid]) < 0:
            beg0 = np.where(valid, beg, -1)
        idx = {"lat": lat, "lon": lon, "cnt": cnt, "beg0": beg0,
               "seg_len": seg_len, "seg_id": seg_id}
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **idx)
        except Exception:
            pass
        return self._finalize_index(idx)

    @staticmethod
    def _finalize_index(idx: dict) -> dict:
        valid = idx["cnt"] > 0
        beg0 = idx["beg0"]
        idx["has_ph"] = valid
        # 供 searchsorted 使用的"有效段"查找表
        idx["beg_valid"] = beg0[valid]
        idx["seg_valid"] = np.nonzero(valid)[0]
        idx["n_segments"] = int(beg0.size)
        return idx

    # ------------------------------------------------------------------
    @property
    def point_count(self) -> int:
        return int(self._photon_total)

    def segment_count(self, beam: str) -> int:
        return int(self._index[beam]["n_segments"]) if beam in self._index else 0

    @property
    def bounds(self):
        if not self._index:
            return None
        lon_min = min(float(np.nanmin(self._index[b]["lon"])) for b in self.beams)
        lon_max = max(float(np.nanmax(self._index[b]["lon"])) for b in self.beams)
        lat_min = min(float(np.nanmin(self._index[b]["lat"])) for b in self.beams)
        lat_max = max(float(np.nanmax(self._index[b]["lat"])) for b in self.beams)
        return (lon_min, lon_max, lat_min, lat_max)

    def _valid_tracks(self) -> List[Tuple[str, np.ndarray, np.ndarray, str]]:
        """仅含光子的段参与轨迹绘制与覆盖判定.

        注意: 本产品 (ATL03 v007) 中 ``segment_ph_cnt == 0`` 的段其
        ``reference_photon_lat/lon`` 是无效值 (与相邻段相差可达 10 km),
        若一并绘制会出现"阶梯状"假轨迹, 因此必须先按光子数过滤.
        """
        cache = getattr(self, "_tracks_cache", None)
        if cache is None:
            cache = []
            for beam in self.beams:
                idx = self._index[beam]
                sel = idx["seg_valid"]
                cache.append((beam, idx["lon"][sel], idx["lat"][sel],
                              BEAM_COLORS.get(beam, "#444444")))
            self._tracks_cache = cache
        return cache

    def tracks(self, max_points: int = 6000) -> List[dict]:
        out = []
        for label, lon, lat, color in self._valid_tracks():
            n = lon.size
            stride = max(1, int(np.ceil(n / max(int(max_points), 100))))
            out.append({"label": label, "lon": lon[::stride], "lat": lat[::stride],
                        "color": color})
        return out

    def coverage_index(self) -> CoverageIndex:
        if self._coverage is None:
            lons, lats, labels = [], [], []
            for label, lon, lat, _color in self._valid_tracks():
                n = lon.size
                stride = max(1, int(np.ceil(n / 200000)))  # 每波束 <= 20 万点
                sub_lon, sub_lat = lon[::stride], lat[::stride]
                lons.append(sub_lon)
                lats.append(sub_lat)
                labels.append(np.full(sub_lon.size, label, dtype="U16"))
            self._coverage = CoverageIndex(np.concatenate(lons), np.concatenate(lats),
                                           np.concatenate(labels) if labels else None)
        return self._coverage

    def track_full_arrays(self):
        """直接暴露有效段的索引数组, 供地图按视野裁剪."""
        return self._valid_tracks()

    def summary(self) -> List[Tuple[str, str]]:
        rows = [("文件", self.name), ("产品", self.product), ("类型", "ATL03 光子 (惰性读取)"),
                ("波束", ", ".join(self.beams)),
                ("光子总数", f"{self.point_count:,}"),
                ("段数/波束", f"{self.segment_count(self.beams[0]):,}")]
        b = self.bounds
        if b:
            rows.append(("经度范围", f"{b[0]:.4f} ~ {b[1]:.4f}"))
            rows.append(("纬度范围", f"{b[2]:.4f} ~ {b[3]:.4f}"))
        return rows

    # ------------------------------------------------------------------
    def _geoid_of(self, beam: str) -> Optional[np.ndarray]:
        if beam in self._geoid:
            return self._geoid[beam]
        arr = None
        try:
            grp = self.h5[f"{beam}/geophys_corr"]
            if "geoid" in grp:
                arr = grp["geoid"][:].astype("f8")
        except Exception:
            arr = None
        self._geoid[beam] = arr
        return arr

    def select(self, corridor: CorridorGeometry, options: ReadOptions) -> PhotonSelection:
        hw = float(options.half_width_m)
        L = float(corridor.length_m)
        buf = hw + 60.0                    # 段长约 20 m + 光斑横向展布余量
        beams = [b for b in self.beams if b in set(options.beams)] or list(self.beams)

        chunks: List[dict] = []
        read_total = 0
        truncated = 0
        for beam in beams:
            idx = self._index[beam]
            along_s, cross_s = corridor.project(idx["lon"], idx["lat"])
            sel = np.nonzero((along_s >= -buf) & (along_s <= L + buf) &
                             (np.abs(cross_s) <= buf) & idx["has_ph"])[0]
            if sel.size == 0:
                continue
            starts = idx["beg0"][sel]
            ends = starts + idx["cnt"][sel]
            order = np.argsort(starts, kind="stable")
            sel = sel[order]
            starts = starts[order]
            ends = ends[order]
            # 合并光子索引区间 (允许 50000 光子的空隙, 减少 IO 次数)
            gap = 50000
            run_bounds = np.zeros(starts.size, dtype=bool)
            run_bounds[0] = True
            run_bounds[1:] = starts[1:] > (np.maximum.accumulate(ends)[:-1] + gap)
            group = np.cumsum(run_bounds) - 1
            n_run = int(group[-1]) + 1
            run_start = np.full(n_run, np.iinfo("i8").max, dtype="i8")
            run_end = np.zeros(n_run, dtype="i8")
            np.minimum.at(run_start, group, starts)
            np.maximum.at(run_end, group, ends)

            dset = self.h5[f"{beam}/heights"]
            for s, e in zip(run_start, run_end):
                s, e = int(s), int(e)
                if e <= s:
                    continue
                if read_total + (e - s) > MAX_READ_PHOTONS:
                    truncated += (e - s)
                    continue
                part = self._read_run(beam, dset, s, e, corridor, hw, options)
                read_total += (e - s)
                if part is not None and part["lon"].size:
                    chunks.append(part)

        if not chunks:
            return PhotonSelection(
                lon=np.array([]), lat=np.array([]), elev=np.array([]),
                along=np.array([]), cross=np.array([]), beam=np.array([], dtype="U8"),
                corridor=corridor, product=self.product, source=self.name,
                truncated=truncated)

        lon = np.concatenate([c["lon"] for c in chunks])
        lat = np.concatenate([c["lat"] for c in chunks])
        elev = np.concatenate([c["elev"] for c in chunks])
        along = np.concatenate([c["along"] for c in chunks])
        cross = np.concatenate([c["cross"] for c in chunks])
        beam = np.concatenate([c["beam"] for c in chunks])
        geoid_parts = [c["geoid"] for c in chunks if c["geoid"] is not None]
        geoid = np.concatenate(geoid_parts) if len(geoid_parts) == len(chunks) else None

        note = ""
        if options.conf_mode != "全部":
            note = f"置信度筛选: {options.conf_mode} ({options.conf_channel}通道)"
        if geoid is None:
            note = (note + "; " if note else "") + "未找到 geoid, 正交高不可用"
        sel_out = PhotonSelection(lon=lon, lat=lat, elev=elev, along=along, cross=cross,
                                  beam=beam, corridor=corridor, product=self.product,
                                  source=self.name, geoid=geoid, truncated=truncated,
                                  note=note)
        if options.max_photons and len(sel_out) > options.max_photons:
            keep = np.random.default_rng(0).choice(len(sel_out), size=int(options.max_photons),
                                                   replace=False)
            keep.sort()
            dropped = len(sel_out) - keep.size
            sel_out = PhotonSelection(lon=lon[keep], lat=lat[keep], elev=elev[keep],
                                      along=along[keep], cross=cross[keep], beam=beam[keep],
                                      corridor=corridor, product=self.product,
                                      source=self.name,
                                      geoid=None if geoid is None else geoid[keep],
                                      truncated=truncated + dropped,
                                      note=note + f"; 超出 {options.max_photons:,} 已随机抽稀")
        return sel_out

    # ------------------------------------------------------------------
    def _read_run(self, beam: str, dset, s: int, e: int, corridor: CorridorGeometry,
                  hw: float, options: ReadOptions) -> Optional[dict]:
        """读取一段连续光子并做廊道/质量筛选."""
        try:
            lon = dset["lon_ph"][s:e].astype("f8")
            lat = dset["lat_ph"][s:e].astype("f8")
            h_ph = dset["h_ph"][s:e].astype("f8")
        except Exception:
            return None
        along, cross = corridor.project(lon, lat)
        keep = (along >= 0.0) & (along <= corridor.length_m) & (np.abs(cross) <= hw)

        # 置信度筛选
        if options.conf_mode != "全部" and "signal_conf_ph" in dset:
            conf = self._read_conf(dset, s, e, options.conf_channel)
            if conf is not None:
                if options.conf_mode == "中等及以上":
                    keep &= conf >= 3
                elif options.conf_mode == "仅高置信":
                    keep &= conf >= 4
        # 剔除 afterpulse / 脉冲响应等非 nominal 光子
        if options.remove_afterpulse and "quality_ph" in dset:
            try:
                quality = dset["quality_ph"][s:e]
                keep &= (quality == 0)
            except Exception:
                pass

        if not np.any(keep):
            return None
        idx_keep = np.nonzero(keep)[0]
        geoid = None
        geoid_full = self._geoid_of(beam)
        if geoid_full is not None:
            index = self._index[beam]
            ph_index = s + idx_keep
            pos = np.searchsorted(index["beg_valid"], ph_index, side="right") - 1
            pos = np.clip(pos, 0, index["seg_valid"].size - 1)
            seg = index["seg_valid"][pos]
            seg = np.clip(seg, 0, geoid_full.size - 1)
            geoid = geoid_full[seg]
        return {
            "lon": lon[idx_keep], "lat": lat[idx_keep], "elev": h_ph[idx_keep],
            "along": along[idx_keep], "cross": cross[idx_keep],
            "beam": np.full(idx_keep.size, beam, dtype="U8"),
            "geoid": geoid,
        }

    @staticmethod
    def _read_conf(dset, s: int, e: int, channel: str):
        try:
            if channel in CONF_CHANNELS:
                col = CONF_CHANNELS[channel]
                data = dset["signal_conf_ph"][s:e]
                if data.ndim == 2 and data.shape[1] > col:
                    return data[:, col].astype("i2")
                return None
            data = dset["signal_conf_ph"][s:e]
            if data.ndim == 2:
                return data.max(axis=1).astype("i2")
            return data.astype("i2")
        except Exception:
            return None

    def close(self) -> None:
        try:
            self.h5.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# 通用 HDF5 (ATL06 / ATL08 / ATL13 / GLAH14 ...)
# ----------------------------------------------------------------------------
LAT_CANDIDATES = ("segment_lat", "reference_photon_lat", "latitude", "lat_ph",
                  "lat", "d_lat", "lat_30m")
LON_CANDIDATES = ("segment_lon", "reference_photon_lon", "longitude", "lon_ph",
                  "lon", "d_lon", "lon_30m")
ELEV_CANDIDATES = ("h_li", "h_te_best_fit", "ht_ortho", "height", "elevation",
                   "h_ph", "d_elev", "h_mean", "z")
GEOID_CANDIDATES = ("geoid", "d_gdht", "geoid_h", "geoid_free2mean")


def _find_dataset(group, candidates: Sequence[str]):
    """在 group 及其一级子组中按候选名查找数据集."""
    for name in candidates:
        obj = group.get(name)
        if isinstance(obj, h5py.Dataset) and obj.size:
            return name, obj
    for name in group:
        obj = group[name]
        if isinstance(obj, h5py.Group):
            for cand in candidates:
                sub = obj.get(cand)
                if isinstance(sub, h5py.Dataset) and sub.size:
                    return f"{name}/{cand}", sub
    return None, None


class H5PointDataset(BaseDataset):
    """通用 HDF5 点数据集 (ATL06/ATL08/ATL13/GLAH14 以及类似结构)."""

    def __init__(self, path: str | Path, cache_dir: Optional[Path] = None,
                 progress: Optional[Callable[[float, str], None]] = None):
        super().__init__(path)
        self.h5 = _open_h5(self.path)
        self.product = _product_name(self.h5, "HDF5")
        self.kind = self.product.split("_")[0] if self.product else "HDF5"
        self._groups: List[Tuple[str, Dict[str, np.ndarray]]] = []
        self._coverage: Optional[CoverageIndex] = None
        self._load(progress)

    # ------------------------------------------------------------------
    def _load(self, progress) -> None:
        beam_names = [b for b in BEAM_ORDER if b in self.h5]
        targets: List[Tuple[str, object]] = ([(b, self.h5[b]) for b in beam_names]
                                            if beam_names else [("data", self.h5)])
        total = 0
        for i, (label, group) in enumerate(targets):
            if progress:
                progress(i / max(len(targets), 1), f"读取 {label} …")
            lat_path, lat_ds = _find_dataset(group, LAT_CANDIDATES)
            lon_path, lon_ds = _find_dataset(group, LON_CANDIDATES)
            elev_path, elev_ds = _find_dataset(group, ELEV_CANDIDATES)
            if lat_ds is None or lon_ds is None or elev_ds is None:
                continue
            n = int(min(lat_ds.size, lon_ds.size, elev_ds.size))
            if n > 60_000_000:      # 保护: 避免误读超大数组
                continue
            lon = np.asarray(lon_ds[:n], "f8").reshape(-1)
            lat = np.asarray(lat_ds[:n], "f8").reshape(-1)
            elev = np.asarray(elev_ds[:n], "f8").reshape(-1)
            ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(elev)
            if not np.any(ok):
                continue
            entry = {"lon": lon[ok], "lat": lat[ok], "elev": elev[ok],
                     "paths": (lat_path, lon_path, elev_path)}
            _, geoid_ds = _find_dataset(group, GEOID_CANDIDATES)
            if geoid_ds is not None and geoid_ds.size >= n:
                g = np.asarray(geoid_ds[:n], "f8").reshape(-1)
                if g.size == n:
                    entry["geoid"] = g[ok]
            self._groups.append((label, entry))
            total += int(ok.sum())
            self.beams.append(label)
        self._count = total
        if progress:
            progress(1.0, "读取完成")
        if not self._groups:
            raise ValueError("未能识别经纬度/高程数据集, 不支持的文件结构")

    # ------------------------------------------------------------------
    @property
    def point_count(self) -> int:
        return int(self._count)

    @property
    def bounds(self):
        if not self._groups:
            return None
        lon = np.concatenate([g["lon"] for _, g in self._groups])
        lat = np.concatenate([g["lat"] for _, g in self._groups])
        return (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))

    def tracks(self, max_points: int = 6000) -> List[dict]:
        out = []
        for label, grp in self._groups:
            n = grp["lon"].size
            stride = max(1, int(np.ceil(n / max(int(max_points), 100))))
            out.append({"label": label, "lon": grp["lon"][::stride], "lat": grp["lat"][::stride],
                        "color": BEAM_COLORS.get(label, "#444444")})
        return out

    def coverage_index(self) -> CoverageIndex:
        if self._coverage is None:
            lon = np.concatenate([g["lon"] for _, g in self._groups])
            lat = np.concatenate([g["lat"] for _, g in self._groups])
            labels = np.concatenate([np.full(g["lon"].size, label, dtype="U16")
                                     for label, g in self._groups])
            self._coverage = CoverageIndex(lon, lat, labels)
        return self._coverage

    def track_full_arrays(self):
        return [(label, grp["lon"], grp["lat"], BEAM_COLORS.get(label, "#444444"))
                for label, grp in self._groups]

    def summary(self) -> List[Tuple[str, str]]:
        rows = [("文件", self.name), ("产品", self.product), ("类型", f"{self.kind} 点数据"),
                ("点数", f"{self.point_count:,}")]
        for label, grp in self._groups:
            rows.append((f"  {label}", f"{grp['lon'].size:,} 点"))
        paths = self._groups[0][1].get("paths")
        if paths:
            rows.append(("字段", " / ".join(str(p) for p in paths)))
        b = self.bounds
        if b:
            rows.append(("经度范围", f"{b[0]:.4f} ~ {b[1]:.4f}"))
            rows.append(("纬度范围", f"{b[2]:.4f} ~ {b[3]:.4f}"))
        return rows

    def select(self, corridor: CorridorGeometry, options: ReadOptions) -> PhotonSelection:
        hw = float(options.half_width_m)
        lons, lats, elevs, alongs, crosss, beams, geoids = [], [], [], [], [], [], []
        have_geoid = True
        for label, grp in self._groups:
            if options.beams and label not in set(options.beams) and label != "data":
                continue
            along, cross = corridor.project(grp["lon"], grp["lat"])
            keep = ((along >= 0) & (along <= corridor.length_m) & (np.abs(cross) <= hw))
            if not np.any(keep):
                continue
            lons.append(grp["lon"][keep]); lats.append(grp["lat"][keep])
            elevs.append(grp["elev"][keep]); alongs.append(along[keep]); crosss.append(cross[keep])
            beams.append(np.full(int(keep.sum()), label, dtype="U8"))
            if "geoid" in grp:
                geoids.append(grp["geoid"][keep])
            else:
                have_geoid = False
        if not lons:
            return PhotonSelection(lon=np.array([]), lat=np.array([]), elev=np.array([]),
                                   along=np.array([]), cross=np.array([]),
                                   beam=np.array([], dtype="U8"), corridor=corridor,
                                   product=self.product, source=self.name)
        return PhotonSelection(
            lon=np.concatenate(lons), lat=np.concatenate(lats), elev=np.concatenate(elevs),
            along=np.concatenate(alongs), cross=np.concatenate(crosss),
            beam=np.concatenate(beams), corridor=corridor, product=self.product,
            source=self.name,
            geoid=np.concatenate(geoids) if (have_geoid and geoids) else None)

    def close(self) -> None:
        try:
            self.h5.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# CSV / TXT
# ----------------------------------------------------------------------------
class TableDataset(BaseDataset):
    """ASCII 表格点数据 (CSV/TXT), 自动识别列名."""

    kind = "CSV"

    def __init__(self, path: str | Path, cache_dir: Optional[Path] = None,
                 progress: Optional[Callable[[float, str], None]] = None):
        super().__init__(path)
        self.product = self.path.suffix.lstrip(".").upper() or "CSV"
        self.kind = "CSV"
        self._coverage = None
        self._geoid = None
        self._load(progress)

    def _load(self, progress) -> None:
        if progress:
            progress(0.1, "解析文本 …")
        has_header = self._has_header()
        names: Optional[List[str]] = None
        data = None
        try:
            import pandas as pd  # waterenv 自带

            frame = pd.read_csv(self.path, sep=None, engine="python", comment="#",
                                header=0 if has_header else None, nrows=20_000_000)
            names = [str(c).strip() for c in frame.columns] if has_header else None
            data = frame
        except Exception:
            data = None
        if data is None:
            arr = np.genfromtxt(self.path, delimiter=None, names=has_header, comments="#",
                                invalid_raise=False, max_rows=20_000_000)
            if arr.ndim != 2 and not has_header:
                raise ValueError("文本文件解析失败 (至少需要 3 列: lon, lat, elev)")
            names = list(arr.dtype.names) if has_header else None
            values = (np.column_stack([arr[n] for n in arr.dtype.names])
                      if has_header else arr)
        else:
            values = data.to_numpy(dtype="f8", na_value=np.nan)
        values = np.atleast_2d(values)
        if values.shape[1] < 3:
            raise ValueError("文本文件列数少于 3, 无法解析 (需要 lon, lat, elev)")

        idx = self._detect_columns(names, values)
        lon = values[:, idx["lon"]].astype("f8")
        lat = values[:, idx["lat"]].astype("f8")
        elev = values[:, idx["elev"]].astype("f8")
        ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(elev)
        self.lon, self.lat, self.elev = lon[ok], lat[ok], elev[ok]
        self._count = int(ok.sum())
        if self._count == 0:
            raise ValueError("文本文件中没有有效数据行")
        if progress:
            progress(1.0, "读取完成")

    def _has_header(self) -> bool:
        """通过首行能否解析为数字来判断是否存在表头."""
        try:
            with open(self.path, "r", encoding="utf-8", errors="ignore") as fh:
                for _ in range(10):
                    line = fh.readline()
                    if not line:
                        return False
                    text = line.strip()
                    if not text or text.startswith("#") or text.startswith("//"):
                        continue
                    tokens = [t for t in text.replace(",", " ").replace("\t", " ")
                              .replace(";", " ").split() if t]
                    if not tokens:
                        continue
                    try:
                        for tok in tokens[:6]:
                            float(tok)
                        return False           # 首行全是数字 -> 无表头
                    except ValueError:
                        return True            # 首行含文字 -> 有表头
        except OSError:
            return False
        return False

    @staticmethod
    def _detect_columns(names: Optional[List[str]], values: np.ndarray) -> Dict[str, int]:
        def find(cands, exclude=()):
            if names:
                low = [n.lower() for n in names]
                for cand in cands:
                    for i, nm in enumerate(low):
                        if i in exclude:
                            continue
                        if nm == cand or nm.startswith(cand):
                            return i
            return None

        lat_i = find(["lat", "latitude", "y"])
        lon_i = find(["lon", "lng", "long", "longitude", "x"])
        elev_i = find(["elev", "height", "h_elev", "ortho", "ellip", "z", "h"],
                      exclude={i for i in (lat_i, lon_i) if i is not None})
        if lat_i is None or lon_i is None or elev_i is None:
            ncol = values.shape[1]
            if ncol < 3:
                raise ValueError("文本文件列数少于 3, 无法解析 (需要 lon,lat,elev)")
            # 无表头: 用取值范围推测
            first, second = values[:, 0], values[:, 1]
            finite = np.isfinite(first)
            looks_lon_first = finite.any() and np.nanmax(np.abs(first[finite])) > 90
            lon_i = 0 if looks_lon_first else 1
            lat_i = 1 - lon_i
            elev_i = 2
        return {"lon": int(lon_i), "lat": int(lat_i), "elev": int(elev_i)}

    # ------------------------------------------------------------------
    @property
    def point_count(self) -> int:
        return int(self._count)

    @property
    def bounds(self):
        return (float(self.lon.min()), float(self.lon.max()),
                float(self.lat.min()), float(self.lat.max()))

    def tracks(self, max_points: int = 6000) -> List[dict]:
        n = self.lon.size
        stride = max(1, int(np.ceil(n / max(int(max_points), 100))))
        return [{"label": self.name, "lon": self.lon[::stride], "lat": self.lat[::stride],
                 "color": BEAM_COLORS["CSV"]}]

    def coverage_index(self) -> CoverageIndex:
        if self._coverage is None:
            self._coverage = CoverageIndex(self.lon, self.lat,
                                           np.full(self.lon.size, self.kind, dtype="U16"))
        return self._coverage

    def track_full_arrays(self):
        return [(self.name, self.lon, self.lat, BEAM_COLORS["CSV"])]

    def summary(self) -> List[Tuple[str, str]]:
        b = self.bounds
        return [("文件", self.name), ("产品", self.product), ("类型", "文本点数据"),
                ("点数", f"{self.point_count:,}"),
                ("经度范围", f"{b[0]:.4f} ~ {b[1]:.4f}"),
                ("纬度范围", f"{b[2]:.4f} ~ {b[3]:.4f}")]

    def select(self, corridor: CorridorGeometry, options: ReadOptions) -> PhotonSelection:
        hw = float(options.half_width_m)
        along, cross = corridor.project(self.lon, self.lat)
        keep = (along >= 0) & (along <= corridor.length_m) & (np.abs(cross) <= hw)
        n = int(keep.sum())
        return PhotonSelection(
            lon=self.lon[keep], lat=self.lat[keep], elev=self.elev[keep],
            along=along[keep], cross=cross[keep],
            beam=np.full(n, self.kind, dtype="U8"), corridor=corridor,
            product=self.product, source=self.name)


# ----------------------------------------------------------------------------
# LAS / LAZ (内置最小读取器)
# ----------------------------------------------------------------------------
class LASDataset(BaseDataset):
    """LAS/LAZ 点云. LAZ 需要 laspy; LAS 由内置解析器直接读取."""

    kind = "LAS"

    def __init__(self, path: str | Path, cache_dir: Optional[Path] = None,
                 progress: Optional[Callable[[float, str], None]] = None):
        super().__init__(path)
        self.product = self.path.suffix.lstrip(".").upper()
        self._coverage = None
        self._load(progress)

    def _load(self, progress) -> None:
        if self.path.suffix.lower() == ".laz":
            try:
                import laspy  # type: ignore

                las = laspy.read(str(self.path))
                self.lon = np.asarray(las.x, "f8")
                self.lat = np.asarray(las.y, "f8")
                self.elev = np.asarray(las.z, "f8")
                if np.nanmax(np.abs(self.lon)) > 180:   # 投影坐标 -> 经纬度
                    self.lon, self.lat = self._to_wgs84(self.lon, self.lat, las.header)
            except ImportError as exc:
                raise ValueError("读取 LAZ 需要 laspy, 请执行: pip install laspy[lazrs]") from exc
        else:
            self.lon, self.lat, self.elev = self._read_las(progress)
        if progress:
            progress(1.0, "读取完成")
        self._count = int(self.lon.size)

    # ------------------------------------------------------------------
    def _read_las(self, progress=None):
        with open(self.path, "rb") as fh:
            head = fh.read(375)
            if head[:4] != b"LASF":
                raise ValueError("不是有效的 LAS 文件")
            offset_to_points = int.from_bytes(head[96:100], "little")
            point_format = head[104] & 0x3F
            record_len = int.from_bytes(head[105:107], "little")
            legacy_count = int.from_bytes(head[107:111], "little")
            scale = np.frombuffer(head[131:155], "<f8", 3)
            offs = np.frombuffer(head[155:179], "<f8", 3)
            count = legacy_count
            if len(head) >= 375 and point_format >= 6:
                c64 = int.from_bytes(head[247:255], "little")
                if c64:
                    count = c64
            if count == 0:
                raise ValueError("LAS 文件点数为 0")
            if progress:
                progress(0.2, f"读取 {count:,} 个 LAS 点 …")
            fh.seek(offset_to_points)
            dtype = np.dtype({"names": ["x", "y", "z", "_pad"],
                              "formats": ["<i4", "<i4", "<i4", f"V{max(record_len - 12, 0)}"],
                              "offsets": [0, 4, 8, 12], "itemsize": record_len})
            raw = np.frombuffer(fh.read(record_len * count), dtype=dtype, count=count)
        x = raw["x"].astype("f8") * scale[0] + offs[0]
        y = raw["y"].astype("f8") * scale[1] + offs[1]
        z = raw["z"].astype("f8") * scale[2] + offs[2]
        if np.nanmax(np.abs(x)) > 180 or np.nanmax(np.abs(y)) > 90:
            x, y = self._to_wgs84(x, y)
        return x, y, z

    def _to_wgs84(self, x, y, header=None):
        """投影坐标 -> WGS84 经纬度 (优先读 LAS 内的 WKT)."""
        wkt = self._read_wkt()
        if wkt:
            try:
                from pyproj import CRS, Transformer

                transformer = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_epsg(4326),
                                                   always_xy=True)
                lon, lat = transformer.transform(x, y)
                return np.asarray(lon, "f8"), np.asarray(lat, "f8")
            except Exception:
                pass
        raise ValueError("LAS 使用投影坐标且未找到可用坐标系定义, 请先转换为经纬度")

    def _read_wkt(self) -> Optional[str]:
        try:
            with open(self.path, "rb") as fh:
                head = fh.read(375)
                n_vlr = int.from_bytes(head[100:104], "little")
                offset = int.from_bytes(head[96:100], "little")
                fh.seek(offset)
                for _ in range(max(n_vlr, 0)):
                    rec = fh.read(54)
                    if len(rec) < 54:
                        break
                    user_id = rec[2:18].rstrip(b"\x00").decode("ascii", "ignore")
                    record_id = int.from_bytes(rec[18:20], "little")
                    length = int.from_bytes(rec[20:22], "little")
                    payload = fh.read(length)
                    if user_id.strip() == "LASF_Projection" and record_id in (2111, 2112):
                        text = payload.rstrip(b"\x00").decode("utf-8", "ignore")
                        if "PROJCS" in text or "PROJCRS" in text:
                            return text
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    @property
    def point_count(self) -> int:
        return int(self._count)

    @property
    def bounds(self):
        return (float(self.lon.min()), float(self.lon.max()),
                float(self.lat.min()), float(self.lat.max()))

    def tracks(self, max_points: int = 6000) -> List[dict]:
        n = self.lon.size
        stride = max(1, int(np.ceil(n / max(int(max_points), 100))))
        return [{"label": self.name, "lon": self.lon[::stride], "lat": self.lat[::stride],
                 "color": BEAM_COLORS["LAS"]}]

    def coverage_index(self) -> CoverageIndex:
        if self._coverage is None:
            self._coverage = CoverageIndex(self.lon, self.lat,
                                           np.full(self.lon.size, self.kind, dtype="U16"))
        return self._coverage

    def track_full_arrays(self):
        return [(self.name, self.lon, self.lat, BEAM_COLORS["LAS"])]

    def summary(self) -> List[Tuple[str, str]]:
        b = self.bounds
        return [("文件", self.name), ("产品", self.product), ("类型", "LAS/LAZ 点云"),
                ("点数", f"{self.point_count:,}"),
                ("经度范围", f"{b[0]:.4f} ~ {b[1]:.4f}"),
                ("纬度范围", f"{b[2]:.4f} ~ {b[3]:.4f}")]

    def select(self, corridor: CorridorGeometry, options: ReadOptions) -> PhotonSelection:
        hw = float(options.half_width_m)
        along, cross = corridor.project(self.lon, self.lat)
        keep = (along >= 0) & (along <= corridor.length_m) & (np.abs(cross) <= hw)
        n = int(keep.sum())
        return PhotonSelection(
            lon=self.lon[keep], lat=self.lat[keep], elev=self.elev[keep],
            along=along[keep], cross=cross[keep],
            beam=np.full(n, self.kind, dtype="U8"), corridor=corridor,
            product=self.product, source=self.name)


# ----------------------------------------------------------------------------
# 工厂
# ----------------------------------------------------------------------------
H5_SUFFIXES = {".h5", ".hdf5", ".hdf", ".h4", ".he5"}
TEXT_SUFFIXES = {".csv", ".txt", ".dat", ".xyz", ".tsv", ".asc"}
LAS_SUFFIXES = {".las", ".laz"}

OPEN_FILETYPES = [
    ("ICESat 数据", "*.h5 *.hdf5 *.hdf *.he5 *.csv *.txt *.dat *.xyz *.tsv *.las *.laz"),
    ("ICESat-2 / ICESat HDF5", "*.h5 *.hdf5 *.hdf *.he5"),
    ("文本点数据", "*.csv *.txt *.dat *.xyz *.tsv"),
    ("LAS/LAZ 点云", "*.las *.laz"),
    ("全部文件", "*.*"),
]


def merge_selections(selections: Sequence[PhotonSelection],
                     corridor: CorridorGeometry) -> PhotonSelection:
    """合并多个数据集在同一条剖切线上的选择结果."""
    parts = [s for s in selections if s is not None and len(s) > 0]
    if not parts:
        return PhotonSelection(lon=np.array([]), lat=np.array([]), elev=np.array([]),
                               along=np.array([]), cross=np.array([]),
                               beam=np.array([], dtype="U8"), corridor=corridor)
    geoid = (np.concatenate([p.geoid for p in parts])
             if all(p.geoid is not None for p in parts) else None)
    notes = [f"{p.source}: {len(p):,} 光子" + (f"({p.note})" if p.note else "")
             for p in parts]
    return PhotonSelection(
        lon=np.concatenate([p.lon for p in parts]),
        lat=np.concatenate([p.lat for p in parts]),
        elev=np.concatenate([p.elev for p in parts]),
        along=np.concatenate([p.along for p in parts]),
        cross=np.concatenate([p.cross for p in parts]),
        beam=np.concatenate([p.beam for p in parts]),
        corridor=corridor,
        product="+".join(sorted({p.product for p in parts})),
        source=" + ".join(p.source for p in parts),
        geoid=geoid,
        truncated=int(sum(p.truncated for p in parts)),
        note="; ".join(notes))


def open_dataset(path: str | Path, cache_dir: Optional[Path] = None,
                 progress: Optional[Callable[[float, str], None]] = None) -> BaseDataset:
    """按扩展名/内容自动选择读取器."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix in LAS_SUFFIXES:
        return LASDataset(path, cache_dir, progress)
    if suffix in TEXT_SUFFIXES:
        return TableDataset(path, cache_dir, progress)
    if suffix in H5_SUFFIXES or suffix == "":
        product = ""
        is_atl03 = False
        if h5py is None:
            raise RuntimeError("缺少 h5py, 无法读取 HDF5 文件")
        with _open_h5(path) as h5:
            product = _product_name(h5, "")
            for beam in BEAM_ORDER:
                grp = h5.get(beam)
                if isinstance(grp, h5py.Group) and "geolocation" in grp:
                    geo_grp = grp["geolocation"]
                    if "ph_index_beg" in geo_grp and "segment_ph_cnt" in geo_grp:
                        is_atl03 = True
                        break
        if is_atl03 or "ATL03" in product.upper():
            return ATL03Dataset(path, cache_dir, progress)
        return H5PointDataset(path, cache_dir, progress)
    raise ValueError(f"不支持的文件类型: {suffix}")
