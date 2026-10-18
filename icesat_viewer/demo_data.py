# -*- coding: utf-8 -*-
"""生成 ATL03 结构的合成演示数据.

用途: 在没有真实 ICESat-2 文件时可以立即体验"两点连线剖面"功能, 也可用于回归测试.
生成文件结构与真实 ATL03 保持一致 (geolocation 段索引 + heights 光子 + geophys_corr),
因此可以被 :class:`icesat_viewer.icesat_io.ATL03Dataset` 正常惰性读取.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .geo import GEOD
from .icesat_io import BEAM_ORDER

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None  # type: ignore


def generate_demo_atl03(path: str | Path,
                        lon0: float = 120.170, lat0: float = 31.190,
                        azimuth: float = 105.0, length_km: float = 12.0,
                        beams: Sequence[str] = BEAM_ORDER,
                        segment_length_m: float = 20.0,
                        seed: int = 20250105) -> Path:
    """生成一个含湖泊/丘陵场景的合成 ATL03 文件.

    参数
    ----
    path      输出 .h5 路径
    lon0/lat0 轨迹起点
    azimuth   轨迹方位角 (度)
    length_km 轨迹长度
    beams     需要生成的波束
    """
    if h5py is None:
        raise RuntimeError("生成演示数据需要 h5py")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_seg = int(length_km * 1000 / segment_length_m)
    s_axis = np.arange(n_seg) * segment_length_m
    lon_c, lat_c = GEOD.forward(np.full(n_seg, lon0), np.full(n_seg, lat0),
                                np.full(n_seg, azimuth), s_axis)
    # 沿迹真实地表: 湖面 + 丘陵 + 缓坡
    surface = (3.6 + 12.0 * np.exp(-((s_axis - 0.45 * length_km * 1000)
                                     / (0.18 * length_km * 1000)) ** 2)
               + 6.0 * np.sin(s_axis / 900.0) + s_axis * 2e-5)

    with h5py.File(path, "w") as h5:
        h5.attrs["short_name"] = np.bytes_("ATL03")
        h5.attrs["identifier_product_type"] = np.bytes_("ATL03")
        h5.attrs["title"] = np.bytes_("ATLAS/ICESat-2 L2A Global Geolocated Photon Data (demo)")
        h5.attrs["geospatial_lon_min"] = float(lon_c.min())
        h5.attrs["geospatial_lon_max"] = float(lon_c.max())
        h5.attrs["geospatial_lat_min"] = float(lat_c.min())
        h5.attrs["geospatial_lat_max"] = float(lat_c.max())

        orbit = h5.create_group("orbit_info")
        orbit.create_dataset("rgt", data=np.array([999], "i2"))
        orbit.create_dataset("cycle_number", data=np.array([1], "i1"))
        orbit.create_dataset("sc_orient", data=np.array([1], "i1"))

        for bi, beam in enumerate(beams):
            # 各波束横向错开约 3 km (强弱波束对间距 ~90 m, 波束组间距 ~3.3 km)
            offset_km = (bi // 2) * 3.3 + (0.045 if bi % 2 else -0.045)
            az_perp = azimuth + 90.0
            lon_b, lat_b = GEOD.forward(lon_c, lat_c, np.full(n_seg, az_perp),
                                        np.full(n_seg, offset_km * 1000.0))
            n_ph_seg = rng.integers(120, 220, n_seg)
            total = int(n_ph_seg.sum())

            # 每个光子: 段内随机位置 + 高斯噪声 (信号) / 均匀噪声 (背景)
            seg_of_ph = np.repeat(np.arange(n_seg), n_ph_seg)
            frac = rng.random(total)
            along = s_axis[seg_of_ph] + frac * segment_length_m
            signal = rng.random(total) < 0.55
            h_ph = np.where(signal,
                            surface[seg_of_ph] + rng.normal(0, 0.28, total),
                            rng.uniform(-45, 45, total))
            # 光子经纬度: 沿轨位置 + 光斑内随机横向偏移
            d_along = along - s_axis[seg_of_ph]
            lon_ph, lat_ph = GEOD.forward(lon_b[seg_of_ph], lat_b[seg_of_ph],
                                          np.full(total, azimuth), d_along)
            across = rng.normal(0, 4.5, total)
            lon_ph, lat_ph = GEOD.forward(lon_ph, lat_ph, np.full(total, az_perp), across)

            geoloc = h5.create_group(f"{beam}/geolocation")
            geoloc.create_dataset("segment_lat", data=lat_b)
            geoloc.create_dataset("segment_lon", data=lon_b)
            geoloc.create_dataset("reference_photon_lat", data=lat_b)
            geoloc.create_dataset("reference_photon_lon", data=lon_b)
            geoloc.create_dataset("segment_length", data=np.full(n_seg, segment_length_m))
            geoloc.create_dataset("segment_id", data=np.arange(n_seg, dtype="i4"))
            geoloc.create_dataset("segment_ph_cnt", data=n_ph_seg.astype("i4"))
            starts = np.concatenate([[1], np.cumsum(n_ph_seg)[:-1] + 1]).astype("i8")
            geoloc.create_dataset("ph_index_beg", data=starts)
            geoloc.create_dataset("delta_time", data=s_axis / 7000.0)
            conf = np.where(signal, 4, 1).astype("i1")
            geoloc.create_dataset("surf_type", data=np.zeros((n_seg, 5), "i1"))
            geoloc.create_dataset("reference_photon_index", data=np.ones(n_seg, "i4"))

            heights = h5.create_group(f"{beam}/heights")
            heights.create_dataset("lat_ph", data=lat_ph)
            heights.create_dataset("lon_ph", data=lon_ph)
            heights.create_dataset("h_ph", data=h_ph.astype("f4"))
            heights.create_dataset("dist_ph_along", data=d_along.astype("f4"))
            heights.create_dataset("dist_ph_across", data=across.astype("f4"))
            heights.create_dataset("delta_time", data=(along / 7000.0))
            heights.create_dataset("quality_ph", data=np.where(signal, 0, 1).astype("i1"))
            conf5 = np.repeat(conf[:, None], 5, axis=1)
            conf5[~signal, 4] = 2               # 内陆水体通道对噪声给中等置信
            heights.create_dataset("signal_conf_ph", data=conf5)

            geophys = h5.create_group(f"{beam}/geophys_corr")
            geophys.create_dataset("geoid", data=(-8.5 + 0.2 * np.sin(s_axis / 2000.0)
                                                  ).astype("f4"))
    return path


if __name__ == "__main__":  # 允许 python -m icesat_viewer.demo_data
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_ATL03.h5")
    print("已生成:", generate_demo_atl03(out))
