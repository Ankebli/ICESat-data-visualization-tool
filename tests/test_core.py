# -*- coding: utf-8 -*-
"""核心算法测试 (无需 GUI).

运行::

    C:\\Users\\ASUS\\miniconda3\\envs\\waterenv\\python.exe tests\\test_core.py

会使用仓库外的真实 ATL03 文件 (若存在) 做一次端到端廊道提取, 否则自动跳过.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from icesat_viewer import geo                                        # noqa: E402
from icesat_viewer.corridor import CorridorGeometry, binned_statistic  # noqa: E402
from icesat_viewer.icesat_io import (ReadOptions, TableDataset,  # noqa: E402
                                     open_dataset)

REAL_ATL03 = Path(r"D:\Paper\Data\satellite image\ICESAT\Taihu"
                  r"\ATL03_20250105065215_03082606_007_01.h5")


class TestGeo(unittest.TestCase):
    def test_mercator_roundtrip(self):
        lon = np.array([120.2, -75.5, 0.0, 179.9])
        lat = np.array([31.2, -12.0, 0.0, -60.0])
        x, y = geo.lonlat_to_mercator(lon, lat)
        lon2, lat2 = geo.mercator_to_lonlat(x, y)
        self.assertTrue(np.allclose(lon, lon2, atol=1e-9))
        self.assertTrue(np.allclose(lat, lat2, atol=1e-9))

    def test_scalar_roundtrip(self):
        x, y = geo.lonlat_to_mercator(120.0, 31.0)
        lon, lat = geo.mercator_to_lonlat(x, y)
        self.assertAlmostEqual(lon, 120.0, places=9)
        self.assertAlmostEqual(lat, 31.0, places=9)

    def test_zoom_and_tiles(self):
        # z=0 分辨率约 156543 m/px
        self.assertAlmostEqual(geo.resolution(0), 156543.03392804097, places=6)
        z = geo.zoom_for_resolution(1200.0)          # 约 1.2 km/px
        self.assertEqual(z, geo.zoom_for_resolution(1200.0))
        self.assertTrue(4 <= z <= 8)
        x, y = geo.lonlat_to_mercator(120.2, 31.2)
        tx0, tx1, ty0, ty1 = geo.tile_index_range(x - 50000, x + 50000, y - 50000, y + 50000, 10)
        self.assertLessEqual(tx0, tx1)
        self.assertLessEqual(ty0, ty1)
        bx0, bx1, by0, by1 = geo.tile_bounds_mercator(tx0, ty0, 10)
        self.assertAlmostEqual(bx1 - bx0, geo.tile_span(10), places=6)
        self.assertAlmostEqual(by1 - by0, geo.tile_span(10), places=6)

    def test_geodesic_distance(self):
        # 北京 -> 上海 约 1067 km
        d = geo.GEOD.distance(116.4074, 39.9042, 121.4737, 31.2304)
        self.assertTrue(1050e3 < d < 1080e3, d)

    def test_project_on_line(self):
        p0, p1 = (120.0, 31.0), (120.1, 31.0)
        along, cross = geo.project_on_line(p0, p1, np.array([120.05, 120.05]),
                                           np.array([31.0, 31.01]))
        self.assertAlmostEqual(float(along[0]), geo.GEOD.distance(*p0, 120.05, 31.0), delta=1.0)
        # 大地线相对纬圈在 4.8 km 尺度上向北凸出约 1 m, 属正常测地线效应
        self.assertAlmostEqual(float(cross[0]), 0.0, delta=2.0)
        # 约定: 沿迹方向右侧为正; 东向线的右侧为南, 故北侧点为负
        self.assertGreater(abs(float(cross[1])), 1000.0)


class TestCorridor(unittest.TestCase):
    def test_mask_and_stats(self):
        rng = np.random.default_rng(0)
        n = 5000
        lon = 120.0 + rng.uniform(0, 0.05, n)
        lat = 31.0 + rng.uniform(-0.01, 0.01, n)
        elev = 10.0 + rng.normal(0, 0.5, n)
        cg = CorridorGeometry((120.0, 31.0), (120.05, 31.0), half_width_m=50.0)
        mask = cg.mask(lon, lat)
        self.assertGreater(mask.sum(), 0)
        along, cross = cg.project(lon[mask], lat[mask])
        self.assertTrue(np.all(np.abs(cross) <= 50.0 + 1e-6))
        self.assertTrue(np.all(along >= -1e-6))
        centers, med, counts = binned_statistic(along, elev[mask], 100.0,
                                                length_m=cg.length_m)
        self.assertEqual(centers.size, counts.size)
        self.assertAlmostEqual(float(np.nanmedian(med)), 10.0, delta=0.3)

    def test_polygon(self):
        cg = CorridorGeometry((120.0, 31.0), (120.05, 31.0), 30.0)
        ring = cg.polygon_lonlat()
        self.assertEqual(ring.shape, (5, 2))
        self.assertTrue(np.allclose(ring[0], ring[-1]))
        width = geo.GEOD.distance(ring[0][0], ring[0][1], ring[3][0], ring[3][1])
        self.assertAlmostEqual(width, 60.0, delta=2.0)


class TestTableDataset(unittest.TestCase):
    def test_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "photons.csv"
            lines = ["lon,lat,elev"]
            for i in range(200):
                lines.append(f"{120.0 + i * 1e-5:.6f},{31.0 + i * 1e-6:.6f},{15.0 + i * 0.01:.3f}")
            path.write_text("\n".join(lines), encoding="utf-8")
            ds = TableDataset(path)
            self.assertEqual(ds.point_count, 200)
            cg = CorridorGeometry((120.0, 31.0), (120.002, 31.0), 20.0)
            sel = ds.select(cg, ReadOptions(half_width_m=20.0))
            self.assertGreater(len(sel), 10)
            out = Path(tmp) / "profile.csv"
            sel.to_csv(out)
            self.assertTrue(out.exists() and out.stat().st_size > 100)

    def test_open_dataset_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "points.xyz"
            path.write_text("120.0 31.0 5.0\n120.001 31.001 6.0\n", encoding="utf-8")
            ds = open_dataset(path)
            self.assertEqual(ds.point_count, 2)

    def test_lat_first_without_header(self):
        """无表头且纬度在前时, 应按取值范围判定列顺序."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lat_first.txt"
            path.write_text("31.2000 120.2000 5.5\n31.2001 120.2001 5.6\n"
                            "31.2002 120.2002 5.7\n", encoding="utf-8")
            ds = open_dataset(path)
            self.assertEqual(ds.point_count, 3)
            b = ds.bounds
            self.assertAlmostEqual(b[0], 120.2, places=3)   # lon_min
            self.assertAlmostEqual(b[2], 31.2, places=3)    # lat_min


class TestLASReader(unittest.TestCase):
    """内置 LAS 解析器 (无需 laspy)."""

    @staticmethod
    def _write_las(path: Path, lon0=120.2, lat0=31.2, n=6):
        import struct

        header_size, offset, point_format, rec_len = 227, 227, 1, 28
        header = bytearray(header_size)
        header[0:4] = b"LASF"
        struct.pack_into("<B", header, 24, 1)          # version major
        struct.pack_into("<B", header, 25, 2)          # version minor
        struct.pack_into("<H", header, 94, header_size)
        struct.pack_into("<I", header, 96, offset)
        header[104] = point_format
        struct.pack_into("<H", header, 105, rec_len)
        struct.pack_into("<I", header, 107, n)
        struct.pack_into("<3d", header, 131, 1e-7, 1e-7, 0.001)      # scale
        struct.pack_into("<3d", header, 155, lon0, lat0, 0.0)        # offset
        body = bytearray()
        for i in range(n):
            body += struct.pack("<3i", int(round((i * 1e-4) / 1e-7)),
                                int(round((i * 0.5e-4) / 1e-7)), int(round((5 + i) / 0.001)))
            body += b"\x00" * (rec_len - 12)
        path.write_bytes(bytes(header) + bytes(body))

    def test_read(self):
        from icesat_viewer.icesat_io import LASDataset

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pts.las"
            self._write_las(path)
            ds = LASDataset(path)
            self.assertEqual(ds.point_count, 6)
            self.assertAlmostEqual(ds.lon[3], 120.2003, places=6)
            self.assertAlmostEqual(ds.lat[3], 31.20015, places=6)
            self.assertAlmostEqual(ds.elev[3], 8.0, places=6)
            cg = CorridorGeometry((120.2, 31.2), (120.201, 31.2002), 20.0)
            sel = ds.select(cg, ReadOptions(half_width_m=20.0))
            self.assertGreater(len(sel), 0)

    def test_reject_non_las(self):
        from icesat_viewer.icesat_io import LASDataset

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.las"
            path.write_bytes(b"NOPE" + b"\x00" * 400)
            with self.assertRaises(ValueError):
                LASDataset(path)


def _has_network(timeout: float = 8.0) -> bool:
    """能否访问天地图服务 (用于决定是否跳过在线校验用例)."""
    try:
        import requests

        requests.get("https://t0.tianditu.gov.cn/DataServer?T=vec_w&x=1&y=0&l=1",
                     timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        return True
    except Exception:
        return False


NETWORK_OK = _has_network()


class TestTiandituToken(unittest.TestCase):
    """天地图密钥: 格式校验, 以及"无效密钥必须抛异常"的行为约定."""

    def test_format_validation(self):
        from icesat_viewer.basemap import is_valid_token_format, mask_token

        self.assertTrue(is_valid_token_format("46921bcb6e07be7d8dc2e2f98c7bbbdf"))
        self.assertTrue(is_valid_token_format("A1B2C3D4E5F60718293A4B5C6D7E8F90"))
        for bad in ("", "abc", "46921bcb6e07be7d8dc2e2f98c7bbbd", "z" * 32, None):
            self.assertFalse(is_valid_token_format(bad), bad)
        self.assertEqual(mask_token("46921bcb6e07be7d8dc2e2f98c7bbbdf"), "46921b…bbdf")
        self.assertEqual(mask_token(""), "(空)")

    def test_project_token_configured(self):
        """本机应已配置天地图密钥 (来自本地 config.json 或环境变量 TIANDITU_TK)."""
        from icesat_viewer.settings import PROJECT_TIANDITU_TOKEN, Settings

        # 源码中不应硬编码密钥 (仓库公开), 该常量应为空
        self.assertEqual(PROJECT_TIANDITU_TOKEN, "")
        settings = Settings()
        if not settings.has_token:
            self.skipTest("本机未配置天地图密钥 (config.json / TIANDITU_TK), 跳过")
        self.assertTrue(settings.token_format_ok(), settings.token)

    def test_missing_or_bad_token_raises(self):
        from icesat_viewer.basemap import TiandituTokenError, TileEngine
        from icesat_viewer.settings import TILE_CACHE_DIR

        engine = TileEngine(TILE_CACHE_DIR, token="")
        try:
            with self.assertRaises(TiandituTokenError):
                engine.validate_provider("tianditu_vec")
            engine.set_token("not-a-key")
            with self.assertRaises(TiandituTokenError):
                engine.validate_provider("tianditu_vec")
            engine.set_token("4" * 31)
            with self.assertRaises(TiandituTokenError):
                engine.validate_provider("tianditu_img")
        finally:
            engine.close()

    @unittest.skipUnless(NETWORK_OK, "无法访问天地图服务, 跳过在线校验")
    def test_online_rejected_key_raises(self):
        from icesat_viewer.basemap import BasemapError, TileEngine
        from icesat_viewer.settings import TILE_CACHE_DIR

        engine = TileEngine(TILE_CACHE_DIR, token="0" * 32)   # 格式合法但不存在
        try:
            with self.assertRaises(BasemapError) as ctx:
                engine.validate_provider("tianditu_vec")
            # 403=密钥无效, 418=限流; 两者都属于"不可用"并已抛异常
            msg = str(ctx.exception)
            self.assertTrue("403" in msg or "418" in msg, msg)
        finally:
            engine.close()

    @unittest.skipUnless(NETWORK_OK, "无法访问天地图服务, 跳过在线校验")
    def test_online_valid_key_and_cache_not_masking(self):
        from icesat_viewer.basemap import (PROVIDERS, TiandituQuotaError,
                                           TiandituTokenError, TileEngine,
                                           is_valid_token_format)
        from icesat_viewer.settings import TILE_CACHE_DIR, Settings

        token = Settings().token
        if not is_valid_token_format(token):
            self.skipTest("未配置有效的天地图密钥")
        engine = TileEngine(TILE_CACHE_DIR, token=token)
        try:
            for provider in ("tianditu_img", "tianditu_vec", "tianditu_ter"):
                try:
                    engine.validate_provider(provider)      # 不抛异常即通过
                except TiandituQuotaError as exc:
                    self.skipTest(f"天地图限流/配额用尽, 跳过 (非密钥问题): {exc}")
            arr, info = engine.mosaic("tianditu_img", 12, 3414, 3416, 1674, 1675)
            self.assertIsNone(info["error_exc"])
            self.assertGreater(float(arr[..., :3].std()), 3.0)
            # 已写入缓存的瓦片也不能掩盖"已被服务端拒绝"的密钥
            engine.set_token("0" * 32)
            with self.assertRaises((TiandituTokenError, TiandituQuotaError)):
                engine.validate_provider("tianditu_vec")     # 联网校验 -> 抛异常并记住
            with self.assertRaises((TiandituTokenError, TiandituQuotaError)):
                engine.load_tile(PROVIDERS["tianditu_img"].base, 12, 3415, 1675,
                                 use_cache=True)             # 该瓦片此时已在内存/磁盘缓存中
        finally:
            engine.close()


class TestBasemap(unittest.TestCase):
    def test_provider_urls(self):
        from icesat_viewer.basemap import PROVIDERS, provider_choices

        tdt = PROVIDERS["tianditu_img"]
        url = tdt.base.format_url(10, 843, 388, "abcd1234efgh5678")
        self.assertTrue(url.startswith("https://t"), url)
        self.assertIn("T=img_w", url)
        self.assertIn("x=843", url)
        self.assertIn("y=388", url)
        self.assertIn("l=10", url)
        self.assertIn("tk=abcd1234efgh5678", url)
        # 注记层
        self.assertIn("T=cia_w", tdt.label.format_url(10, 1, 2, "tk"))
        # Esri: {z}/{y}/{x} 顺序
        esri = PROVIDERS["esri_img"].base.format_url(5, 26, 12)
        self.assertTrue(esri.endswith("/tile/5/12/26"), esri)

        # 无密钥时天地图不可用, 备用底图仍可用
        choices = {k: (usable, reason) for k, _n, usable, reason in provider_choices("")}
        self.assertFalse(choices["tianditu_img"][0])
        self.assertTrue(choices["esri_img"][0])
        choices = {k: usable for k, _n, usable, _r in provider_choices("0123456789abcdef")}
        self.assertTrue(choices["tianditu_img"])

    def test_geo_tick_helpers(self):
        from icesat_viewer.mapview import _fmt_deg, _nice_step

        self.assertAlmostEqual(_nice_step(1.0, 5), 0.2, places=6)
        self.assertAlmostEqual(_nice_step(0.37, 6), 0.1, places=6)
        self.assertAlmostEqual(_nice_step(12.0, 6), 2.0, places=6)
        self.assertEqual(_fmt_deg(120.25, 0.05), "120.25°")
        self.assertEqual(_fmt_deg(31.0, 1.0), "31°")


class TestDemoData(unittest.TestCase):
    """合成演示数据 -> 惰性读取 -> 廊道剖面 全链路."""

    def test_demo_roundtrip(self):
        from icesat_viewer.demo_data import generate_demo_atl03

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo_ATL03.h5"
            generate_demo_atl03(path, length_km=6.0)
            self.assertTrue(path.exists())
            ds = open_dataset(path)
            try:
                self.assertEqual(ds.kind, "ATL03")
                self.assertEqual(len(ds.beams), 6)
                self.assertGreater(ds.point_count, 100_000)
                cg = CorridorGeometry((120.16, 31.19), (120.22, 31.19), 30.0)
                sel = ds.select(cg, ReadOptions(half_width_m=30.0))
                self.assertGreater(len(sel), 500)
                self.assertTrue(np.all(np.abs(sel.cross) <= 30.0 + 1e-6))
                # 沿轨迹方向应能找到湖面高程 ~3.6 m
                self.assertLess(abs(float(np.median(sel.elev)) - 3.6), 8.0)
            finally:
                ds.close()


@unittest.skipUnless(REAL_ATL03.exists(), "未找到真实 ATL03 文件, 跳过")
class TestRealATL03(unittest.TestCase):
    ds = None

    @classmethod
    def setUpClass(cls):
        t0 = time.time()
        cls.ds = open_dataset(REAL_ATL03)
        cls.t_open = time.time() - t0

    @classmethod
    def tearDownClass(cls):
        if cls.ds is not None:
            cls.ds.close()

    def test_index(self):
        self.assertEqual(self.ds.kind, "ATL03")
        self.assertEqual(len(self.ds.beams), 6)
        self.assertGreater(self.ds.point_count, 50_000_000)
        print(f"\n[ATL03] 打开耗时 {self.t_open:.2f}s, 光子总数 {self.ds.point_count:,}, "
              f"波束 {self.ds.beams}")

    def test_corridor_over_taihu(self):
        """在太湖附近取一条真实轨迹上的两点, 提取剖面."""
        tracks = self.ds.tracks()
        # 找到离太湖中心最近的轨迹点
        best = None
        for tr in tracks:
            d = np.hypot((tr["lon"] - 120.2) * 95.0, (tr["lat"] - 31.2) * 111.0)
            i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                best = (float(d[i]), tr["label"], float(tr["lon"][i]), float(tr["lat"][i]))
        _, label, lon0, lat0 = best
        d_km = geo.GEOD.distance(120.2, 31.2, lon0, lat0) / 1000.0
        print(f"[ATL03] 距太湖中心最近的轨迹点: {label} ({lon0:.5f}, {lat0:.5f}), "
              f"约 {d_km:.1f} km")
        cg = CorridorGeometry((lon0 - 0.05, lat0), (lon0 + 0.05, lat0), half_width_m=30.0)
        t0 = time.time()
        sel = self.ds.select(cg, ReadOptions(half_width_m=30.0))
        dt = time.time() - t0
        print(f"[ATL03] 廊道提取 {dt:.2f}s -> {len(sel):,} 光子, "
              f"线长 {cg.length_m:,.0f} m")
        print(sel.describe())
        self.assertGreater(len(sel), 1000)
        self.assertTrue(np.all(np.abs(sel.cross) <= 30.0 + 1e-6))
        self.assertGreaterEqual(float(sel.along.min()), -1e-6)
        self.assertLessEqual(float(sel.along.max()), cg.length_m + 1e-6)

    def test_confidence_filter(self):
        tracks = self.ds.tracks()
        tr = tracks[0]
        i = int(np.argmin(np.hypot(tr["lon"] - 120.2, tr["lat"] - 31.2)))
        lon0, lat0 = float(tr["lon"][i]), float(tr["lat"][i])
        cg = CorridorGeometry((lon0 - 0.02, lat0), (lon0 + 0.02, lat0), half_width_m=25.0)
        all_sel = self.ds.select(cg, ReadOptions(half_width_m=25.0))
        high_sel = self.ds.select(cg, ReadOptions(half_width_m=25.0, conf_mode="仅高置信"))
        print(f"[ATL03] 全部 {len(all_sel):,} -> 仅高置信 {len(high_sel):,}")
        self.assertLessEqual(len(high_sel), len(all_sel))

    def test_coverage_index(self):
        idx = self.ds.coverage_index()
        near = idx.nearest_distance_m(120.2, 31.2)
        far = idx.nearest_distance_m(100.0, 45.0)
        print(f"[ATL03] 覆盖判定: 太湖中心距轨迹 {near / 1000:.1f} km, 塔里木 {far / 1000:.0f} km")
        self.assertLess(near, far)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    unittest.main(verbosity=2)
