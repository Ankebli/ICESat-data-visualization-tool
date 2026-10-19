# -*- coding: utf-8 -*-
"""DEM 剖切对比 (M0.5 新功能) 测试.

覆盖: terrarium 解码 / 本地 GeoTIFF 采样 / 沿剖切线采样 / 垂直基准换算 /
      对比统计 / 网络源可用性(可跳过)。

运行::

    C:\\Users\\ASUS\\miniconda3\\envs\\waterenv\\python.exe tests\\test_dem.py
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from icesat_viewer import geo                                          # noqa: E402
from icesat_viewer.corridor import CorridorGeometry                   # noqa: E402
from icesat_viewer.dem import (DEM_DATUMS, DemError, RasterDem,       # noqa: E402
                               TerrariumDem, _decode_terrarium, _lat_to_tiley,
                               adjust_dem_to_elev_kind, compare_with_profile,
                               corridor_bounds, open_dem_file,
                               sample_along_corridor)
from icesat_viewer.icesat_io import PhotonSelection                    # noqa: E402


def _has_rasterio() -> bool:
    try:
        import rasterio  # noqa: F401
        return True
    except Exception:
        return False


def _write_test_tiff(path: Path, west=120.0, north=31.1, res=0.002, n=60) -> None:
    """写一个已知线性地形的 GeoTIFF: 高程 = 100 + 1000*(north - lat) + 500*(lon - west)."""
    import rasterio
    from rasterio.transform import from_origin

    xs = west + (np.arange(n) + 0.5) * res
    ys = north - (np.arange(n) + 0.5) * res
    lon, lat = np.meshgrid(xs, ys)
    z = 100.0 + 1000.0 * (north - lat) + 500.0 * (lon - west)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_origin(west, north, res, res)) as dst:
        dst.write(z.astype("float32"), 1)


class TestTerrariumDecode(unittest.TestCase):
    def test_decode(self):
        # 高程 0 m -> (128, 0, 0); 5 m -> (128, 5, 0); -10 m -> (127, 246, 0)
        rgb = np.array([[[128, 0, 0], [128, 5, 0], [127, 246, 0]]], dtype="uint8")
        z = _decode_terrarium(rgb)
        self.assertAlmostEqual(float(z[0, 0]), 0.0, places=3)
        self.assertAlmostEqual(float(z[0, 1]), 5.0, places=3)
        self.assertAlmostEqual(float(z[0, 2]), -10.0, places=3)

    def test_tiley(self):
        # z=0 时整张瓦片覆盖全球, 赤道处行号应约为 0.5
        self.assertAlmostEqual(float(_lat_to_tiley(0.0, 0)), 0.5, places=3)
        self.assertLess(float(_lat_to_tiley(60.0, 0)), 0.5)


@unittest.skipUnless(_has_rasterio(), "未安装 rasterio, 跳过本地 DEM 测试")
class TestRasterDem(unittest.TestCase):
    def test_sample_linear_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dem.tif"
            _write_test_tiff(path)
            dem = RasterDem(path)
            lon = np.array([120.02, 120.05, 120.08])
            lat = np.array([31.08, 31.05, 31.02])
            z = dem.sample(lon, lat)
            expect = 100.0 + 1000.0 * (31.1 - lat) + 500.0 * (lon - 120.0)
            # 双线性 + 半像元偏移, 允许 1 m 误差
            self.assertTrue(np.allclose(z, expect, atol=1.5), (z, expect))
            self.assertEqual(dem.bounds[0], 120.0)
            self.assertIn("正交高", dem.datum)
            self.assertGreater(dem.resolution_m, 100.0)

    def test_nodata_and_outside(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dem.tif"
            _write_test_tiff(path)
            dem = RasterDem(path)
            z = dem.sample(np.array([150.0]), np.array([10.0]))
            self.assertTrue(np.isnan(z[0]))

    def test_open_dem_file_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dem.tif"
            _write_test_tiff(path)
            dem = open_dem_file(path)
            self.assertIsInstance(dem, RasterDem)

    def test_missing_file(self):
        with self.assertRaises(DemError):
            open_dem_file(Path(tempfile.gettempdir()) / "no_such_dem.tif")


class TestCorridorSampling(unittest.TestCase):
    def test_sample_along(self):
        class FlatDem:
            datum = DEM_DATUMS[1]
            bounds = None
            resolution_m = 30.0
            name = "flat"

            def sample(self, lon, lat):
                return np.full(len(lon), 42.0)

            def covers(self, *a):
                return True

            @property
            def is_orthometric(self):
                return False

        cg = CorridorGeometry((120.0, 31.0), (120.0, 31.05), 25.0)
        along, z = sample_along_corridor(FlatDem(), cg, step_m=10.0)
        self.assertGreater(along.size, 100)
        self.assertAlmostEqual(float(along[0]), 0.0, places=6)
        self.assertAlmostEqual(float(along[-1]), cg.length_m, places=3)
        self.assertTrue(np.allclose(z, 42.0))
        # 步长应接近 10 m
        self.assertAlmostEqual(float(np.median(np.diff(along))), 10.0, delta=1.0)

    def test_corridor_bounds(self):
        cg = CorridorGeometry((120.0, 31.0), (120.02, 31.02), 25.0)
        b = corridor_bounds(cg, half_width_m=25.0, margin_m=200.0)
        self.assertLess(b[0], 120.0)
        self.assertGreater(b[1], 120.02)
        self.assertLess(b[2], 31.0)
        self.assertGreater(b[3], 31.02)


class _Sel:
    """最小 selection 替身 (只需要 along/geoid)."""

    def __init__(self, along, geoid=None):
        self.along = np.asarray(along, "f8")
        self.geoid = None if geoid is None else np.asarray(geoid, "f8")


class TestDatumConversion(unittest.TestCase):
    def test_orthometric_dem_to_ellipsoidal_profile(self):
        along = np.array([0.0, 500.0, 1000.0])
        z = np.array([3.0, 3.0, 3.0])
        sel = _Sel([0.0, 1000.0], [7.13, 7.13])
        out, note = adjust_dem_to_elev_kind(along, z, sel, "椭球高", True)
        self.assertTrue(np.allclose(out, 10.13, atol=1e-6), out)
        self.assertIn("geoid", note)

    def test_orthometric_dem_with_ortho_profile(self):
        along = np.array([0.0, 1000.0])
        z = np.array([3.0, 3.0])
        sel = _Sel([0.0, 1000.0], [7.13, 7.13])
        out, _ = adjust_dem_to_elev_kind(along, z, sel, "正交高(减geoid)", True)
        self.assertTrue(np.allclose(out, 3.0))

    def test_ellipsoidal_dem_to_ortho_profile(self):
        along = np.array([0.0, 1000.0])
        z = np.array([10.13, 10.13])
        sel = _Sel([0.0, 1000.0], [7.13, 7.13])
        out, _ = adjust_dem_to_elev_kind(along, z, sel, "正交高(减geoid)", False)
        self.assertTrue(np.allclose(out, 3.0, atol=1e-6))

    def test_relative_and_offset(self):
        along = np.array([0.0, 500.0, 1000.0])
        z = np.array([1.0, 3.0, 5.0])
        sel = _Sel([0.0, 1000.0], [0.0, 0.0])
        out, note = adjust_dem_to_elev_kind(along, z, sel, "相对高(减中值)", True)
        self.assertAlmostEqual(float(np.median(out)), 0.0, places=6)
        out2, _ = adjust_dem_to_elev_kind(along, z, sel, "椭球高", True, offset_m=1.5)
        self.assertTrue(np.allclose(out2, z + 1.5), out2)   # geoid=0, 只加偏移

    def test_no_geoid_no_conversion(self):
        along = np.array([0.0, 1000.0])
        z = np.array([3.0, 3.0])
        sel = _Sel([0.0, 1000.0], None)
        out, note = adjust_dem_to_elev_kind(along, z, sel, "椭球高", True)
        self.assertTrue(np.allclose(out, 3.0))
        self.assertIn("缺少 geoid", note)


class TestComparison(unittest.TestCase):
    def test_flat_offset(self):
        along = np.linspace(0, 1000, 2000)
        elev = np.full(along.size, 10.0) + np.random.default_rng(0).normal(0, 0.02, along.size)
        dem_along = np.linspace(0, 1000, 500)
        dem_z = np.full(dem_along.size, 9.5)
        cmp = compare_with_profile(along, elev, dem_along, dem_z, bin_size_m=20.0)
        self.assertGreater(cmp.n_points, 10)
        self.assertAlmostEqual(cmp.mean_m, -0.5, delta=0.05)
        self.assertAlmostEqual(cmp.rmse_m, 0.5, delta=0.05)
        self.assertIn("DEM 对比", cmp.summary())
        self.assertGreater(cmp.dem_flat_ratio, 0.9)      # 常数 DEM -> 提示 SRTM 水面
        self.assertTrue(cmp.note)

    def test_empty_inputs(self):
        cmp = compare_with_profile(np.array([]), np.array([]), np.array([]), np.array([]))
        self.assertEqual(cmp.n_points, 0)
        self.assertEqual(cmp.summary(), "DEM 在该范围无有效值")
        # DEM 全 NaN
        along = np.linspace(0, 100, 50)
        cmp2 = compare_with_profile(along, np.full(50, 5.0),
                                   np.linspace(0, 100, 20), np.full(20, np.nan))
        self.assertEqual(cmp2.n_points, 0)


@unittest.skipUnless(_has_rasterio(), "未安装 rasterio, 跳过网络 DEM 测试")
class TestTerrariumNetwork(unittest.TestCase):
    """AWS Terrain Tiles 免密钥可用性 (网络不可用时自动跳过)."""

    def test_fetch_taihu_land(self):
        bounds = (120.40, 120.46, 31.22, 31.28)      # 苏州穹窿山一带 (确实为陆地)
        try:
            dem = TerrariumDem(bounds, cache_dir=Path(tempfile.gettempdir()) / "dem_test")
        except DemError as exc:
            self.skipTest(f"DEM 瓦片不可用 (网络?): {exc}")
        lon = np.linspace(bounds[0], bounds[1], 20)
        lat = np.linspace(bounds[2], bounds[3], 20)
        z = dem.sample(lon, lat)
        valid = z[np.isfinite(z)]
        self.assertGreater(valid.size, 10)
        print(f"\n[DEM] 丘陵区采样: min {valid.min():.1f} / max {valid.max():.1f} m")
        self.assertGreater(float(valid.max()), 20.0)
        self.assertLess(float(valid.max()), 2000.0)
        self.assertEqual(dem.datum, DEM_DATUMS[0])
        self.assertGreater(dem.resolution_m or 0, 1.0)

    def test_fetch_water_is_flat(self):
        """SRTM 类产品在水面按 0 填充 -> 应几乎全为 0 (文档化该行为)."""
        bounds = (120.18, 120.20, 31.18, 31.20)
        try:
            dem = TerrariumDem(bounds, cache_dir=Path(tempfile.gettempdir()) / "dem_test")
        except DemError as exc:
            self.skipTest(f"DEM 瓦片不可用 (网络?): {exc}")
        z = dem.sample(np.linspace(bounds[0], bounds[1], 30),
                       np.linspace(bounds[2], bounds[3], 30))
        valid = z[np.isfinite(z)]
        if valid.size == 0:
            self.skipTest("该范围无 DEM 值")
        water_ratio = float(np.mean(np.abs(valid) < 0.5))
        print(f"\n[DEM] 太湖水面: {water_ratio * 100:.0f}% 采样为 0 m "
              f"(min {valid.min():.1f} / max {valid.max():.1f})")
        self.assertGreater(water_ratio, 0.5)


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    unittest.main(verbosity=2)
