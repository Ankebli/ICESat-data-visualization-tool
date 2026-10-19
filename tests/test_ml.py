# -*- coding: utf-8 -*-
"""M0: 官方向导模型 (ATL03 signal_conf) 与评估指标测试.

运行::

    C:\\Users\\ASUS\\miniconda3\\envs\\waterenv\\python.exe tests\\test_ml.py

不依赖界面; 真实 ATL03 用例在文件缺失时自动跳过。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from icesat_viewer.corridor import CorridorGeometry                      # noqa: E402
from icesat_viewer.icesat_io import PhotonSelection, ReadOptions, open_dataset  # noqa: E402
from icesat_viewer.ml import (REFERENCE_CHOICES, REFERENCE_NONE,         # noqa: E402
                              ModelMetrics, SignalConfidenceModel, evaluate,
                              model_from_selection, reference_model_for)
from icesat_viewer.ml.base import (ModelContractError, PhotonFrame,      # noqa: E402
                                   apply_model)
from icesat_viewer.ml.metrics import surface_residual_rmse               # noqa: E402

REAL_ATL03 = Path(r"D:\Paper\Data\satellite image\ICESAT\Taihu"
                  r"\ATL03_20250105065215_03082606_007_01.h5")


def _frame(n=6, conf=None, quality=None):
    along = np.linspace(0.0, 1000.0, n)
    return PhotonFrame(along=along, cross=np.zeros(n), elev=np.linspace(5, 6, n),
                       length_m=1000.0, half_width_m=25.0,
                       beam=np.array(["gt1l"] * n),
                       extras={} if conf is None else {
                           "signal_conf_ph": np.asarray(conf),
                           **({} if quality is None else {"quality_ph": np.asarray(quality)})})


class TestMetrics(unittest.TestCase):
    def test_confusion_and_scores(self):
        pred = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
        truth = np.array([1, 0, 0, 1, 1, 0], dtype=bool)
        m = evaluate(pred, truth)
        self.assertTrue(m.available)
        self.assertEqual((m.true_positive, m.false_positive, m.false_negative, m.true_negative),
                         (2, 1, 1, 2))
        self.assertAlmostEqual(m.precision, 2 / 3, places=6)
        self.assertAlmostEqual(m.recall, 2 / 3, places=6)
        self.assertAlmostEqual(m.f1, 2 / 3, places=6)
        self.assertAlmostEqual(m.accuracy, 4 / 6, places=6)
        self.assertAlmostEqual(m.keep_ratio, 3 / 6, places=6)
        self.assertIn("P 0.667", m.summary())
        self.assertIn("命中 TP", m.table())

    def test_degenerate(self):
        self.assertFalse(evaluate(None, np.array([True])).available)
        self.assertFalse(evaluate(np.array([True]), None).available)
        self.assertFalse(evaluate(np.array([True]), np.array([True, False])).available)
        self.assertFalse(evaluate(np.array([], bool), np.array([], bool)).available)
        empty = ModelMetrics(available=False, note="不可评估")
        self.assertEqual(empty.summary(), "不可评估")
        self.assertTrue(np.isnan(empty.precision))

    def test_surface_rmse(self):
        rng = np.random.default_rng(0)
        along = np.repeat(np.arange(0, 2000, 5.0), 4)
        elev = np.full(along.size, 10.0) + rng.normal(0, 0.05, along.size)
        rmse_clean = surface_residual_rmse(along, elev, bin_size_m=50.0)
        noisy = elev + rng.normal(0, 3.0, elev.size)
        rmse_noisy = surface_residual_rmse(along, noisy, bin_size_m=50.0)
        self.assertLess(rmse_clean, 0.2)
        self.assertGreater(rmse_noisy, rmse_clean)
        self.assertTrue(np.isnan(surface_residual_rmse(along[:5], elev[:5])))


class TestReferenceModel(unittest.TestCase):
    def test_score_and_label(self):
        conf = np.array([[4, 4, 4, 4, 4],
                         [0, 0, 0, 0, 0],
                         [2, 2, 2, 2, 2],
                         [-2, -2, -2, -2, -2],
                         [1, 1, 1, 1, 4],
                         [3, 3, 3, 3, 3]], dtype="i1")
        frame = _frame(conf=conf)
        model = SignalConfidenceModel()          # 自动通道 + >=4
        score = model.predict(frame)
        self.assertEqual(score.shape, (6,))
        self.assertAlmostEqual(float(score[0]), 1.0, places=6)
        self.assertAlmostEqual(float(score[1]), 0.0, places=6)
        label = model.label(frame)
        self.assertEqual(label.tolist(), [True, False, False, False, True, False])
        # 指定通道: 内陆水体 (索引 4) 才有 4 -> 第 5 个光子
        model_ch = SignalConfidenceModel(channel="内陆水体", min_conf=4)
        self.assertEqual(model_ch.label(frame).tolist(),
                         [True, False, False, False, True, False])
        # 阈值降到 3
        model_med = SignalConfidenceModel(min_conf=3)
        self.assertEqual(model_med.label(frame).tolist(),
                         [True, False, False, False, True, True])
        result = model.result(frame)
        self.assertEqual(result.name, model.name)
        self.assertAlmostEqual(result.meta["threshold"], 1.0, places=6)

    def test_missing_conf(self):
        frame = _frame(conf=None)
        model = SignalConfidenceModel()
        self.assertTrue(np.allclose(model.predict(frame), 0.5))
        self.assertIsNone(model.label(frame))
        self.assertIsNone(model.result(frame))
        self.assertEqual(model.describe()["kind"], "reference")

    def test_choices(self):
        self.assertIsNone(reference_model_for(REFERENCE_NONE))
        self.assertIsNone(reference_model_for("不存在的选项"))
        for name in REFERENCE_CHOICES:
            model = reference_model_for(name)
            self.assertIsNotNone(model)
            self.assertIn(model.min_conf, (3, 4))


class TestApplyModelContract(unittest.TestCase):
    class _Good:
        name = "good"
        version = "1"

        def predict(self, frame):
            return np.full(len(frame), 0.8)

    class _Short:
        name = "short"
        version = "1"

        def predict(self, frame):
            return np.zeros(3)

    class _Raises:
        name = "boom"
        version = "1"

        def predict(self, frame):
            raise RuntimeError("模型内部错误")

    class _NoPredict:
        name = "nopredict"
        version = "1"

    def test_good_and_threshold(self):
        frame = _frame()
        result = apply_model(self._Good(), frame, threshold=0.5)
        self.assertEqual(len(result), len(frame))
        self.assertTrue(result.label.all())
        result2 = apply_model(self._Good(), frame, threshold=0.9)
        self.assertFalse(result2.label.any())

    def test_nan_and_clipping(self):
        class Weird:
            name = "weird"
            version = "1"

            def predict(self, frame):
                out = np.array([np.nan, 2.0, -3.0, 0.5, np.inf, -np.inf])
                return out

        frame = _frame(n=6)
        result = apply_model(Weird(), frame, threshold=0.5)
        self.assertTrue(np.all(result.score >= 0) and np.all(result.score <= 1))
        self.assertAlmostEqual(float(result.score[0]), 0.0)
        self.assertAlmostEqual(float(result.score[1]), 1.0)

    def test_contract_errors(self):
        frame = _frame()
        with self.assertRaises(ModelContractError):
            apply_model(self._Short(), frame)
        with self.assertRaises(ModelContractError):
            apply_model(self._NoPredict(), frame)
        with self.assertRaises(RuntimeError):        # 模型自身异常应向上抛, 由界面捕获
            apply_model(self._Raises(), frame)


class TestSelectionExtras(unittest.TestCase):
    """PhotonSelection 必须带出 signal_conf_ph (M0 的基础设施)."""

    def test_extras_from_synthetic_selection(self):
        n = 500
        rng = np.random.default_rng(1)
        along = np.linspace(0, 500, n)
        conf = rng.integers(-2, 5, size=(n, 5)).astype("i1")
        cg = CorridorGeometry((120.0, 31.0), (120.0, 31.01), 25.0)
        sel = PhotonSelection(lon=np.full(n, 120.0), lat=np.linspace(31, 31.01, n),
                              elev=np.full(n, 5.0), along=along, cross=np.zeros(n),
                              beam=np.full(n, "gt1l"), corridor=cg,
                              extras={"signal_conf_ph": conf})
        frame = model_from_selection(sel, "椭球高")
        self.assertEqual(frame.signal_conf.shape, (n, 5))
        self.assertEqual(len(frame), n)
        model = SignalConfidenceModel(min_conf=4)
        mask = model.label(frame)
        self.assertEqual(int(mask.sum()), int((conf.max(axis=1) >= 4).sum()))
        # 与 evaluate 联用
        metrics = evaluate(mask, mask)
        self.assertAlmostEqual(metrics.precision, 1.0)
        self.assertAlmostEqual(metrics.recall, 1.0)


@unittest.skipUnless(REAL_ATL03.exists(), "未找到真实 ATL03 文件, 跳过")
class TestRealATL03Reference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = open_dataset(REAL_ATL03)
        tracks = cls.ds.tracks()
        best = None
        for tr in tracks:
            d = np.hypot((tr["lon"] - 120.2) * 95.0, (tr["lat"] - 31.2) * 111.0)
            i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                best = (float(d[i]), float(tr["lon"][i]), float(tr["lat"][i]))
        _, lon0, lat0 = best
        cls.cg = CorridorGeometry((lon0 - 0.04, lat0), (lon0 + 0.04, lat0), 25.0)

    @classmethod
    def tearDownClass(cls):
        cls.ds.close()

    def test_extras_present_and_consistent(self):
        sel_all = self.ds.select(self.cg, ReadOptions(half_width_m=25.0, conf_mode="全部"))
        self.assertIn("signal_conf_ph", sel_all.extras)
        conf = sel_all.extras["signal_conf_ph"]
        self.assertEqual(conf.shape, (len(sel_all), 5))
        self.assertIn("quality_ph", sel_all.extras)

        # 参考模型(>=4) 的数量应等于"仅高置信"筛选出的光子数
        sel_high = self.ds.select(self.cg, ReadOptions(half_width_m=25.0,
                                                      conf_mode="仅高置信"))
        model = SignalConfidenceModel(min_conf=4)
        mask = model.label(model_from_selection(sel_all))
        print(f"\n[ML] 全部 {len(sel_all):,} 光子, 参考高置信 {int(mask.sum()):,}, "
              f"仅高置信筛选 {len(sel_high):,}")
        self.assertEqual(int(mask.sum()), len(sel_high))

    def test_afterpulse_filter_matches_quality(self):
        sel_all = self.ds.select(self.cg, ReadOptions(half_width_m=25.0, conf_mode="全部"))
        sel_q = self.ds.select(self.cg, ReadOptions(half_width_m=25.0, conf_mode="全部",
                                                    remove_afterpulse=True))
        q = sel_all.extras["quality_ph"]
        self.assertEqual(int((q == 0).sum()), len(sel_q))


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    unittest.main(verbosity=2)
