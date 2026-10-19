# -*- coding: utf-8 -*-
"""模型评估指标: 混淆矩阵 / P-R-F1 / 保留率 / 表面残差.

约定
----
* ``truth`` 或 ``pred`` 为 None 时, ``available=False``, 界面显示"不可评估"
  (例如 CSV/LAS 数据没有 ``signal_conf``);
* 所有函数对空数组安全。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ModelMetrics:
    """二分类指标 (正类 = 信号)."""

    available: bool = False
    n_total: int = 0
    n_truth_signal: int = 0
    n_pred_signal: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    note: str = ""
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return float(self.true_positive) / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return float(self.true_positive) / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not np.isfinite(p) or not np.isfinite(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float:
        if not self.n_total:
            return float("nan")
        return float(self.true_positive + self.true_negative) / float(self.n_total)

    @property
    def keep_ratio(self) -> float:
        """当前预测保留的光子占全部光子的比例."""
        if not self.n_total:
            return float("nan")
        return float(self.n_pred_signal) / float(self.n_total)

    # ------------------------------------------------------------------
    def summary(self, digits: int = 3) -> str:
        if not self.available:
            return self.note or "不可评估"
        return (f"P {self.precision:.{digits}f} / R {self.recall:.{digits}f} / "
                f"F1 {self.f1:.{digits}f} · 保留 {self.keep_ratio * 100:.1f}% "
                f"({self.n_pred_signal:,}/{self.n_total:,})")

    def table(self) -> str:
        if not self.available:
            return self.note or "不可评估"
        return "\n".join([
            f"样本总数      : {self.n_total:,}",
            f"参考信号数    : {self.n_truth_signal:,}",
            f"预测保留数    : {self.n_pred_signal:,}  (保留率 {self.keep_ratio * 100:.2f}%)",
            f"命中 TP / FP / FN / TN : {self.true_positive:,} / {self.false_positive:,} / "
            f"{self.false_negative:,} / {self.true_negative:,}",
            f"精确率 P      : {self.precision:.4f}",
            f"召回率 R      : {self.recall:.4f}",
            f"F1            : {self.f1:.4f}",
            f"总体准确率    : {self.accuracy:.4f}",
        ])


def evaluate(pred: Optional[np.ndarray], truth: Optional[np.ndarray],
             note: str = "") -> ModelMetrics:
    """比较预测标签与参考(真值)标签."""
    if pred is None or truth is None:
        return ModelMetrics(available=False, note=note or "缺少参考置信度, 不可评估")
    pred = np.asarray(pred, dtype=bool).reshape(-1)
    truth = np.asarray(truth, dtype=bool).reshape(-1)
    if pred.size != truth.size:
        return ModelMetrics(available=False,
                            note=f"长度不一致 (预测 {pred.size} / 参考 {truth.size})")
    n = int(pred.size)
    if n == 0:
        return ModelMetrics(available=False, note="该剖面没有光子")
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    return ModelMetrics(
        available=True, n_total=n, n_truth_signal=int(np.count_nonzero(truth)),
        n_pred_signal=int(np.count_nonzero(pred)),
        true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn,
        note=note)


def surface_residual_rmse(along: np.ndarray, elev: np.ndarray,
                          mask: Optional[np.ndarray] = None,
                          bin_size_m: float = 20.0) -> float:
    """去噪后表面残差 RMSE (与沿迹分箱中值线比较), 用于衡量"表面是否更干净".

    仅统计每个沿迹分箱内落点与中值线的偏差, 空箱忽略。
    """
    along = np.asarray(along, "f8")
    elev = np.asarray(elev, "f8")
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        along, elev = along[m], elev[m]
    ok = np.isfinite(along) & np.isfinite(elev)
    along, elev = along[ok], elev[ok]
    if along.size < 10:
        return float("nan")
    bin_size = max(float(bin_size_m), 0.5)
    idx = np.floor((along - along.min()) / bin_size).astype(np.int64)
    resid = np.empty_like(elev)
    for b in np.unique(idx):
        sel = idx == b
        if np.count_nonzero(sel) < 3:
            resid[sel] = np.nan
            continue
        resid[sel] = elev[sel] - np.median(elev[sel])
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(resid ** 2)))
