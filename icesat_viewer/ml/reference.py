# -*- coding: utf-8 -*-
"""官方参考模型: 直接使用 ATL03 产品自带的 ``signal_conf_ph``.

ATL03 的 ``signal_conf_ph`` 是 NASA 官方信号识别 (基于 ATL08 的 signal finding)
给出的逐光子置信度, 5 个通道分别对应

    0 陆地  1 海洋  2 海冰  3 陆地冰  4 内陆水体

取值 -2 ~ 4 (4 = 高置信)。它相当于一枚"官方已经调好、随产品免费发布"的基线,
本模块把它包装成统一的 :class:`~icesat_viewer.ml.base.PhotonModel`,
既可作为后续自定义模型的对照标签, 也可直接用于剖面去噪显示。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .base import PhotonFrame, PhotonModelResult

CONF_CHANNEL_NAMES: Dict[str, int] = {
    "陆地": 0, "海洋": 1, "海冰": 2, "陆地冰": 3, "内陆水体": 4,
}

#: 界面下拉框可选项: 显示名 -> (通道, 最低置信度)
REFERENCE_NONE = "无 (不评估)"
REFERENCE_CHOICES: Dict[str, tuple] = {
    "ATL03 高置信 (≥4)": ("自动", 4),
    "ATL03 中等及以上 (≥3)": ("自动", 3),
    "ATL03 高置信·陆地": ("陆地", 4),
    "ATL03 高置信·海洋": ("海洋", 4),
    "ATL03 高置信·内陆水体": ("内陆水体", 4),
}

#: 参考模型的显示方式 (界面下拉框)
REF_DISPLAY_MODES: Dict[str, str] = {
    "高亮参考信号": "highlight",   # 非信号淡化 (默认)
    "仅参考信号": "only",          # 只画信号
    "全部光子": "all",             # 不区分, 原样绘制
}


class SignalConfidenceModel:
    """把 ATL03 ``signal_conf_ph`` 包装成模型 (score = conf/4)."""

    name = "ATL03 signal_conf (官方)"
    version = "1.0"

    def __init__(self, channel: str = "自动", min_conf: int = 4):
        self.channel = channel if channel in CONF_CHANNEL_NAMES else "自动"
        self.min_conf = int(min_conf)

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "kind": "reference",
            "channel": self.channel,
            "min_conf": self.min_conf,
            "note": "直接使用 ATL03 产品自带的 signal_conf_ph, 无需训练",
        }

    def score_values(self, frame: PhotonFrame) -> Optional[np.ndarray]:
        """返回每个光子的原始置信度 (未归一化); 无该字段时返回 None."""
        conf = frame.signal_conf
        if conf is None or conf.size == 0:
            return None
        conf = np.asarray(conf)
        if conf.ndim == 1:
            conf = conf.reshape(-1, 1)
        if self.channel in CONF_CHANNEL_NAMES:
            col = CONF_CHANNEL_NAMES[self.channel]
            raw = conf[:, col] if col < conf.shape[1] else conf.max(axis=1)
        else:                      # 自动: 取 5 个通道最大值
            raw = conf.max(axis=1)
        return raw.astype("f8")

    def predict(self, frame: PhotonFrame) -> np.ndarray:
        raw = self.score_values(frame)
        if raw is None:
            # 该产品没有 signal_conf: 全部判为未知(0.5), 由界面提示不可评估
            return np.full(len(frame), 0.5, dtype="f8")
        return np.clip(raw / 4.0, 0.0, 1.0)

    # ------------------------------------------------------------------
    def label(self, frame: PhotonFrame) -> Optional[np.ndarray]:
        """直接给出布尔标签 (conf >= min_conf); 无字段返回 None."""
        raw = self.score_values(frame)
        if raw is None:
            return None
        return raw >= float(self.min_conf)

    def result(self, frame: PhotonFrame) -> Optional[PhotonModelResult]:
        """便捷方法: 生成带 label 的结果."""
        label = self.label(frame)
        if label is None:
            return None
        raw = self.score_values(frame)
        thr = float(self.min_conf) / 4.0
        return PhotonModelResult(
            score=np.clip(raw / 4.0, 0.0, 1.0), label=label, name=self.name,
            meta={"channel": self.channel, "min_conf": self.min_conf, "threshold": thr})


def reference_model_for(choice: str) -> Optional[SignalConfidenceModel]:
    """根据界面选项构造参考模型; "无" 或未知选项返回 None."""
    if not choice or choice == REFERENCE_NONE or choice not in REFERENCE_CHOICES:
        return None
    channel, min_conf = REFERENCE_CHOICES[choice]
    return SignalConfidenceModel(channel=channel, min_conf=min_conf)
