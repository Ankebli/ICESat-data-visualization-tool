# -*- coding: utf-8 -*-
"""面向光子去噪的机器学习/后处理层 (M0 最小骨架).

当前只实现 **官方参考模型**: 直接使用 ATL03 产品自带的 ``signal_conf_ph``
(以及 ``quality_ph``) 作为"已训练好的基线"，用来:

1. 作为"信号/噪声"的对照标签, 评估当前筛选或其他模型的表现 (P/R/F1);
2. 让后续自定义模型走同一套接口 (:class:`PhotonModel`), 直接接入即可。

后续里程碑 (M1+) 计划在本包内增加:
    features.py   邻域统计特征工程
    builtin.py    内置基线 (分位数带 / 网格连通域 / YAPC 移植 / BDT-ADBSCAN 移植)
    adapters.py   sklearn / xgboost / lightgbm / onnx / 用户脚本适配
    registry.py   模型发现、参数校验、结果缓存
"""

from __future__ import annotations

from .base import PhotonFrame, PhotonModelResult, apply_model, model_from_selection
from .metrics import ModelMetrics, evaluate
from .reference import (
    REFERENCE_CHOICES,
    REFERENCE_NONE,
    SignalConfidenceModel,
    reference_model_for,
)

__all__ = [
    "PhotonFrame", "PhotonModelResult", "apply_model", "model_from_selection",
    "ModelMetrics", "evaluate",
    "SignalConfidenceModel", "reference_model_for", "REFERENCE_CHOICES", "REFERENCE_NONE",
]
