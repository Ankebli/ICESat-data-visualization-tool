# -*- coding: utf-8 -*-
"""模型接口与"一帧光子"数据结构.

设计原则
--------
* 模型只负责给每个光子一个 ``score ∈ [0, 1]`` (越大越像信号), **不做阈值/后处理**,
  阈值与形态学清理由框架统一做 —— 换模型时显示与统计口径保持一致。
* 模型不关心数据来自哪个产品、也不关心地图: 它只看到 :class:`PhotonFrame`。
* 任何模型异常都必须在框架层被捕获, 由界面提示并回退, 不允许影响主流程。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np


class ModelError(RuntimeError):
    """模型相关的异常基类."""


class ModelContractError(ModelError):
    """模型不满足接口约定 (如返回长度不符、缺少 predict)."""


@dataclass
class PhotonFrame:
    """一帧光子 (可能是一个沿迹分块), 供模型使用.

    所有数组长度一致; 与"哪个数据集/哪条剖切线"无关, 只保留模型可能用到的量。
    """

    along: np.ndarray                 # 沿剖切线距离 (m)
    cross: np.ndarray                 # 横向偏距 (m)
    elev: np.ndarray                  # 高程 (m, 幅面与当前高程基准一致)
    length_m: float                   # 剖切线长度 (m)
    half_width_m: float               # 廊道半宽 (m)
    beam: Optional[np.ndarray] = None          # 波束标签 (str)
    extras: Dict[str, np.ndarray] = field(default_factory=dict)  # signal_conf/quality 等

    def __len__(self) -> int:
        return int(self.elev.size)

    @property
    def signal_conf(self) -> Optional[np.ndarray]:
        """ATL03 ``signal_conf_ph`` (N×5, 通道: 陆地/海洋/海冰/陆地冰/内陆水体)."""
        conf = self.extras.get("signal_conf_ph")
        return None if conf is None else np.asarray(conf)

    @property
    def quality(self) -> Optional[np.ndarray]:
        q = self.extras.get("quality_ph")
        return None if q is None else np.asarray(q)


@dataclass
class PhotonModelResult:
    """模型输出.

    ``score``: 每个光子的信号得分 (0~1, 越大越像信号)
    ``label``: 可选布尔标签 (模型自带判定时给出; 否则由框架按阈值生成)
    """

    score: np.ndarray
    label: Optional[np.ndarray] = None
    name: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.score.size)


@runtime_checkable
class PhotonModel(Protocol):
    """自定义模型需要实现的协议.

    最小实现::

        class MyModel:
            name = "my-denoiser"
            version = "1.0"

            def predict(self, frame: PhotonFrame) -> np.ndarray:
                # 返回长度 == len(frame) 的 score 数组
                ...
    """

    name: str
    version: str

    def predict(self, frame: PhotonFrame) -> np.ndarray:  # pragma: no cover - 协议
        ...


# ----------------------------------------------------------------------------
# 模型调用与结果校验
# ----------------------------------------------------------------------------
def apply_model(model: PhotonModel, frame: PhotonFrame,
                threshold: float = 0.5) -> PhotonModelResult:
    """调用模型并校验输出, 统一生成 score/label.

    * 返回值长度必须等于光子数, 否则抛 :class:`ModelContractError`;
    * 允许模型返回 ``(N,)`` 数组或 :class:`PhotonModelResult`;
    * score 会裁剪到 [0, 1] 并把 NaN 视为 0。
    """
    if not hasattr(model, "predict"):
        raise ModelContractError(f"模型 {type(model).__name__} 缺少 predict() 方法")
    raw = model.predict(frame)
    label = None
    if isinstance(raw, PhotonModelResult):
        score, label = raw.score, raw.label
    else:
        score = raw
    score = np.asarray(score, dtype="f8").reshape(-1)
    if score.size != len(frame):
        raise ModelContractError(
            f"模型 {getattr(model, 'name', type(model).__name__)} 返回 {score.size} 个结果, "
            f"但输入有 {len(frame)} 个光子")
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    score = np.clip(score, 0.0, 1.0)
    if label is None:
        label = score >= float(threshold)
    else:
        label = np.asarray(label, dtype=bool).reshape(-1)
        if label.size != score.size:
            raise ModelContractError("模型返回的 label 长度与 score 不一致")
    return PhotonModelResult(score=score, label=label,
                             name=getattr(model, "name", type(model).__name__),
                             meta={"threshold": float(threshold)})


def model_from_selection(selection, elev_kind: str = "椭球高") -> PhotonFrame:
    """把 :class:`PhotonSelection` 转成 :class:`PhotonFrame` (供模型推理)."""
    return PhotonFrame(
        along=np.asarray(selection.along, "f8"),
        cross=np.asarray(selection.cross, "f8"),
        elev=np.asarray(selection.display_elev(elev_kind), "f8"),
        length_m=float(selection.corridor.length_m),
        half_width_m=float(selection.corridor.half_width_m),
        beam=np.asarray(selection.beam),
        extras=dict(selection.extras),
    )
