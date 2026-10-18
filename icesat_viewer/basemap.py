# -*- coding: utf-8 -*-
"""瓦片底图引擎.

支持:
    * 天地图 (tianditu.gov.cn) WMTS/DataServer 瓦片: 矢量 vec_w / 影像 img_w / 地形 ter_w
      以及对应注记层 cva_w / cia_w / cta_w (半透明叠加);
    * 免密钥备用底图 (Esri World Imagery/Topo/Ocean) —— 未配置 tk 时仍可正常看图;
    * 内存 LRU + 磁盘缓存 + 线程池并发下载, 失败瓦片用占位块填充并记录原因.

所有瓦片均为 Web Mercator (EPSG:3857) 256x256 PNG/JPEG, 与 ``geo`` 模块的投影一致.
"""

from __future__ import annotations

import io
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:  # Pillow 用于拼接与注记叠加
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from . import geo

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

#: 天地图 tk 密钥格式: 32 位十六进制
TIANDITU_TOKEN_PATTERN = re.compile(r"^[0-9A-Fa-f]{32}$")


# ----------------------------------------------------------------------------
# 异常
# ----------------------------------------------------------------------------
class BasemapError(RuntimeError):
    """底图相关异常基类."""


class TiandituTokenError(BasemapError):
    """天地图 tk 密钥缺失 / 格式错误 / 被服务端拒绝 (HTTP 401/403/418)."""


class TileDownloadError(BasemapError):
    """瓦片下载失败 (网络异常、服务不可用等)."""


def is_valid_token_format(token: str) -> bool:
    """天地图密钥格式校验 (32 位十六进制)."""
    return bool(TIANDITU_TOKEN_PATTERN.match((token or "").strip()))


def mask_token(token: str) -> str:
    """日志/提示中脱敏显示密钥."""
    token = (token or "").strip()
    if len(token) <= 10:
        return token or "(空)"
    return f"{token[:6]}…{token[-4:]}"


# ----------------------------------------------------------------------------
# 数据源定义
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TileSource:
    """单个瓦片图层."""

    key: str                       # 图层标识, 同时用于缓存目录
    name: str                      # 中文名
    url: str                       # 模板: {s} 子域 {x} {y} {z} {token}
    subdomains: Tuple[str, ...] = ("",)
    max_zoom: int = 18
    requires_token: bool = False
    attribution: str = ""

    def format_url(self, z: int, x: int, y: int, token: str = "") -> str:
        sub = self.subdomains[(int(x) + int(y)) % len(self.subdomains)]
        return (self.url.replace("{s}", str(sub))
                .replace("{x}", str(int(x)))
                .replace("{y}", str(int(y)))
                .replace("{z}", str(int(z)))
                .replace("{token}", token or ""))


@dataclass(frozen=True)
class Provider:
    """底图方案 = 底图图层 (+ 可选注记图层)."""

    key: str
    name: str
    base: TileSource
    label: Optional[TileSource] = None
    group: str = "天地图"
    note: str = ""

    @property
    def requires_token(self) -> bool:
        return self.base.requires_token or bool(self.label and self.label.requires_token)

    @property
    def max_zoom(self) -> int:
        return min(self.base.max_zoom, self.label.max_zoom if self.label else 99)


def _tdt_source(layer: str, name: str, max_zoom: int = 18) -> TileSource:
    """天地图 DataServer 瓦片地址 (tk 为必填密钥)."""
    url = ("https://t{s}.tianditu.gov.cn/DataServer?"
           f"T={layer}&x={{x}}&y={{y}}&l={{z}}&tk={{token}}")
    return TileSource(key=f"tdt_{layer}", name=name, url=url,
                      subdomains=tuple(str(i) for i in range(8)),
                      max_zoom=max_zoom, requires_token=True,
                      attribution="© 天地图 (tianditu.gov.cn)")


def _esri_source(service: str, name: str, max_zoom: int = 19) -> TileSource:
    url = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
           f"{service}/MapServer/tile/{{z}}/{{y}}/{{x}}")
    return TileSource(key="esri_" + service.split("/")[-1].lower(), name=name, url=url,
                      subdomains=("",), max_zoom=max_zoom,
                      attribution="© Esri, Maxar, Earthstar Geographics")


PROVIDERS: "OrderedDict[str, Provider]" = OrderedDict()


def _register(provider: Provider) -> None:
    PROVIDERS[provider.key] = provider


# ---- 天地图 (需要 tk 密钥) --------------------------------------------------
_register(Provider("tianditu_img", "天地图·影像+注记",
                   _tdt_source("img_w", "影像底图"), _tdt_source("cia_w", "影像注记"),
                   note="需要 tk 密钥; CGCS2000/WGS84, 与 ICESat 数据可直接套合"))
_register(Provider("tianditu_vec", "天地图·矢量+注记",
                   _tdt_source("vec_w", "矢量底图"), _tdt_source("cva_w", "矢量注记"),
                   note="需要 tk 密钥; CGCS2000/WGS84"))
_register(Provider("tianditu_ter", "天地图·地形+注记",
                   _tdt_source("ter_w", "地形底图"), _tdt_source("cta_w", "地形注记"),
                   note="需要 tk 密钥"))

# ---- 免密钥备用底图 ---------------------------------------------------------
_register(Provider("esri_img", "Esri 影像 (免密钥)",
                   _esri_source("World_Imagery", "World Imagery"),
                   group="免密钥备用"))
_register(Provider("esri_topo", "Esri 地形 (免密钥)",
                   _esri_source("World_Topo_Map", "World Topo Map"),
                   group="免密钥备用"))


def provider_choices(token: str) -> List[Tuple[str, str, bool, str]]:
    """返回 [(key, 显示名, 是否可用, 原因)] 供界面下拉框使用."""
    out = []
    for key, prov in PROVIDERS.items():
        usable, reason = True, ""
        if prov.requires_token and len(token or "") < 8:
            usable, reason = False, "未配置天地图 tk 密钥"
        out.append((key, f"{prov.name}", usable, reason))
    return out


# ----------------------------------------------------------------------------
# 瓦片引擎
# ----------------------------------------------------------------------------
class TileEngine:
    """瓦片下载 / 缓存 / 拼接.

    线程安全: 内部使用 ``ThreadPoolExecutor`` 抓取瓦片, 拼接在调用线程完成;
    内存缓存用 ``OrderedDict`` 做 LRU.
    """

    def __init__(self, cache_dir: Path, token: str = "", timeout: float = 12.0,
                 workers: int = 8, memory_tiles: int = 900):
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self._token = token or ""
        self._memory: "OrderedDict[Tuple[str, int, int, int], np.ndarray]" = OrderedDict()
        self._memory_limit = int(memory_tiles)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=workers,
                                            thread_name_prefix="tile")
        self._session = None
        self._blocked: Dict[str, Tuple[float, str]] = {}   # source.key -> (解除时间, 原因)
        self._fail_streak: Dict[str, int] = {}
        self._auth_error: Optional[TiandituTokenError] = None   # 密钥被拒后记住, 快速失败
        self.stats = {"hit_mem": 0, "hit_disk": 0, "download": 0, "fail": 0}
        self.last_error = ""

    # ------------------------------------------------------------------
    def set_token(self, token: str) -> None:
        if token != self._token:
            self._memory.clear()
            self._blocked.clear()          # 换密钥后重新尝试
            self._fail_streak.clear()
            self._auth_error = None
        self._token = token or ""

    def require_valid_token(self, source: TileSource) -> None:
        """需要密钥的图层: 校验密钥, 不合规直接抛 :class:`TiandituTokenError`."""
        if not source.requires_token:
            return
        token = (self._token or "").strip()
        if not token:
            raise TiandituTokenError(
                "未配置天地图 tk 密钥。请在 工具栏→天地图密钥 中填写 "
                "(免费申请: https://console.tianditu.gov.cn/api/key), "
                "或设置环境变量 TIANDITU_TK。")
        if not is_valid_token_format(token):
            raise TiandituTokenError(
                f"天地图 tk 密钥格式不正确: {token!r} (应为 32 位十六进制字符)")
        if self._auth_error is not None:
            raise self._auth_error          # 已知被服务端拒绝, 不再重复请求

    def validate_provider(self, provider_key: str, z: int = 12,
                          x: int = 3415, y: int = 1675) -> None:
        """主动校验底图可用性 (默认取一张太湖区域的瓦片).

        密钥缺失/格式错误/被拒绝时抛 :class:`TiandituTokenError`,
        网络异常时抛 :class:`TileDownloadError`.
        """
        if provider_key not in PROVIDERS:
            raise BasemapError(f"未知底图: {provider_key}")
        provider = PROVIDERS[provider_key]
        self.require_valid_token(provider.base)
        if provider.label is not None:
            self.require_valid_token(provider.label)
        try:
            # use_cache=False: 强制走网络, 避免旧缓存掩盖失效的密钥
            self.load_tile(provider.base, int(z), int(x), int(y),
                           allow_download=True, use_cache=False)
        except BasemapError:
            raise
        except Exception as exc:
            raise TileDownloadError(f"底图不可用 ({provider.name}): {exc}") from exc
        return

    def block_reason(self, source_key: str) -> str:
        """返回当前被熔断的原因 (无则空串)."""
        item = self._blocked.get(source_key)
        if item and time.time() < item[0]:
            return item[1]
        return ""

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    @property
    def _requests(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _cache_file(self, source: TileSource, z: int, x: int, y: int) -> Path:
        return self.cache_dir / source.key / str(z) / str(int(x)) / f"{int(y)}.png"

    def _placeholder(self, text: str = "", size: int = geo.TILE_SIZE) -> np.ndarray:
        """缺图占位块 (浅灰, 可选文字)."""
        if Image is None:
            return np.full((size, size, 4), 235, dtype=np.uint8)
        img = Image.new("RGBA", (size, size), (238, 238, 240, 255))
        if text:
            draw = ImageDraw.Draw(img)
            draw.text((8, size // 2 - 6), text[:22], fill=(150, 150, 155, 255))
            draw.rectangle([0, 0, size - 1, size - 1], outline=(210, 210, 214, 255))
        return np.asarray(img)

    # ------------------------------------------------------------------
    def load_tile(self, source: TileSource, z: int, x: int, y: int,
                  allow_download: bool = True, use_cache: bool = True) -> np.ndarray:
        """取单张瓦片 (RGBA ndarray).

        密钥语义 (天地图等需要 tk 的图层):

        * 密钥为空 / 格式不合法 -> 立即抛 :class:`TiandituTokenError` (在查缓存之前);
        * 密钥已被服务端拒绝 (HTTP 401/403/418) -> 记住异常, 之后每次调用都立即抛出,
          即使该瓦片命中缓存也不会被掩盖;
        * 未经校验的密钥命中缓存时直接返回缓存瓦片 (离线可用), 需要联网取新瓦片时
          若被拒绝则抛异常 —— 界面可通过 :meth:`validate_provider` 主动校验。

        ``use_cache=False`` 时跳过内存/磁盘缓存, 强制走网络 (用于密钥在线校验).
        """
        key = (source.key, int(z), int(x), int(y))
        self.require_valid_token(source)          # 无效密钥 -> 直接抛异常

        if use_cache:
            with self._lock:
                hit = self._memory.get(key)
                if hit is not None:
                    self._memory.move_to_end(key)
                    self.stats["hit_mem"] += 1
                    return hit

        blocked = self.block_reason(source.key)
        if blocked:
            return self._placeholder("密钥无效" if "41" in blocked else "暂不可用")

        path = self._cache_file(source, z, x, y)
        if use_cache and path.exists():
            try:
                with Image.open(path) as im:
                    arr = np.asarray(im.convert("RGBA"))
                self._remember(key, arr)
                self.stats["hit_disk"] += 1
                return arr
            except Exception:
                try:
                    path.unlink()
                except Exception:
                    pass

        if not allow_download:
            return self._placeholder("")

        url = source.format_url(z, x, y, self._token)
        try:
            resp = self._requests.get(url, timeout=self.timeout,
                                      headers={"User-Agent": USER_AGENT,
                                               "Referer": "https://www.tianditu.gov.cn/"})
            ctype = resp.headers.get("Content-Type", "")
            if resp.status_code != 200 or "image" not in ctype:
                self.stats["fail"] += 1
                self.last_error = f"HTTP {resp.status_code} {ctype.split(';')[0]}"
                if source.requires_token and (resp.status_code in (401, 403, 418)
                                              or "html" in ctype.lower()):
                    exc = TiandituTokenError(
                        f"天地图密钥被拒绝 (HTTP {resp.status_code}), "
                        f"tk={mask_token(self._token)} 无效或已过期; "
                        f"请到 工具栏→天地图密钥 更换。")
                    self._auth_error = exc
                    self._blocked[source.key] = (time.time() + 180.0, f"HTTP {resp.status_code}")
                    raise exc
                self._note_failure(source, resp.status_code, ctype)
                return self._placeholder(f"HTTP {resp.status_code}")
            with Image.open(io.BytesIO(resp.content)) as im:
                im = im.convert("RGBA")
                if im.size != (geo.TILE_SIZE, geo.TILE_SIZE):
                    im = im.resize((geo.TILE_SIZE, geo.TILE_SIZE))
                arr = np.asarray(im)
            self.stats["download"] += 1
            self._note_success(source)
            self._remember(key, arr)
            try:  # 磁盘缓存
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".part")
                with Image.fromarray(arr) as im_out:
                    im_out.save(tmp, format="PNG")
                tmp.replace(path)
            except Exception:
                pass
            return arr
        except TiandituTokenError:
            raise                                   # 密钥异常必须原样抛出
        except Exception as exc:  # 网络异常
            self.stats["fail"] += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._note_failure(source, None, type(exc).__name__)
            return self._placeholder("离线")

    def _note_failure(self, source: TileSource, status, ctype: str) -> None:
        """连续失败达到阈值时熔断该图层, 避免无密钥时反复请求拖慢界面."""
        key = source.key
        streak = self._fail_streak.get(key, 0) + 1
        self._fail_streak[key] = streak
        auth_error = status in (401, 403, 418) or "html" in str(ctype).lower()
        if streak >= 8 and auth_error:
            reason = f"HTTP {status}" if status else str(ctype)
            self._blocked[key] = (time.time() + 180.0, reason)
            self.last_error = f"{reason} (已暂停请求该图层 3 分钟)"

    def _note_success(self, source: TileSource) -> None:
        self._fail_streak[source.key] = 0
        self._blocked.pop(source.key, None)

    def _remember(self, key, arr: np.ndarray) -> None:
        with self._lock:
            self._memory[key] = arr
            self._memory.move_to_end(key)
            while len(self._memory) > self._memory_limit:
                self._memory.popitem(last=False)

    # ------------------------------------------------------------------
    def mosaic(self, provider_key: str, z: int, tx0: int, tx1: int, ty0: int, ty1: int,
               progress: Optional[Callable[[int, int], None]] = None,
               should_cancel: Optional[Callable[[], bool]] = None
               ) -> Tuple[np.ndarray, dict]:
        """拼接瓦片矩阵.

        返回 ``(rgba 数组, 信息 dict)``; 数组的行列与瓦片索引一一对应,
        extent 可用 ``geo.tile_bounds_mercator(tx0, ty0, z)`` 与右下角推算.
        """
        provider = PROVIDERS[provider_key]
        z = int(z)
        n = 2 ** z
        cols = int(tx1 - tx0 + 1)
        rows = int(ty1 - ty0 + 1)
        if cols <= 0 or rows <= 0:
            return np.zeros((1, 1, 4), np.uint8), {"tiles": 0, "fail": 0}

        jobs: List[Tuple[TileSource, int, int, int]] = []
        for ty in range(int(ty0), int(ty1) + 1):
            if ty < 0 or ty >= n:
                continue
            for tx in range(int(tx0), int(tx1) + 1):
                jobs.append((provider.base, z, geo.wrap_tile_x(tx, z), ty))
                if provider.label is not None:
                    jobs.append((provider.label, z, geo.wrap_tile_x(tx, z), ty))

        results: Dict[Tuple[str, int, int, int], np.ndarray] = {}
        auth_error: Optional[TiandituTokenError] = None
        if jobs:
            futures = {self._executor.submit(self.load_tile, s, zz, xx, yy): (s.key, zz, xx, yy)
                       for (s, zz, xx, yy) in jobs}
            done = 0
            for fut, jkey in futures.items():
                if should_cancel is not None and should_cancel():
                    # 视野已变化, 放弃等待剩余瓦片 (避免占用线程池与带宽)
                    results.setdefault(jkey, self._placeholder(""))
                    continue
                try:
                    results[jkey] = fut.result()
                except TiandituTokenError as exc:
                    # 密钥无效: 立即停止等待, 用占位块铺满并记录异常
                    auth_error = auth_error or exc
                    results.setdefault(jkey, self._placeholder("密钥无效"))
                    break
                except Exception:
                    results[jkey] = self._placeholder("")
                done += 1
                if progress is not None and done % 4 == 0:
                    progress(done, len(jobs))

        # ---- 组装 ----
        canvas = np.zeros((rows * geo.TILE_SIZE, cols * geo.TILE_SIZE, 4), np.uint8)
        canvas[..., 3] = 255
        canvas[..., :3] = 240
        fail_count = 0
        for iy, ty in enumerate(range(int(ty0), int(ty1) + 1)):
            for ix, tx in enumerate(range(int(tx0), int(tx1) + 1)):
                if ty < 0 or ty >= n:
                    tile = self._placeholder("")
                else:
                    wtx = geo.wrap_tile_x(tx, z)
                    tile = results.get((provider.base.key, z, wtx, ty))
                    if tile is None:
                        tile = self._placeholder("")
                        fail_count += 1
                    if provider.label is not None:
                        lab = results.get((provider.label.key, z, wtx, ty))
                        if lab is not None:
                            tile = _alpha_over(tile, lab)
                y0 = iy * geo.TILE_SIZE
                x0 = ix * geo.TILE_SIZE
                canvas[y0:y0 + geo.TILE_SIZE, x0:x0 + geo.TILE_SIZE] = tile
        info = {"tiles": len(jobs), "fail": fail_count, "z": z,
                "tx0": int(tx0), "tx1": int(tx1), "ty0": int(ty0), "ty1": int(ty1),
                "error_exc": auth_error}
        return canvas, info

    # ------------------------------------------------------------------
    def cache_size_bytes(self) -> int:
        total = 0
        if self.cache_dir.exists():
            for p in self.cache_dir.rglob("*.png"):
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def clear_cache(self) -> int:
        """清空磁盘瓦片缓存, 返回删除文件数."""
        removed = 0
        if self.cache_dir.exists():
            for p in self.cache_dir.rglob("*.png"):
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
        with self._lock:
            self._memory.clear()
        return removed


def _alpha_over(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """把半透明注记层叠加到底图上 (uint8 RGBA)."""
    if overlay is None or overlay.size == 0:
        return base
    if base.shape[:2] != overlay.shape[:2]:
        return base
    out = base.astype(np.float32)
    alpha = (overlay[..., 3:4].astype(np.float32) / 255.0)
    if not np.any(alpha > 0):
        return base
    out[..., :3] = out[..., :3] * (1.0 - alpha) + overlay[..., :3].astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
