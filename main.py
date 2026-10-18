# -*- coding: utf-8 -*-
"""ICESat 光子剖面查看器 —— 程序入口.

用法::

    python main.py                        # 直接启动
    python main.py --open 文件.h5          # 启动并打开数据
    python main.py --demo                 # 生成并打开一份合成演示数据
    python main.py --provider esri_img    # 指定底图
    python main.py --token <天地图tk>      # 临时指定天地图密钥

推荐使用已装好依赖的 conda 环境 (waterenv), 见 run.bat / README.md.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED = [("numpy", "numpy"), ("matplotlib", "matplotlib"), ("PIL", "pillow"),
            ("requests", "requests"), ("h5py", "h5py")]
OPTIONAL = [("pyproj", "pyproj (高精度大地线)"), ("scipy", "scipy (覆盖判定加速)"),
            ("pandas", "pandas (文本数据读取)")]


def check_dependencies() -> list:
    """返回缺失的必需依赖列表."""
    import importlib

    missing = []
    for module, package in REQUIRED:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(package)
    return missing


def make_console_tolerant() -> None:
    """容忍控制台编码: 中文 Windows 的 cmd 默认 GBK, 个别字符无法编码时
    只替换该字符而不是抛 UnicodeEncodeError 让程序崩掉."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")   # type: ignore[union-attr]
        except Exception:
            pass


def enable_windows_dpi_awareness() -> None:
    """让窗口在高分屏下清晰 (必须在创建 Tk 之前调用)."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)      # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def setup_matplotlib() -> None:
    """中文字体与负号显示."""
    import matplotlib

    matplotlib.use("TkAgg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei UI", "Microsoft YaHei",
                                              "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.autolayout"] = False
    matplotlib.rcParams["savefig.facecolor"] = "white"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ICESat 光子剖面查看器 (天地图底图 + 两点连线剖切)")
    parser.add_argument("--open", "-o", nargs="*", default=[], metavar="FILE",
                        help="启动时打开的 ICESat 数据文件 (可多个)")
    parser.add_argument("--provider", default=None,
                        help="底图方案: tianditu_img / tianditu_vec / tianditu_ter / "
                             "esri_img / esri_topo")
    parser.add_argument("--token", default=None, help="天地图 tk 密钥 (临时覆盖配置)")
    parser.add_argument("--demo", action="store_true",
                        help="生成一份合成 ATL03 演示数据并打开")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    make_console_tolerant()
    args = _parse_args(argv)

    missing = check_dependencies()
    if missing:
        print("缺少依赖: " + ", ".join(missing), file=sys.stderr)
        print("请安装后重试, 例如:\n"
              "    C:\\Users\\ASUS\\miniconda3\\envs\\waterenv\\python.exe -m pip install "
              + " ".join(missing), file=sys.stderr)
        print("或直接双击 run.bat (已指向 waterenv 环境)。", file=sys.stderr)
        return 2

    enable_windows_dpi_awareness()
    setup_matplotlib()

    from icesat_viewer.app import IcesatViewerApp
    from icesat_viewer.settings import Settings

    files = list(args.open)
    if args.demo:
        from icesat_viewer.demo_data import generate_demo_atl03

        demo_path = ROOT / "demo_ATL03.h5"
        generate_demo_atl03(demo_path)
        print(f"已生成演示数据: {demo_path}")
        files.append(str(demo_path))

    app = IcesatViewerApp(settings=Settings(), provider=args.provider,
                          token=args.token, files=files)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
