# -*- coding: utf-8 -*-
"""生成合成 ATL03 演示数据 (命令行入口).

用法::

    python tools\\make_demo_data.py [输出路径] [--lon 120.17] [--lat 31.19] [--az 105]

生成的文件结构与真实 ICESat-2 ATL03 一致, 可直接用主程序打开,
用于在没有真实数据时试用"两点连线剖面"功能.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from icesat_viewer.demo_data import generate_demo_atl03  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成 ATL03 演示数据")
    parser.add_argument("output", nargs="?", default=str(ROOT / "demo_ATL03.h5"),
                        help="输出 .h5 路径")
    parser.add_argument("--lon", type=float, default=120.170, help="轨迹起点经度")
    parser.add_argument("--lat", type=float, default=31.190, help="轨迹起点纬度")
    parser.add_argument("--az", type=float, default=105.0, help="轨迹方位角 (度)")
    parser.add_argument("--length-km", type=float, default=12.0, help="轨迹长度 (km)")
    args = parser.parse_args()

    path = generate_demo_atl03(args.output, lon0=args.lon, lat0=args.lat,
                               azimuth=args.az, length_km=args.length_km)
    print(f"已生成演示数据: {path}")
    print("可用以下命令打开:  run.bat --open \"%s\"" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
