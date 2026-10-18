# -*- coding: utf-8 -*-
"""ICESat 光子剖面查看器 (天地图底图 + 两点连线剖面).

模块划分::

    geo.py         经纬度 / Web Mercator 投影 / 大地线计算
    settings.py    配置读写 (天地图 tk 密钥、显示参数)
    basemap.py     瓦片底图引擎 (天地图 WMTS + 免密钥备用底图, 带磁盘缓存)
    icesat_io.py   ICESat 数据读取 (ATL03 惰性读取 / ATL06 / ATL08 / GLAH14 / CSV / LAS)
    corridor.py    两点连线廊道的光子提取与剖面统计
    mapview.py     tkinter 地图画布 (平移 / 缩放 / 拾取)
    profileview.py 剖面视图 (高程-沿轨距离散点 + 高程直方图)
    app.py         主窗体, 组织以上部件
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
