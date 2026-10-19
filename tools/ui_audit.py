# -*- coding: utf-8 -*-
"""界面布局审计: 检查所有控件是否被挤出父容器 (布局溢出).

用法::

    C:\\Users\\ASUS\\miniconda3\\envs\\waterenv\\python.exe tools\\ui_audit.py [宽 高]

对工具栏、剖面工具条、左侧面板、状态栏逐个控件检查 ``x + w > 父宽`` 或宽高 <= 1
的情况 (控件被 pack 挤到不可见时的典型现象), 同时打印各行的占用宽度。
发现问题时以非 0 退出, 便于当回归测试用。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import enable_windows_dpi_awareness, setup_matplotlib  # noqa: E402

enable_windows_dpi_awareness()
setup_matplotlib()

from icesat_viewer.app import IcesatViewerApp  # noqa: E402


def label_of(widget) -> str:
    try:
        text = widget.cget("text")
    except Exception:
        text = ""
    if not text:
        try:
            text = widget.cget("textvariable")
        except Exception:
            text = ""
    return str(text)[:24]


CONTROL_CLASSES = {"TButton", "TCombobox", "TCheckbutton", "TSpinbox", "TEntry",
                   "TRadiobutton", "TScale"}
CONTAINER_CLASSES = {"TFrame", "Treeview", "Text", "TPanedwindow", "TNotebook",
                     "TScrollbar", "TKFrame", "Canvas", "Toplevel"}


def audit_row(row, name: str, problems: list) -> None:
    row.update_idletasks()
    row_w = row.winfo_width()
    used = 0
    squeezed = []
    # 只看真实控件: 忽略撑满父容器的子框架 / 列表 / 画布
    measured = [c for c in row.winfo_children()
                if c.winfo_class() not in CONTAINER_CLASSES]
    for child in measured:
        w = child.winfo_width()
        x = child.winfo_x()
        right = x + w
        used = max(used, right)
        if w <= 1 or (right > row_w + 1 and row_w > 1):
            problems.append(f"{name}: {child.winfo_class()} {label_of(child)!r} "
                            f"x={x} w={w} 右端={right} > 行宽={row_w}")
        elif w < 8 and child.winfo_class() != "TSeparator":
            squeezed.append(f"{child.winfo_class()}:{label_of(child)}(w={w})")
    ratio = used / max(row_w, 1)
    note = ""
    if squeezed:
        note = f"  挤压: {', '.join(squeezed)}"
    if ratio >= 0.98 and row_w > 1:
        note += "  [已占满, 再加控件会被挤掉]"
    print(f"  [{name}] 宽度 {row_w}px, 最右控件 {used}px ({ratio * 100:.0f}% 占用){note}")


def audit_vertical(widget, problems: list, depth: int = 0) -> None:
    """垂直方向: 子控件底边是否超出父容器 (会被下一层裁掉)."""
    if depth > 6:
        return
    try:
        h = widget.winfo_height()
        managed = widget.winfo_manager()
    except Exception:
        return
    if managed and h > 1:
        for child in widget.winfo_children():
            try:
                if not child.winfo_manager():
                    continue
                if child.winfo_class() in ("Menu", "Toplevel", "Tk"):
                    continue            # 菜单/子窗口不参与 Tk 几何布局
                bottom = child.winfo_y() + child.winfo_height()
                if child.winfo_height() <= 1:
                    problems.append(f"垂直: {child.winfo_class()} {label_of(child)!r} 高度异常 "
                                    f"{child.winfo_height()} (父 {widget.winfo_class()} h={h})")
                elif bottom > h + 1:
                    problems.append(f"垂直: {child.winfo_class()} {label_of(child)!r} "
                                    f"底边 {bottom} > 父高 {h} ({widget.winfo_class()})")
            except Exception:
                continue
    for child in widget.winfo_children():
        audit_vertical(child, problems, depth + 1)


def main() -> int:
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 950
    app = IcesatViewerApp(provider="esri_img")
    app.geometry(f"{width}x{height}+30+20")

    def pump(sec: float = 0.5) -> None:
        end = time.time() + sec
        while time.time() < end:
            app.update()
            time.sleep(0.01)

    pump(1.8)
    problems: list = []

    print(f"窗口 {width}x{height}")
    # 递归审计所有"控件行": 直接含 >=2 个真实控件的框架 (工具栏 / 剖面参数行 / 左侧行 / 状态栏)
    seen = set()

    def row_signature(frame) -> str:
        texts = []
        for child in frame.winfo_children():
            if child.winfo_class() in CONTAINER_CLASSES:
                continue
            if child.winfo_class() == "TSeparator":
                continue
            txt = label_of(child)
            if txt:
                texts.append(txt)
            if len(texts) >= 3:
                break
        return "+".join(texts) if texts else frame.winfo_class()

    def walk(widget, depth=0):
        if depth > 5 or id(widget) in seen:
            return
        seen.add(id(widget))
        if widget.winfo_class() in ("TFrame", "TKFrame"):
            n_ctrl = sum(1 for c in widget.winfo_children()
                         if c.winfo_class() in CONTROL_CLASSES)
            if n_ctrl >= 2:
                audit_row(widget, row_signature(widget), problems)
        for child in widget.winfo_children():
            walk(child, depth + 1)

    walk(app)
    audit_vertical(app, problems)

    # 剖面图落点: 画布底边必须在本窗口内 (否则横轴标签会被状态栏切掉)
    try:
        canvas_w = app.profile.canvas.get_tk_widget()
        bottom = canvas_w.winfo_rooty() - app.winfo_rooty() + canvas_w.winfo_height()
        status_top = app.status_label.winfo_rooty() - app.winfo_rooty()
        print(f"  [剖面画布] 高 {canvas_w.winfo_height()}px, 底边 {bottom}px, "
              f"状态栏顶 {status_top}px, 窗口高 {app.winfo_height()}px")
        if bottom > status_top + 1:
            problems.append(f"剖面画布底边 {bottom} 超出状态栏顶 {status_top} (横轴标签会被裁掉)")
    except Exception as exc:
        print("  剖面画布检查跳过:", exc)

    # 关键控件可见性
    key_widgets = {
        "参考模型下拉框": app.reference_combo,
        "显示方式下拉框": app.ref_mode_combo,
        "DEM 开关": app.dem_check,
        "底图下拉框": app.provider_combo,
        "缩放按钮": app.zoom_button,
        "拾取开关": app.pick_button,
        "数据列表": app.tree,
    }
    for name, widget in key_widgets.items():
        widget.update_idletasks()
        ok = widget.winfo_width() > 10 and widget.winfo_rootx() >= 0 \
            and widget.winfo_rooty() >= 0
        parent_w = widget.master.winfo_width()
        right = widget.winfo_x() + widget.winfo_width()
        visible = ok and (parent_w <= 1 or right <= parent_w + 1)
        print(f"  {name}: {'可见' if visible else '不可见/被裁剪'} "
              f"(x={widget.winfo_x()}, w={widget.winfo_width()}, 父宽={parent_w})")
        if not visible:
            problems.append(f"{name} 不可见或被裁剪")

    app.after(100, app._on_close)
    pump(0.6)

    print()
    if problems:
        print(f"发现 {len(problems)} 处布局问题:")
        for p in problems:
            print("  !!", p)
        return 1
    print("布局审计通过: 无控件被挤出可视区 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
