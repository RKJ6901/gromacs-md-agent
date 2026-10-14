"""共享绘图层：GPCR Nature 风格排版原语。

md_tools.py（xvg 基础图）与 md_gpcr_plots.py（GPCR 分层图）都从这里取样式，
保证全项目出图规范一致（照 plot_aplnr_nature.py 的惯例）：
  - 细线框、去顶右脊、外向刻度
  - 字号 7-8.5pt，单栏宽 3.6 in
  - PDF 矢量 + PNG 600dpi 双份
只依赖 matplotlib/numpy，不 import 项目内其它模块，避免循环依赖。
"""
import os
import csv
import math

import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#202124"
GREY = "#697078"
COLORS = {"A": "#167A86", "B": "#D04A36"}

# 跨膜螺旋：青色系深浅；环区：暖灰褐系；末端：中性灰
_TM_RAMP = ["#0E4E57", "#166A76", "#1E8794", "#2FA3AF", "#0B414A", "#247F8D", "#63B8C1"]
_LOOP_RAMP = {"ICL1": "#B07A33", "ICL2": "#C99548", "ICL3": "#DDB877",
              "ECL1": "#8F6A34", "ECL2": "#A8813F", "ECL3": "#C9A566"}


def seg_color(name):
    """段名 -> 颜色：TMn 用青色梯度，环区用暖褐梯度，其余中性灰。"""
    if name in _LOOP_RAMP:
        return _LOOP_RAMP[name]
    if name.startswith("TM") and name[2:].isdigit():
        k = int(name[2:]) - 1
        if 0 <= k < len(_TM_RAMP):
            return _TM_RAMP[k]
    return GREY


def setup_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white",
        "legend.frameon": False, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def clean(ax, grid=False, grid_axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=.7, colors=INK)
    if grid:
        ax.grid(axis=grid_axis, color="#D9DDDF", linewidth=.55, zorder=0)


def panel(ax, letter):
    ax.text(-.14, 1.08, letter, transform=ax.transAxes, weight="bold",
            size=10, va="top", ha="left", color=INK)


def save(fig, stem):
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 读表
def read_csv(path):
    """读 worker 写出的 CSV，返回 (header: list[str], rows: list[list[str]])。缺文件返回 (None, None)。"""
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rd = list(csv.reader(f))
    if not rd:
        return None, None
    return rd[0], rd[1:]


def col(header, rows, name, cast=float):
    """按列名取一列（名字不区分大小写、去空白），转不成数值的行记 NaN。"""
    if not header or name not in header:
        return None
    j = header.index(name)
    out = []
    for r in rows:
        if j >= len(r):
            out.append(float("nan"))
            continue
        try:
            out.append(cast(r[j]))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def grid_from_triplets(header, rows, xcol, ycol, zcol):
    """把 (x, y, z) 长表转成规整网格矩阵，返回 (xs, ys, Z)（Z[i][j] 对应 ys[i], xs[j]）。

    sum_hills 与 FES 输出都是长表（每行一个 (cv1, cv2, F)），画 contourf 前必须转网格。
    """
    sx, sy, sz = set(), set(), {}
    for x, y, z in zip(col(header, rows, xcol), col(header, rows, ycol), col(header, rows, zcol)):
        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            continue
        sx.add(x)
        sy.add(y)
        sz[(x, y)] = z
    xs = sorted(sx)
    ys = sorted(sy)
    if not xs or not ys:
        return [], [], []
    Z = [[sz.get((x, y), float("nan")) for x in xs] for y in ys]
    return xs, ys, Z


def finite(vals):
    return [v for v in vals if v is not None and not math.isnan(v)]


def nan_safe_min(vals, default=0.0):
    f = finite(vals)
    return min(f) if f else default


def nan_safe_max(vals, default=1.0):
    f = finite(vals)
    return max(f) if f else default


# ---------------------------------------------------------------- GPCR 专用助手
def shade_segments(ax, segments, order=None, alpha=0.10, label=False):
    """在坐标轴上按段给背景打色带（x 轴为残基号），直观标出 TM/环区边界。"""
    if not segments:
        return
    names = order or list(segments.keys())
    for k in names:
        if k not in segments:
            continue
        a, b = segments[k]
        ax.axvspan(a - .5, b + .5, color=seg_color(k), alpha=alpha, lw=0, zorder=0)
        if label:
            ax.text((a + b) / 2.0, 0.98, k.replace("TM", "TM"), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=5.5, color=GREY, rotation=90)


def mark_bw(ax, anchors, y, bw_list, color=None):
    """在 y 高度处标出若干 BW 位点，返回实际标注的 BW 列表。"""
    used = []
    for bw in bw_list:
        if bw not in anchors:
            continue
        ax.plot([anchors[bw]], [y], marker="v", ms=3.2, mfc=color or COLORS["B"],
                mec="white", mew=.4, zorder=5)
        ax.annotate(bw, (anchors[bw], y), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=5.5, color=INK)
        used.append(bw)
    return used


def ribbon(ax, x, y, color=COLORS["A"], lw=0.9, alpha=0.18, label=None):
    """折线 + 半透明误差带（±1σ 或 CI 半宽）；y 为均值序列，需自行传入上下界时用 band。"""
    ax.plot(x, y, color=color, lw=lw, label=label)
    return ax


def band(ax, x, lo, hi, color=COLORS["A"], alpha=0.16):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)


def bar_labels(ax, labels, values, colors=None, fmt="%.2f", errs=None):
    xs = range(len(labels))
    ax.bar(xs, values, color=colors or COLORS["A"], width=.62,
           yerr=errs, error_kw=dict(lw=.7, capsize=1.8, ecolor=GREY))
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        ax.annotate(fmt % v, (i, v), textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=5.5, color=GREY)


def heatmap(ax, Z, extent=None, cmap="RdBu_r", vmin=None, vmax=None, cbar_label="", fig=None):
    import numpy as np
    A = np.asarray(Z, dtype=float)
    im = ax.imshow(A, origin="lower", aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest")
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
        cb.ax.tick_params(labelsize=6, length=2, width=.6)
        cb.outline.set_linewidth(.6)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=6.5)
    return im
