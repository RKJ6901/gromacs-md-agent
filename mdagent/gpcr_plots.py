"""GPCR / 元动力学分析出图：读 worker 写出的 CSV/JSON，产出 Nature 风格多面板图。

设计原则：
  - 与 tools.plot_md_figures 共用 plot_style 的排版规范（细线、去脊、600dpi）
  - 每个 figure family 独立取数：缺哪个 CSV 就跳过该 family 并写明原因，
    绝不因为缺一块数据就整组不出图（元动力学分析里 PLUMED 数据经常缺失）
  - 图里凡是有"均值"的地方都尽量带上 CI/ESS 或时间序列，避免只给一个数

产物：<ws>/figs/gpcr_<family>.pdf 与 .png
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt

import plot_style as S

SEG_ORDER = ['Nterm', 'TM1', 'ICL1', 'TM2', 'ECL1', 'TM3', 'ICL2', 'TM4',
             'ECL2', 'TM5', 'ICL3', 'TM6', 'ECL3', 'TM7', 'H8', 'Cterm']
C_A = S.COLORS["A"]
C_B = S.COLORS["B"]


def _json(ws, name):
    p = os.path.join(ws, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _tbl(ws, name):
    return S.read_csv(os.path.join(ws, name))


def _stem(ws, out_dir, family):
    return os.path.join(ws, out_dir, "gpcr_" + family)


def _profile(ws):
    return _json(ws, "gpcr.json") or {}


def _title(ax, text):
    ax.set_title(text, loc="left", pad=4)


# ================================================================ 1. 分段稳定性
def fig_segments(ws, out_dir):
    h1, r1 = _tbl(ws, "segments_rmsd.csv")
    h2, r2 = _tbl(ws, "segments_rmsf.csv")
    if not h1 and not h2:
        return False, "缺 segments_rmsd.csv / segments_rmsf.csv（先跑 analyze_segments）"
    prof = _profile(ws)
    segs = prof.get("segments", {})
    anchors = prof.get("anchors", {})

    n_panels = (1 if (h1 and r1) else 0) + (1 if (h2 and r2) else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(4.6, 2.05 * n_panels), squeeze=False)
    axes = [a[0] for a in axes]
    letters = "ABCDEFGH"
    k = 0

    if h1 and r1:
        ax = axes[k]
        t = S.col(h1, r1, "time_ns")
        names = [c for c in h1[1:] if c.strip()]
        for nm in names:
            y = S.col(h1, r1, nm)
            if y is None:
                continue
            ax.plot(t, y, color=S.seg_color(nm), lw=0.85, label=nm)
        S.clean(ax, grid=True)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("C-alpha RMSD (nm)")
        _title(ax, "Per-segment RMSD (core TM1-TM4 aligned)")
        ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.0))
        S.panel(ax, letters[k])
        k += 1

    if h2 and r2:
        ax = axes[k]
        res = S.col(h2, r2, "resnum")
        rmsf = S.col(h2, r2, "rmsf_nm")
        S.shade_segments(ax, segs, order=SEG_ORDER, alpha=0.11)
        ax.plot(res, rmsf, color=C_A, lw=0.9)
        S.clean(ax)
        ax.set_xlabel("Residue number")
        ax.set_ylabel("C-alpha RMSF (nm)")
        _title(ax, "Residue flexibility")
        ymax = S.nan_safe_max(rmsf, 1.0)
        S.mark_bw(ax, anchors, ymax * 0.97,
                  ["3.49", "3.50", "5.50", "6.44", "6.47", "6.48", "7.39", "7.49", "7.53"])
        S.panel(ax, letters[k])
        k += 1

    fin = S.finite(S.col(h2, r2, "rmsf_nm")) if (h2 and r2) else []
    note = "flexible max %.2f nm" % max(fin) if fin else ""
    S.save(fig, _stem(ws, out_dir, "segments"))
    return True, ("分段 RMSD/RMSF 已出图" + (("；" + note) if note else ""))


# ================================================================ 2. 微开关
def _switch_tables(ws):
    import glob
    files = sorted(glob.glob(os.path.join(ws, "microswitch_*.csv")))
    tabs = []
    for fp in files:
        nm = os.path.basename(fp)
        sw = nm[len("microswitch_"):-len(".csv")]
        h, r = S.read_csv(fp)
        if h and r:
            tabs.append((sw, h, r))
    return tabs


def fig_switches(ws, out_dir):
    tabs = _switch_tables(ws)
    if not tabs:
        return False, "缺 microswitch_*.csv（先跑 analyze_microswitches）"
    dist_cols, dih_cols = [], []
    for sw, h, r in tabs:
        for c in h[1:]:
            if "chi" in c:
                dih_cols.append((sw, c))
            else:
                dist_cols.append((sw, c))
    n_panels = 1 + (1 if dih_cols else 0) + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(4.6, 1.95 * n_panels), squeeze=False)
    axes = [a[0] for a in axes]
    letters = "ABCDEFGH"
    idx = {sw: (h, r) for sw, h, r in tabs}

    ax = axes[0]
    for sw, c in dist_cols:
        h, r = idx[sw]
        ax.plot(S.col(h, r, "time_ns"), S.col(h, r, c), lw=0.85, label=sw + " " + c)
    S.clean(ax, grid=True)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Distance (nm)")
    _title(ax, "Microswitch distances")
    ax.legend(ncol=2, loc="best")
    S.panel(ax, letters[0])

    k = 1
    if dih_cols:
        ax = axes[k]
        for sw, c in dih_cols:
            h, r = idx[sw]
            ax.plot(S.col(h, r, "time_ns"), S.col(h, r, c), lw=0.85, label=sw + " " + c)
        S.clean(ax, grid=True)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Dihedral (deg)")
        ax.axhline(0, color=S.GREY, lw=0.5, ls=":")
        _title(ax, "Sidechain rotamers (chi1/chi2)")
        ax.legend(ncol=2, loc="best")
        S.panel(ax, letters[k])
        k += 1

    ax = axes[k]
    js = _json(ws, "analysis_microswitches.json") or {}
    sws = js.get("switches", {})
    lab, occ4, occ5 = [], [], []
    for sw in list(sws.keys()):
        for pair, v in (sws.get(sw, {}).get("pairs", {}) or {}).items():
            lab.append(pair)
            occ4.append(v.get("occ_0.4nm", float("nan")))
            occ5.append(v.get("occ_0.5nm", float("nan")))
    if lab:
        xs = np.arange(len(lab))
        ax.bar(xs - 0.18, occ4, width=0.34, color=C_A, label="engaged <0.4 nm")
        ax.bar(xs + 0.18, occ5, width=0.34, color=C_B, label="engaged <0.5 nm")
        ax.set_xticks(xs)
        ax.set_xticklabels(lab, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Occupancy")
        ax.legend(loc="best")
    else:
        ax.text(.5, .5, "no pairs in analysis_microswitches.json",
                transform=ax.transAxes, ha="center", va="center", color=S.GREY, fontsize=7)
    S.clean(ax, grid=False)
    _title(ax, "Microswitch engagement occupancy")
    S.panel(ax, letters[k])
    S.save(fig, _stem(ws, out_dir, "switches"))
    return True, "%d 个微开关文件，%d 组距离 / %d 组二面角" % (len(tabs), len(dist_cols), len(dih_cols))


# ================================================================ 3. TM6 胞内开放
def fig_tm6(ws, out_dir):
    h, r = _tbl(ws, "tm6_opening.csv")
    if not h:
        return False, "缺 tm6_opening.csv（先跑 analyze_tm6_opening）"
    t = S.col(h, r, "time_ns")
    d63 = S.col(h, r, "TM6ic_TM3ic_nm")
    d67 = S.col(h, r, "TM6ic_TM7ic_nm")
    sh = S.col(h, r, "TM6ic_shift_nm")
    fig, axes = plt.subplots(3, 1, figsize=(4.6, 5.4), squeeze=False)
    axes = [a[0] for a in axes]

    ax = axes[0]
    ax.plot(t, d63, color=C_A, lw=0.9, label="TM6ic-TM3ic")
    if d67 and S.finite(d67):
        ax.plot(t, d67, color=C_B, lw=0.9, label="TM6ic-TM7ic")
    S.clean(ax, grid=True)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Distance (nm)")
    _title(ax, "Intracellular TM6 opening")
    ax.legend(loc="best")
    S.panel(ax, "A")

    ax = axes[1]
    ax.plot(t, sh, color=S.GREY, lw=0.9)
    S.clean(ax, grid=True)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("TM6ic displacement (nm)")
    _title(ax, "TM6ic displacement from frame 0")
    S.panel(ax, "B")

    ax = axes[2]
    v = S.finite(d63)
    if v:
        ax.hist(v, bins=min(60, max(8, len(v) // 12)), color=C_A, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        mu = float(np.mean(v))
        ax.axvline(mu, color=C_B, lw=1.0, ls="--")
        ax.annotate("mean %.3f nm" % mu, (mu, ax.get_ylim()[1] * 0.92),
                    color=C_B, fontsize=6.5, ha="left")
    S.clean(ax, grid=False)
    ax.set_xlabel("TM6ic-TM3ic (nm)")
    ax.set_ylabel("Frames")
    _title(ax, "Distribution")
    S.panel(ax, "C")
    S.save(fig, _stem(ws, out_dir, "tm6"))
    if v:
        return True, "TM6ic-TM3ic mean %.3f nm (min %.3f, max %.3f)" % (
            float(np.mean(v)), min(v), max(v))
    return True, "TM6 开放度已出图"


# ================================================================ 4. 配体保留与口袋接触
def fig_ligand(ws, out_dir):
    h, r = _tbl(ws, "ligand_retention.csv")
    hc, rc = _tbl(ws, "ligand_contacts.csv")
    if not h and not hc:
        return False, "缺 ligand_retention.csv / ligand_contacts.csv（先跑 analyze_ligand_retention）"
    n_panels = (2 if h else 0) + (1 if hc else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(4.6, 1.95 * n_panels), squeeze=False)
    axes = [a[0] for a in axes]
    k = 0

    if h:
        t = S.col(h, r, "time_ns")
        ax = axes[k]
        ax.plot(t, S.col(h, r, "ligand_rmsd_nm"), color=C_A, lw=0.9, label="ligand RMSD")
        cs = S.col(h, r, "ligand_com_shift_nm")
        if cs:
            ax.plot(t, cs, color=C_B, lw=0.9, label="COM displacement")
        S.clean(ax, grid=True)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Displacement (nm)")
        _title(ax, "Ligand stays in pocket?")
        ax.legend(loc="best")
        S.panel(ax, "A")
        k += 1

        ax = axes[k]
        ax.plot(t, S.col(h, r, "min_dist_nm"), color=S.GREY, lw=0.9,
                label="min heavy-atom distance")
        ax.axhline(0.45, color=C_B, lw=0.7, ls="--")
        S.clean(ax, grid=True)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Distance (nm)")
        _title(ax, "Closest approach to receptor (dashed = 0.45 nm)")
        ax.legend(loc="best")
        S.panel(ax, "B")
        k += 1

    if hc:
        ax = axes[k]
        res = S.col(hc, rc, "resnum")
        occ = S.col(hc, rc, "occupancy")
        seg = S.col(hc, rc, "segment", cast=str)
        bw = S.col(hc, rc, "bw", cast=str)
        pairs = [(occ[i], res[i], seg[i], bw[i]) for i in range(len(occ))]
        pairs.sort(key=lambda z: -(z[0] if z[0] == z[0] else -1))
        pairs = [p for p in pairs if p[0] > 0][:15]
        if pairs:
            lab = ["%s %s" % (int(p[1]) if p[1] == p[1] else "?", p[3] or p[2] or "") for p in pairs]
            val = [p[0] for p in pairs]
            cols = [S.seg_color(p[2] or "") for p in pairs]
            ys = np.arange(len(lab))[::-1]
            ax.barh(ys, val, color=cols, height=0.66)
            ax.set_yticks(ys)
            ax.set_yticklabels(lab)
            ax.set_xlabel("Contact occupancy (fraction of frames < 0.45 nm)")
            ax.set_xlim(0, 1.02)
        else:
            ax.text(.5, .5, "no residue contacts above 0", transform=ax.transAxes,
                    ha="center", va="center", color=S.GREY, fontsize=7)
        S.clean(ax, grid=True, grid_axis="x")
        _title(ax, "Top pocket residues by contact occupancy")
        S.panel(ax, "C")
    S.save(fig, _stem(ws, out_dir, "ligand"))
    if h:
        md = S.finite(S.col(h, r, "min_dist_nm"))
        if md:
            return True, "共 %d 帧；最近重原子距离 mean %.3f nm，max %.3f nm" % (
                len(md), float(np.mean(md)), max(md))
    return True, "配体保留与口袋接触已出图"


# ================================================================ 5. 构象景观
def _fes_panel(ax, ws, fname, xcol, ycol, fmax):
    hp, rp = _tbl(ws, fname)
    if not hp:
        return None
    xs, ys, Z = S.grid_from_triplets(hp, rp, xcol, ycol, "F_kJmol")
    if not xs or not ys:
        return None
    A = np.asarray(Z, dtype=float)
    A = np.where(A > fmax, np.nan, A)
    if not np.isfinite(A).any():
        return None
    X, Y = np.meshgrid(np.asarray(xs), np.asarray(ys))
    lo = float(np.nanmin(A))
    hi = float(np.nanpercentile(A, 98))
    lv = np.linspace(lo, max(hi, lo + 1.0), 22)
    ax.contourf(X, Y, A, levels=lv, cmap="viridis_r", extend="max")
    ax.contour(X, Y, A, levels=lv[::4], colors="white", linewidths=0.3, alpha=0.5)
    return (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))


def _scatter_states(ax, ws, px, py):
    h, r = _tbl(ws, "landscape_projection.csv")
    if not h:
        return
    sx = S.col(h, r, px)
    sy = S.col(h, r, py)
    st = S.col(h, r, "state")
    if not sx or not st:
        return
    step = max(1, len(sx) // 900)
    cm = plt.get_cmap("tab10")
    ks = sorted({int(v) for v in st if v == v})
    for k in ks:
        xs = [sx[i] for i in range(0, len(sx), step) if int(st[i]) == k]
        ys = [sy[i] for i in range(0, len(sy), step) if int(st[i]) == k]
        if xs:
            ax.scatter(xs, ys, s=1.1, color=cm(k % 10), alpha=0.35, lw=0)


def fig_landscape(ws, out_dir):
    hn, rn = _tbl(ws, "landscape_states.csv")
    ht, rt = _tbl(ws, "landscape_transition.csv")
    has_pc = _tbl(ws, "landscape_fes_pc.csv")[0] is not None
    has_tica = _tbl(ws, "landscape_fes_tica.csv")[0] is not None
    if not (has_pc or has_tica or hn):
        return False, "缺 landscape_*.csv（先跑 analyze_landscape）"
    js = _json(ws, "analysis_landscape.json") or {}
    fmax = 200.0
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4), constrained_layout=True)
    axes = axes.ravel()
    for ax in axes:
        S.clean(ax)
    letters = "ABCD"

    ax = axes[0]
    ext = _fes_panel(ax, ws, "landscape_fes_pc.csv", "PC1", "PC2", fmax)
    _scatter_states(ax, ws, "PC1", "PC2")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    _title(ax, "FES on PC1/PC2 (kJ/mol)")
    S.panel(ax, "A")

    ax = axes[1]
    ext2 = _fes_panel(ax, ws, "landscape_fes_tica.csv", "tIC1", "tIC2", fmax)
    _scatter_states(ax, ws, "tIC1", "tIC2")
    ax.set_xlabel("tIC1")
    ax.set_ylabel("tIC2")
    _title(ax, "FES on tICA space (kJ/mol)")
    S.panel(ax, "B")

    ax = axes[2]
    if hn:
        st = [int(v) for v in S.col(hn, rn, "state")]
        occ = S.col(hn, rn, "occupancy")
        dw = S.col(hn, rn, "mean_dwell_ns")
        xs = np.arange(len(st))
        ax.bar(xs, occ, color=C_A, width=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(["S%d" % s for s in st])
        ax.set_ylabel("Occupancy")
        ax.set_ylim(0, min(1.0, max(S.finite(occ) + [0]) * 1.25 + 1e-6))
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.plot(xs, dw, "o--", color=C_B, ms=3.5, lw=0.9)
        ax2.set_ylabel("Mean dwell (ns)", color=C_B)
        ax2.tick_params(colors=C_B, length=2.5, width=0.6)
    _title(ax, "State occupancy / dwell")
    S.panel(ax, "C")

    ax = axes[3]
    if ht:
        names = [c for c in ht[1:] if c.startswith("P_to_")]
        P = np.asarray([[float(v) for v in row[1:1 + len(names)]] for row in rt], dtype=float)
        S.heatmap(ax, P, cmap="Blues", vmin=0.0, vmax=1.0, fig=fig, cbar_label="P")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(["S%d" % i for i in range(len(names))])
        ax.set_yticks(range(len(rt)))
        ax.set_yticklabels(["S%d" % int(v) for v in S.col(ht, rt, "from")])
        ax.set_xlabel("to")
        ax.set_ylabel("from")
    else:
        ax.text(.5, .5, "no transition matrix", transform=ax.transAxes,
                ha="center", va="center", color=S.GREY, fontsize=7)
    _title(ax, "Transition probabilities")
    S.panel(ax, "D")
    S.save(fig, _stem(ws, out_dir, "landscape"))
    return True, "FES(PC %s / tICA %s) + 态占用与转移矩阵已出图" % (has_pc, has_tica)


# ================================================================ 6. 残基耦合 (DCCM)
def _square(ws, fname):
    h, r = _tbl(ws, fname)
    if not h:
        return None, None
    res = [int(float(v)) for v in S.col(h, r, h[0])]
    M = np.asarray([[float(v) for v in row[1:1 + len(res)]] for row in r], dtype=float)
    return res, M


def _seg_edges(res, segs, ax, vertical=True):
    for nm in SEG_ORDER:
        if nm not in segs:
            continue
        a = segs[nm][0]
        if a < res[0] or a > res[-1]:
            continue
        if vertical:
            ax.axvline(a, color="white", lw=0.5, alpha=0.8)
        else:
            ax.axhline(a, color="white", lw=0.5, alpha=0.8)


def fig_coupling(ws, out_dir):
    res, M = _square(ws, "dccm.csv")
    hc, rc = _tbl(ws, "segment_coupling.csv")
    if M is None and not hc:
        return False, "缺 dccm.csv / segment_coupling.csv（先跑 analyze_correlation）"
    n = 1 + (1 if hc else 0)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n + 0.6, 3.6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    segs = _profile(ws).get("segments", {})

    ax = axes[0]
    if M is not None:
        ext = [res[0] - .5, res[-1] + .5, res[0] - .5, res[-1] + .5]
        S.heatmap(ax, M, extent=ext, cmap="RdBu_r", vmin=-1, vmax=1, fig=fig,
                  cbar_label="DCCM")
        _seg_edges(res, segs, ax, True)
        _seg_edges(res, segs, ax, False)
        ax.set_xlabel("Residue number")
        ax.set_ylabel("Residue number")
    else:
        ax.text(.5, .5, "no DCCM", transform=ax.transAxes, ha="center",
                va="center", color=S.GREY, fontsize=7)
    _title(ax, "Dynamic cross-correlation")
    S.panel(ax, "A")

    if hc:
        ax = axes[1]
        names = [c for c in hc[1:] if c.strip()]
        Sm = np.asarray([[float(v) for v in row[1:1 + len(names)]] for row in rc], dtype=float)
        S.heatmap(ax, Sm, cmap="magma_r", vmin=0, fig=fig, cbar_label="|DCCM|")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90)
        ax.set_yticks(range(len(rc)))
        ax.set_yticklabels(names)
    _title(ax, "Segment coupling (mean |DCCM|)")
    S.panel(ax, "B")
    S.save(fig, _stem(ws, out_dir, "coupling"))
    return True, "DCCM + 段耦合矩阵已出图"


# ================================================================ 7. 蛋白结构网络 PSN
def fig_psn(ws, out_dir):
    h, r = _tbl(ws, "psn_residues.csv")
    if not h:
        return False, "缺 psn_residues.csv（先跑 analyze_network）"
    res = S.col(h, r, "resnum")
    bc = S.col(h, r, "betweenness")
    deg = S.col(h, r, "degree")
    com = S.col(h, r, "community")
    rew = S.col(h, r, "rewired")
    segs = _profile(ws).get("segments", {})
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 5.2), squeeze=False)
    axes = [a[0] for a in axes]

    ax = axes[0]
    S.shade_segments(ax, segs, order=SEG_ORDER, alpha=0.11)
    ml, sl, bl = ax.stem(res, bc, basefmt=" ")
    plt.setp(sl, color=C_A, lw=0.7)
    plt.setp(ml, color=C_A, ms=0, lw=0)
    S.clean(ax)
    ax.set_ylabel("Betweenness")
    _title(ax, "Network bottleneck residues")
    top = sorted(range(len(bc)), key=lambda i: -(bc[i] if bc[i] == bc[i] else -1))[:5]
    for i in top:
        ax.annotate(str(int(res[i])), (res[i], bc[i]), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=5.5, color=C_B)
    S.panel(ax, "A")

    ax = axes[1]
    S.shade_segments(ax, segs, order=SEG_ORDER, alpha=0.11)
    ax.scatter(res, com, s=2.2, c=C_A, lw=0, alpha=0.85)
    S.clean(ax)
    ax.set_xlabel("Residue number")
    ax.set_ylabel("Community")
    _title(ax, "Community assignment")
    S.panel(ax, "B")

    ax = axes[2]
    S.shade_segments(ax, segs, order=SEG_ORDER, alpha=0.11)
    ax.plot(res, deg, color=S.GREY, lw=0.7)
    rw = []
    if rew:
        rw = [res[i] for i in range(len(rew)) if rew[i] == 1]
    if rw:
        rwv = [deg[int(res.index(x))] for x in rw]
        ax.scatter(rw, rwv, s=8, facecolor="none", edgecolor=C_B, lw=0.6, zorder=5)
    S.clean(ax, grid=True)
    ax.set_xlabel("Residue number")
    ax.set_ylabel("Degree")
    _title(ax, "Contact degree (open circles = rewired residues)")
    S.panel(ax, "C")
    S.save(fig, _stem(ws, out_dir, "psn"))
    return True, "PSN 已出图；重排残基 %d 个" % len(rw)


# ================================================================ 8. 水网络
def fig_water(ws, out_dir):
    h, r = _tbl(ws, "water_sites.csv")
    if not h:
        return False, "缺 water_sites.csv（先跑 analyze_water_network）"
    site = S.col(h, r, "site", cast=str)
    rn = S.col(h, r, "resnum")
    seg = S.col(h, r, "segment", cast=str)
    occ = S.col(h, r, "occupancy")
    brg = S.col(h, r, "water_bridge_fraction")
    nmax = min(12, len(site))
    lab = ["%s %s" % (site[i] or "", int(rn[i]) if rn[i] == rn[i] else "") for i in range(nmax)]
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 5.2), squeeze=False)
    axes = [a[0] for a in axes]

    ax = axes[0]
    ys = np.arange(nmax)[::-1]
    ax.barh(ys, occ[:nmax], color=[S.seg_color(seg[i] or "") for i in range(nmax)], height=0.66)
    ax.set_yticks(ys)
    ax.set_yticklabels(lab)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Site occupancy (fraction of frames)")
    S.clean(ax, grid=True, grid_axis="x")
    _title(ax, "Top water sites")
    S.panel(ax, "A")

    ax = axes[1]
    h2, r2 = _tbl(ws, "water_occupancy.csv")
    if h2 and r2:
        t = S.col(h2, r2, "time_ns")
        for i in range(min(4, nmax)):
            cn = "water_n_" + (site[i] or "")
            y = S.col(h2, r2, cn)
            if y is not None:
                ax.plot(t, y, lw=0.85, label=lab[i])
        ax.legend(ncol=2, loc="best")
    S.clean(ax, grid=True)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Water count")
    _title(ax, "Water count at top sites")
    S.panel(ax, "B")

    ax = axes[2]
    ax.barh(ys, brg[:nmax], color=C_B, height=0.66)
    ax.set_yticks(ys)
    ax.set_yticklabels(lab)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Water-bridge fraction (same water bridges site and ligand)")
    S.clean(ax, grid=True, grid_axis="x")
    _title(ax, "Water bridges")
    S.panel(ax, "C")
    S.save(fig, _stem(ws, out_dir, "water"))
    o = S.finite(occ[:nmax])
    return True, "%d 个水位点；最高占据率 %.2f" % (nmax, max(o) if o else 0.0)


# ================================================================ 9. 元动力学 FES
def _fes_free_col(h):
    for c in h:
        cl = c.lower()
        if "free" in cl or cl in ("f", "f_kjmol", "fes"):
            return c
    return h[1] if len(h) >= 3 else None


def _panel_fes2d(ax, h, r, fcut, fig):
    xs, ys, Z = S.grid_from_triplets(h, r, h[0], h[1], h[2])
    if not xs or not ys:
        return False
    A = np.asarray(Z, dtype=float)
    A = np.where(A > fcut, np.nan, A)
    if not np.isfinite(A).any():
        return False
    X, Y = np.meshgrid(np.asarray(xs), np.asarray(ys))
    lo = float(np.nanmin(A))
    hi = max(float(np.nanpercentile(A, 98)), lo + 1.0)
    lv = np.linspace(lo, hi, 24)
    cf = ax.contourf(X, Y, A, levels=lv, cmap="viridis_r", extend="max")
    cb = fig.colorbar(cf, ax=ax, fraction=.046, pad=.02)
    cb.set_label("F (kJ/mol)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=2, width=.6)
    ax.set_xlabel(h[0])
    ax.set_ylabel(h[1])
    return True


def _panel_fes1d(ax, h, r, fcut, bas):
    cv = S.col(h, r, h[0])
    fc = _fes_free_col(h)
    if not cv or not fc:
        return False
    fv = np.asarray(S.col(h, r, fc), dtype=float)
    fv = fv - np.nanmin(fv)
    ax.plot(cv, fv, color=C_A, lw=1.0)
    ax.axhline(fcut, color=C_B, lw=0.7, ls="--")
    gm = bas.get("cv1_at_global_min")
    if gm is not None:
        ax.axvline(gm, color=S.GREY, lw=0.7, ls=":")
        ax.annotate("min %.2f" % gm, (gm, fcut * 0.85), fontsize=6,
                    color=S.GREY, ha="left")
    ax.set_xlabel(h[0])
    ax.set_ylabel("F (kJ/mol)")
    ax.set_ylim(0, fcut * 1.08)
    return True


def _panel_conv(ax, js):
    conv = js.get("convergence", []) or []
    fr, rc, rv = [], [], []
    for e in conv:
        if e.get("rms_centered") is None:
            continue
        fr.append(int(e.get("fraction", 0) * 100))
        rc.append(e["rms_centered"])
        rv.append(e.get("rms_vs_full", float("nan")))
    if not fr:
        ax.text(.5, .5, "no convergence data", transform=ax.transAxes, ha="center",
                va="center", color=S.GREY, fontsize=7)
        return ""
    xs = np.arange(len(fr))
    ax.bar(xs - 0.18, rc, width=0.34, color=C_A, label="rms_centered")
    ax.bar(xs + 0.18, rv, width=0.34, color=S.GREY, alpha=0.7, label="rms_vs_full")
    ax.set_xticks(xs)
    ax.set_xticklabels(["%d%%" % f for f in fr])
    ax.set_ylabel("RMS (kJ/mol)")
    ax.legend(loc="best")
    verdict = "converged" if rc[-1] <= max(1.0, 0.6 * max(rc)) else "NOT converged"
    return verdict


def fig_fes(ws, out_dir):
    h, r = _tbl(ws, "plumed_fes.csv")
    js = _json(ws, "analysis_plumed_fes.json") or {}
    if not h and not js:
        return False, "缺 plumed_fes.csv（先跑 analyze_fes；需要 HILLS 文件）"
    dims = int(js.get("dims", 1) or 1)
    bas = js.get("basins", {}) or {}
    fcut = float(bas.get("f_cut_kJmol", 200.0) or 200.0)
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 6.2), squeeze=False)
    axes = [a[0] for a in axes]

    ax = axes[0]
    ok = _panel_fes2d(ax, h, r, fcut, fig) if (h and dims >= 2) else _panel_fes1d(ax, h, r, fcut, bas)
    if not ok:
        ax.text(.5, .5, "no FES grid", transform=ax.transAxes, ha="center",
                va="center", color=S.GREY, fontsize=7)
    _title(ax, "PLUMED reweighted FES (well-tempered, %dD)" % dims)
    S.clean(ax)
    S.panel(ax, "A")

    ax = axes[1]
    verdict = _panel_conv(ax, js)
    _title(ax, "Block convergence (%s)" % (verdict or "n/a"))
    S.clean(ax, grid=True)
    S.panel(ax, "B")

    ax = axes[2]
    h3, r3 = _tbl(ws, "plumed_colvar.csv")
    if h3 and r3:
        t = S.col(h3, r3, "time")
        for c in h3[1:]:
            if c.startswith("metad") or "bias" in c.lower():
                continue
            y = S.col(h3, r3, c)
            if y:
                ax.plot(t, y, lw=0.8, label=c)
        ax.legend(ncol=2, loc="best")
    S.clean(ax, grid=True)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("CV value")
    _title(ax, "Collective variable trajectory")
    S.panel(ax, "C")
    S.save(fig, _stem(ws, out_dir, "fes"))
    b = bas.get("barrier_between_minima_kJmol")
    extra = ""
    if b is not None:
        extra = "；双井间势垒 %.1f kJ/mol" % b
    return True, "元动力学 FES 已出图（收敛判定：%s）%s" % (verdict or "数据不足", extra)


# ================================================================ 10. A/B 体系对比
def _cmp_rows(h, r):
    out = []
    for row in r:
        d = {}
        for k, v in zip(h, row):
            d[k] = v
        if "delta_b_minus_a" in d:
            out.append(d)
    return out


def fig_compare(ws, out_dir):
    h, r = _tbl(ws, "compare_metrics.csv")
    rows = _cmp_rows(h, r) if h else []
    if not rows:
        return False, "缺 compare_metrics.csv（先跑 compare_systems）"
    lab = ["%s : %s" % (x["file"].replace(".csv", ""), x["column"]) for x in rows]
    dl = [float(x["delta_b_minus_a"]) for x in rows]
    lo = [float(x["ci95_lo"]) for x in rows]
    hi = [float(x["ci95_hi"]) for x in rows]
    sig = [int(float(x["significant"])) == 1 for x in rows]
    ma = [float(x["mean_a"]) for x in rows]
    mb = [float(x["mean_b"]) for x in rows]
    ys = list(range(len(rows)))[::-1]
    hgt = max(2.4, 0.36 * len(rows) + 1.3)
    nsig = sum(1 for v in sig if v)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, hgt), constrained_layout=True)
    ax = axes[0]
    for i, y in enumerate(ys):
        c = C_B if sig[i] else S.GREY
        ax.errorbar(dl[i], y, xerr=[[dl[i] - lo[i]], [hi[i] - dl[i]]],
                    fmt="o", ms=3.4, color=c, ecolor=c, elinewidth=0.8,
                    capsize=2.0, lw=0)
    ax.axvline(0, color=S.INK, lw=0.7, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(lab)
    ax.set_xlabel("delta (B - A), 95% CI")
    S.clean(ax, grid=True, grid_axis="x")
    _title(ax, "Difference (red = CI excludes 0)")
    S.panel(ax, "A")

    ax = axes[1]
    for i, y in enumerate(ys):
        ax.plot([ma[i], mb[i]], [y, y], color="#C8CDD1", lw=1.0, zorder=1)
    ax.scatter(ma, ys, s=14, color=C_A, zorder=2, label="A")
    ax.scatter(mb, ys, s=14, color=C_B, zorder=2, label="B")
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.set_xlabel("Mean value")
    ax.legend(loc="best")
    S.clean(ax, grid=True, grid_axis="x")
    _title(ax, "Means (A vs B)")
    S.panel(ax, "B")
    S.save(fig, _stem(ws, out_dir, "compare"))
    return True, "A/B 对比图已出图；%d/%d 个指标显著" % (nsig, len(rows))


# ================================================================ 11. 时序先后 (lead-lag)
def fig_leadlag(ws, out_dir):
    hl, rl = _tbl(ws, "leadlag_matrix.csv")
    hc, rc2 = _tbl(ws, "leadlag_corr.csv")
    if not hl and not hc:
        return False, "缺 leadlag_matrix.csv / leadlag_corr.csv（先跑 analyze_leadlag）"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True)
    names = [c for c in hl[1:]] if hl else []

    ax = axes[0]
    if hl:
        M = np.asarray([[float(v) for v in row[1:1 + len(names)]] for row in rl], dtype=float)
        S.heatmap(ax, M, cmap="RdBu_r", fig=fig, cbar_label="best lag (frames)")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90)
        ax.set_yticks(range(len(rl)))
        ax.set_yticklabels(names)
    _title(ax, "Lead-lag (positive = row leads col)")
    S.panel(ax, "A")

    ax = axes[1]
    if hc:
        M2 = np.asarray([[float(v) for v in row[1:1 + len(names)]] for row in rc2], dtype=float)
        S.heatmap(ax, M2, cmap="magma_r", vmin=0, vmax=1, fig=fig,
                  cbar_label="max |corr|")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90)
        ax.set_yticks(range(len(rc2)))
        ax.set_yticklabels(names)
    _title(ax, "Max cross-correlation strength")
    S.panel(ax, "B")
    S.save(fig, _stem(ws, out_dir, "leadlag"))
    return True, "lead-lag 矩阵已出图（%d 条序列）" % len(names)


# ================================================================ 12. 不确定性 (CI / ESS)
def _ess_of(v):
    """autocorr_ess 返回的是 {'tau':..,'ess':..} 字典，这里统一取标量 ESS。"""
    e = (v or {}).get("ess")
    if isinstance(e, dict):
        e = e.get("ess")
    try:
        return float(e)
    except (TypeError, ValueError):
        return float("nan")


def fig_uncertainty(ws, out_dir):
    js = _json(ws, "analysis_uncertainty.json") or {}
    ser = js.get("series", {}) or {}
    got = [(k, v) for k, v in ser.items() if isinstance(v, dict) and "mean" in v]
    if not got:
        return False, "缺 analysis_uncertainty.json（先跑 analyze_uncertainty）"
    lab = [k.split(":")[-1] for k, _v in got]
    mu = [v["mean"] for _k, v in got]
    lo = [v.get("ci95", {}).get("lo", v["mean"]) for _k, v in got]
    hi = [v.get("ci95", {}).get("hi", v["mean"]) for _k, v in got]
    ess = [_ess_of(v) for _k, v in got]
    ys = list(range(len(got)))[::-1]
    hgt = max(2.2, 0.36 * len(got) + 1.2)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, hgt), constrained_layout=True)
    ax = axes[0]
    for i, y in enumerate(ys):
        ax.errorbar(mu[i], y, xerr=[[mu[i] - lo[i]], [hi[i] - mu[i]]],
                    fmt="o", ms=3.4, color=C_A, ecolor=C_A, elinewidth=0.8,
                    capsize=2.0, lw=0)
    ax.set_yticks(ys)
    ax.set_yticklabels(lab)
    ax.set_xlabel("Mean with block-bootstrap 95% CI")
    S.clean(ax, grid=True, grid_axis="x")
    _title(ax, "Means and uncertainty")
    S.panel(ax, "A")

    ax = axes[1]
    ax.barh(ys, ess, color=[C_B if (e == e and e < 50) else C_A for e in ess], height=0.6)
    ax.axvline(50, color=S.INK, lw=0.7, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.set_xlabel("Effective sample size (ESS)")
    S.clean(ax, grid=True, grid_axis="x")
    _title(ax, "ESS (red < 50)")
    S.panel(ax, "B")
    S.save(fig, _stem(ws, out_dir, "uncertainty"))
    bad = sum(1 for e in ess if e == e and e < 50)
    return True, "CI/ESS 已出图；%d 条序列 ESS<50（均值不可信）" % bad


# ================================================================ 注册表与驱动
FAMILIES = [
    ("segments", fig_segments),
    ("switches", fig_switches),
    ("tm6", fig_tm6),
    ("ligand", fig_ligand),
    ("landscape", fig_landscape),
    ("coupling", fig_coupling),
    ("psn", fig_psn),
    ("leadlag", fig_leadlag),
    ("water", fig_water),
    ("fes", fig_fes),
    ("uncertainty", fig_uncertainty),
    ("compare", fig_compare),
]


def plot_all(ws, out_dir="figs", families=""):
    """按 family 依次出图。families 为空则全出；返回 {family: {'ok','msg','files'}}。"""
    S.setup_style()
    want = [x.strip() for x in str(families).split(",") if x.strip()] if families else None
    res = {}
    for name, fn in FAMILIES:
        if want and name not in want:
            continue
        try:
            ok, msg = fn(ws, out_dir)
        except Exception as e:
            ok, msg = False, "出图异常: %s" % e
        files = []
        if ok:
            stem = _stem(ws, out_dir, name)
            files = [stem + ".pdf", stem + ".png"]
        res[name] = {"ok": bool(ok), "msg": msg, "files": files}
    return res


# __APPEND__
