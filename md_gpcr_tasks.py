"""GPCR / 元动力学 高层分析任务（第 3/4/5 层）。

与 md_analysis_worker.py 分开的原因：
  worker 里的 Session、几何/统计助手（superpose、kabsch_rot、block_bootstrap_ci、
  autocorr_ess、write_csv、_sel_tensor、parse_plumed、monotonic_segments）都在单个
  append 里越堆越大、容易写坏；把任务级逻辑单独成文件，基础设施只导入不重写。

本模块只依赖 md_analysis_worker 的公共助手 + numpy/scipy/sklearn，不重复实现。
所有任务签名统一为 task_xxx(ws: str, p: dict) -> dict，由 worker 的 CLI 分发。
"""
import os
import subprocess

import numpy as np

import gpcr_defs as G
import md_analysis_worker as W

Session = W.Session
write_csv = W.write_csv
superpose = W.superpose
kabsch_rot = W.kabsch_rot
block_bootstrap_ci = W.block_bootstrap_ci
autocorr_ess = W.autocorr_ess
_sel_tensor = W._sel_tensor
_auto_stride = W._auto_stride
SEG_ORDER = W.SEG_ORDER


def read_csv_column(path, col):
    """按表头名读 CSV 的一列，返回 (表头列表, numpy 数组)。"""
    with open(path, encoding='utf-8', errors='replace') as f:
        head = f.readline().strip().split(',')
        if col not in head:
            raise RuntimeError('文件里没有列 ' + col + '；可用列 ' + str(head))
        k = head.index(col)
        vals = []
        for line in f:
            toks = line.strip().split(',')
            if len(toks) > k:
                try:
                    vals.append(float(toks[k]))
                except ValueError:
                    continue
    return head, np.asarray(vals, dtype=float)


def _find_series(ws, fname):
    """定位工作区里的 CSV：给了名字就按名字找，没给就取最新的一个。"""
    if fname:
        p = fname if os.path.isabs(fname) else os.path.join(ws, fname)
        return p if os.path.exists(p) else ''
    cands = W.glob.glob(os.path.join(ws, '*.csv'))
    return max(cands, key=os.path.getmtime) if cands else ''


def task_uncertainty(ws, p):
    """对一个标量时间序列做分块 bootstrap 置信区间 + 自相关时间/有效样本量。

    目的：MD 结论不能只报均值。时间序列有自相关，普通标准差会低估误差，
    必须给分块 bootstrap 的 CI 与 ESS（有效样本量）。默认读工作区里已有的
    CSV（microswitch_*.csv / tm6_opening.csv / ligand_retention.csv 等），
    series 形如 "文件名:列名"，也接受直接传列名在当前工作区所有 CSV 里找。
    """
    ser = p.get('series') or ''
    if isinstance(ser, str):
        ser = [x.strip() for x in ser.split(',') if x.strip()]
    n_block = int(p.get('n_block', 6) or 6)
    n_boot = int(p.get('n_boot', 2000) or 2000)
    items = []
    for spec in ser:
        items.append(tuple(spec.split(':', 1)) if ':' in spec else ('', spec))
    if not items:
        return {'ok': False, 'error': '未指定 series；形如 "microswitch_DRY.csv:3.49-3.50"'}
    return _uncertainty_body(ws, items, n_block, n_boot)


def _uncertainty_body(ws, items, n_block, n_boot):
    ser = {}
    for fname, col in items:
        path = _find_series(ws, fname)
        key = (fname or 'auto') + ':' + col
        if not path:
            ser[key] = {'error': '找不到该 CSV（请给文件名，如 tm6_opening.csv）'}
            continue
        head, v = read_csv_column(path, col)
        ser[key] = {'file': os.path.basename(path), 'n': int(v.size)}
        ser[key]['mean'] = float(v.mean())
        ser[key]['sd'] = float(v.std(ddof=1)) if v.size > 1 else 0.0
        ser[key]['ci95'] = block_bootstrap_ci(v, n_block, n_boot)
        ser[key]['ess'] = autocorr_ess(v)
    return {'ok': True, 'n_block': n_block, 'n_boot': n_boot, 'series': ser}


def _pca(Z, ncomp=10):
    """对 (F, D) 特征矩阵做 PCA，返回 (投影, 解释方差比, 成分向量)。"""
    mu = Z.mean(axis=0)
    Zc = Z - mu
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    var = (S ** 2) / max(1, Z.shape[0] - 1)
    ev = var / var.sum()
    proj = U * S
    return proj[:, :ncomp], ev[:ncomp], Vt[:ncomp]


def _tica(Z, lag, ncomp=5, ridge=1e-8):
    """numpy 版 tICA：解广义特征值问题 C(lag) v = lambda C(0) v。

    用 C0 白化把广义问题化为对称问题。两个必须的数值保护：
      (1) 脊正则：C0 的秩不超过 F-lag，当特征维数 D 接近帧数时它接近奇异，
          不加 ridge 会让白化把数值噪声放大到 lambda>1（非物理）；
      (2) 秩截断：只保留 C0 的主子空间，避免对 ~0 特征值做 1/sqrt。
    经验上调用方应先做 PCA 降维（D << F）再进这里。
    """
    A = Z[:-lag]
    B = Z[lag:]
    n = len(A)
    C0 = (A.T @ A + B.T @ B) / (2.0 * n)
    C1 = (A.T @ B) / float(n)
    Cs = 0.5 * (C1 + C1.T)
    d = C0.shape[0]
    C0 = C0 + ridge * np.trace(C0) / max(1, d) * np.eye(d)
    w, V = np.linalg.eigh(C0)
    keep = w > max(1e-10, 1e-6 * float(w.max()))
    if int(keep.sum()) < ncomp + 1:
        keep = np.zeros_like(w, dtype=bool)
        keep[np.argsort(w)[::-1][:min(d, ncomp + 5)]] = True
    Wt = V[:, keep] / np.sqrt(w[keep])
    M = Wt.T @ Cs @ Wt
    lam, U2 = np.linalg.eigh(M)
    order = np.argsort(lam)[::-1][:ncomp]
    return lam[order], Wt @ U2[:, order]


def its_of(lam, lag_ns):
    """隐含时间尺度 ITS_i = -lag / ln(lambda_i(lag))，单位与 lag_ns 一致（ns）。

    对每个 lag 各算一次、再画在一起，就是标准的 ITS 收敛图：
    曲线变平的位置说明该 lag 以上模型已收敛。
    """
    out = []
    for v in lam:
        v = float(v)
        if v >= 1.0:
            v = 1.0 - 1e-12
        out.append(-lag_ns / np.log(v) if v > 0 else float('nan'))
    return out


def _fes_2d(x, y, bins=80, kt=2.5165, pad=0.6, bandwidth=0.35):
    """KDE 估计的 2D 自由能面 F = -kT ln(P/Pmax)，单位 kJ/mol，最低点归零。

    注意：这是轨迹投影的直方图/KDE 估计，**不是**重加权 FES，报告里必须写清。
    kT 默认取 303.15 K（= 2.5165 kJ/mol），与 metadynamics 的温度一致。
    """
    from sklearn.neighbors import KernelDensity
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dx = max(1e-6, float(x.max() - x.min())) * pad
    dy = max(1e-6, float(y.max() - y.min())) * pad
    xg = np.linspace(x.min() - dx, x.max() + dx, bins)
    yg = np.linspace(y.min() - dy, y.max() + dy, bins)
    XX, YY = np.meshgrid(xg, yg)
    grid = np.vstack([XX.ravel(), YY.ravel()]).T
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(np.vstack([x, y]).T)
    F = (-kt * kde.score_samples(grid))
    F = F - F.min()
    return xg, yg, F.reshape(XX.shape)


def _state_stats(labels, times):
    """状态占比 / dwell 次数与时长 / 转移计数矩阵。labels 为逐帧整数标签。"""
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    uniq = sorted(set(int(v) for v in labels.tolist()))
    occ = {}
    for s in uniq:
        occ[str(s)] = float(np.mean(labels == s))
    runs = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and labels[j + 1] == labels[i]:
            j += 1
        runs.setdefault(int(labels[i]), []).append(j - i + 1)
        i = j + 1
    dt = float(np.mean(np.diff(times))) if n > 1 else 1.0
    dwell = {}
    for s in uniq:
        r = runs.get(s, [1])
        dwell[str(s)] = {'n_dwell': len(r)}
        dwell[str(s)]['mean_ns'] = float(np.mean(r) * dt)
        dwell[str(s)]['max_ns'] = float(np.max(r) * dt)
    nst = (max(uniq) + 1) if uniq else 0
    T = np.zeros((nst, nst))
    for a, b in zip(labels[:-1], labels[1:]):
        T[int(a), int(b)] += 1
    return {'occupancy': occ, 'dwell': dwell, 'transitions': T, 'dt_ns': dt}


def task_landscape(ws, p):
    """构象景观：PC/tICA + lag 扫描与 ITS + 聚类 + KDE 的 FES + 状态占比/dwell/转移矩阵。

    特征 = TM1-TM7 的 Cα 坐标（先按核心段去刚体）。聚类在头几个 tIC 上做。
    FES 是轨迹投影的 KDE 估计，**不是重加权 FES**；要重加权请走 plumed_fes。
    """
    s = Session(ws, p.get('top', ''), p.get('traj', ''))
    stride = _auto_stride(s, p, 2500)
    core = s.prof.get('tm_core_align', ['TM1', 'TM2', 'TM3', 'TM4'])
    fit_res = set(s.seg_residues(core))
    feat_res = s.tm_residues()
    if len(feat_res) < 20:
        feat_res = s.chain_residues()
    d = s.ca_tensor(feat_res, stride)
    X, T, res = d['x'], d['t'], d['res']
    fit_mask = np.array([r in fit_res for r in res])
    if fit_mask.sum() < 10:
        fit_mask = np.ones(len(res), dtype=bool)
    Y = superpose(X, fit_mask)
    F, N, _ = Y.shape
    feat = Y.reshape(F, N * 3).astype(float)
    ncomp = int(p.get('ncomp', 10) or 10)
    proj, ev, _vecs = _pca(feat, ncomp)
    kt = float(p.get('kt', 2.5165) or 2.5165)
    dt_ns = float(np.mean(np.diff(T))) if F > 1 else 1.0
    npc = int(p.get('tica_pre_pcs', 0) or 0)
    if npc <= 0:
        npc = max(3, min(30, F // 3))
    npc = min(npc, proj.shape[1])
    z = proj[:, :npc] / np.maximum(proj[:, :npc].std(axis=0), 1e-12)
    lags = [1, 2, 5, 10, 20, 50, 100]
    lags = [l for l in lags if l < max(3, F // 3)]
    if not lags:
        lags = [1]
    its = {}
    for lg in lags:
        lam, _v = _tica(z, lg, ncomp=5)
        its[str(lg)] = its_of(lam, dt_ns * lg)
    best_lag = lags[min(len(lags) - 1, 2)]
    lam_used, vecs = _tica(z, best_lag, ncomp=5)
    tproj = z @ vecs
    k = int(p.get('n_states', 4) or 4)
    mode = str(p.get('cluster', 'kmeans') or 'kmeans')
    space = str(p.get('cluster_space', 'tica') or 'tica')
    base = tproj if space == 'tica' else proj
    nd = int(p.get('cluster_dims', 3) or 3)
    nbf = max(1, min(nd, base.shape[1]))
    Xc = base[:, :nbf]
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    if mode == 'gmm':
        mdl = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
        labels = mdl.fit_predict(Xc)
    else:
        mdl = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = mdl.fit_predict(Xc)
    st = _state_stats(labels, T)
    nbin = int(p.get('fes_bins', 80) or 80)
    xg, yg, Fes = _fes_2d(proj[:, 0], proj[:, 1], bins=nbin, kt=kt)
    xt, yt, Fes_t = _fes_2d(tproj[:, 0], tproj[:, 1], bins=nbin, kt=kt)
    files = []
    write_csv(os.path.join(ws, 'landscape_projection.csv'),
              ['time_ns', 'PC1', 'PC2', 'tIC1', 'tIC2', 'state'],
              list(zip(T, proj[:, 0], proj[:, 1], tproj[:, 0], tproj[:, 1], labels)))
    files.append('landscape_projection.csv')
    write_csv(os.path.join(ws, 'landscape_states.csv'),
              ['state', 'occupancy', 'n_dwell', 'mean_dwell_ns', 'max_dwell_ns'],
              [[kk, st['occupancy'].get(str(kk), 0.0)] + [st['dwell'].get(str(kk), {}).get(g, 0.0)
               for g in ('n_dwell', 'mean_ns', 'max_ns')] for kk in range(k)])
    files.append('landscape_states.csv')
    frows = []
    for i in range(len(yg)):
        for j in range(len(xg)):
            frows.append([float(xg[j]), float(yg[i]), float(Fes[i, j])])
    write_csv(os.path.join(ws, 'landscape_fes_pc.csv'), ['PC1', 'PC2', 'F_kJmol'], frows)
    files.append('landscape_fes_pc.csv')
    ftrows = []
    for i in range(len(yt)):
        for j in range(len(xt)):
            ftrows.append([float(xt[j]), float(yt[i]), float(Fes_t[i, j])])
    write_csv(os.path.join(ws, 'landscape_fes_tica.csv'), ['tIC1', 'tIC2', 'F_kJmol'], ftrows)
    files.append('landscape_fes_tica.csv')
    Tt = st['transitions']
    ns = Tt.shape[0]
    P = np.zeros_like(Tt)
    den = np.maximum(Tt.sum(axis=1, keepdims=True), 1.0)
    np.divide(Tt, den, out=P)
    thead = ['from'] + ['P_to_%d' % j for j in range(ns)] + ['N_to_%d' % j for j in range(ns)]
    trows = []
    for i in range(ns):
        pv = [float(P[i, j]) for j in range(ns)]
        nv = [float(Tt[i, j]) for j in range(ns)]
        trows.append([i] + pv + nv)
    write_csv(os.path.join(ws, 'landscape_transition.csv'), thead, trows)
    files.append('landscape_transition.csv')
    its_min = {}
    for lg in its:
        fin = [v for v in its[lg] if np.isfinite(v)]
        its_min[lg] = float(min(fin)) if fin else float('nan')
    conf = {}
    conf['method'] = mode
    conf['space'] = space
    conf['n_states'] = k
    conf['tica_lag_frames'] = int(best_lag)
    conf['time_span_ns'] = [float(T[0]), float(T[-1])] if F else []
    evr = [float(v) for v in ev]
    lamv = [float(v) for v in lam_used]
    return {'ok': True, 'stride': stride, 'n_frames_used': int(F), 'n_features': int(N * 3),
            'n_ca': int(N), 'explained_variance_ratio': evr, 'tica_eigenvalues': lamv,
            'its_ns_by_lag': its, 'its_shortest_ns_by_lag': its_min,
            'config': conf, 'states': st, 'files': files,
            'caveat': 'FES 是轨迹投影的 KDE 估计，不是重加权 FES；重加权请用 plumed_fes'}


def _gauss_mi(D):
    """高斯估计的互信息矩阵（nats）。D 为 (F, N, 3) 的位移张量。

    对每对残基把两个 3 维位移拼成 6 维联合高斯：
        MI = 0.5 * ln( det(Si) * det(Sj) / det(Sij) )
    全对可以在 numpy 里一次算完（N=200 时约 2 万对，毫秒级）。
    注意：这是**高斯/线性**估计器，抓不到真正的高阶非线性依赖；
    需要真非线性 MI 时改用 mi_subset 走 sklearn 的 mutual_info_regression。
    """
    F, N, _ = D.shape
    C = np.einsum('fni,fnj->nij', D, D) / max(1, F - 1)
    reg = 1e-6 * float(np.trace(C, axis1=1, axis2=2).mean())
    C = C + reg * np.eye(3)[None, :, :]
    iu = np.triu_indices(N, 1)
    ia, ib = iu[0], iu[1]
    A = C[ia]
    B = C[ib]
    Xij = np.einsum('fni,fnj->nij', D[:, ia, :], D[:, ib, :]) / max(1, F - 1)
    P = len(ia)
    J = np.zeros((P, 6, 6))
    J[:, :3, :3] = A
    J[:, 3:, 3:] = B
    J[:, :3, 3:] = Xij
    J[:, 3:, :3] = np.transpose(Xij, (0, 2, 1))
    da = np.maximum(np.linalg.det(A), 1e-30)
    db = np.maximum(np.linalg.det(B), 1e-30)
    dj = np.maximum(np.linalg.det(J), 1e-30)
    mi = 0.5 * np.log(da * db / dj)
    M = np.zeros((N, N))
    M[ia, ib] = mi
    M = M + M.T
    return M


def task_dccm_mi(ws, p):
    """相关运动：DCCM（Cα 位移矢量相关性，线性）+ 互信息 MI（高斯估计）。

    DCCM 抓线性耦合，MI 抓协方差意义上的协同；两者一起看才不容易漏掉
    "不同步但相关"的运动。输出两个矩阵 CSV + 最强耦合对排名。
    """
    s = Session(ws, p.get('top', ''), p.get('traj', ''))
    stride = _auto_stride(s, p, 1500)
    core = s.prof.get('tm_core_align', ['TM1', 'TM2', 'TM3', 'TM4'])
    fit_res = set(s.seg_residues(core))
    feat_res = s.tm_residues() or s.chain_residues()
    d = s.ca_tensor(feat_res, stride)
    X, T, res = d['x'], d['t'], d['res']
    fit_mask = np.array([r in fit_res for r in res])
    if fit_mask.sum() < 10:
        fit_mask = np.ones(len(res), dtype=bool)
    Y = superpose(X, fit_mask)
    F, N, _ = Y.shape
    D = Y - Y.mean(axis=0)
    flat = D.reshape(F, N * 3)
    fc = flat - flat.mean(axis=0)
    sd = np.maximum(fc.std(axis=0), 1e-12)
    z = fc / sd
    Ct = (z.T @ z) / max(1, F - 1)
    dccm = Ct.reshape(N, 3, N, 3).mean(axis=(1, 3))
    mi = _gauss_mi(D)
    header = ['resnum'] + [str(int(r)) for r in res]
    drow = []
    for i in range(N):
        drow.append([int(res[i])] + [float(dccm[i, j]) for j in range(N)])
    write_csv(os.path.join(ws, 'dccm.csv'), header, drow)
    mrow = []
    for i in range(N):
        mrow.append([int(res[i])] + [float(mi[i, j]) for j in range(N)])
    write_csv(os.path.join(ws, 'mutual_information.csv'), header, mrow)
    segs = s.prof.get('segments', {})
    an = s.prof.get('anchors', {})
    ra = np.asarray(res)
    iu = np.triu_indices(N, 1)
    ia, ib = iu[0], iu[1]
    min_sep = int(p.get('min_seq_sep', 10) or 10)
    keep = np.abs(ra[ia] - ra[ib]) > min_sep
    dvals = dccm[ia, ib]
    mvals = mi[ia, ib]
    o_d = np.argsort(-np.abs(np.where(keep, dvals, 0.0)))[:15]
    o_m = np.argsort(-np.where(keep, mvals, -1e9))[:15]
    top_d = []
    top_m = []
    for k in o_d:
        r1, r2 = int(ra[ia[k]]), int(ra[ib[k]])
        top_d.append({'res1': r1, 'res2': r2, 'dccm': float(dvals[k])})
        top_d[-1]['bw1'] = G.bw_label(r1, an, segs) or ''
        top_d[-1]['bw2'] = G.bw_label(r2, an, segs) or ''
    for k in o_m:
        r1, r2 = int(ra[ia[k]]), int(ra[ib[k]])
        top_m.append({'res1': r1, 'res2': r2, 'mi_nats': float(mvals[k])})
        top_m[-1]['bw1'] = G.bw_label(r1, an, segs) or ''
        top_m[-1]['bw2'] = G.bw_label(r2, an, segs) or ''
    seg_list = [k for k in SEG_ORDER if k in segs]
    seg_idx = {}
    for k in seg_list:
        seg_idx[k] = [i for i, r in enumerate(res) if segs[k][0] <= r <= segs[k][1]]
    srows = []
    for a_ in seg_list:
        row = [a_]
        for b_ in seg_list:
            ii, jj = seg_idx[a_], seg_idx[b_]
            row.append(float(np.abs(dccm[np.ix_(ii, jj)]).mean()) if ii and jj else 0.0)
        srows.append(row)
    write_csv(os.path.join(ws, 'segment_coupling.csv'), ['segment'] + seg_list, srows)
    pairs = []
    for ai, a_ in enumerate(seg_list):
        for bi, b_ in enumerate(seg_list):
            if bi > ai:
                pairs.append((srows[ai][bi + 1], a_, b_))
    pairs.sort(reverse=True)
    top_seg = []
    for v, a_, b_ in pairs[:8]:
        top_seg.append({'seg1': a_, 'seg2': b_, 'mean_abs_dccm': float(v)})
    cav = 'MI 为高斯估计，非高阶非线性 MI；DCCM 只反映线性相关'
    return {'ok': True, 'stride': stride, 'n_frames_used': int(F), 'n_ca': int(N),
            'residues': [int(v) for v in res], 'top_dccm_pairs': top_d,
            'top_mi_pairs': top_m, 'mean_abs_dccm_offdiag': float(np.abs(dvals).mean()),
            'mean_mi_nats': float(mvals.mean()), 'caveat': cav,
            'top_segment_coupling': top_seg, 'min_seq_sep': min_sep,
            'files': ['dccm.csv', 'mutual_information.csv', 'segment_coupling.csv']}


def _hills_subset(path, out_path, n_rows):
    """把 HILLS 截断到前 n_rows 个 hill（保留 #! 头），写到 out_path，用于分块收敛。"""
    kept = 0
    with open(path, encoding='utf-8', errors='replace') as fi:
        with open(out_path, 'w', encoding='utf-8') as fo:
            for line in fi:
                s = line.lstrip()
                if s.startswith('#!'):
                    fo.write(line)
                    continue
                if s.startswith('#'):
                    continue
                if kept >= n_rows:
                    break
                fo.write(line)
                kept += 1
    return kept


PLUMED_BIN = W.PLUMED
PLUMED_LIB = os.path.join(os.path.dirname(os.path.dirname(PLUMED_BIN)), 'lib')


def _plumed_env():
    """PLUMED 二进制需要 LD_LIBRARY_PATH 才能找到 libplumedKernel.so，这里自动补上。"""
    e = dict(os.environ)
    cur = e.get('LD_LIBRARY_PATH', '')
    e['LD_LIBRARY_PATH'] = PLUMED_LIB + (':' + cur if cur else '')
    return e


def _sum_hills(hills_path, out_prefix, mins, maxs, bins):
    """调 plumed sum_hills 出 FES。--mintozero 把最低点归零。

    mins/maxs/bins 是每个 CV 的列表（1 维就传长度 1）。PLUMED 对 2 维会写多个
    文件（out_prefix.0/.1...），所以这里把候选文件名都找一遍。
    返回 (成功与否, 实际文件名, 日志尾部)。
    """
    cmd = [W.PLUMED, 'sum_hills', '--hills', hills_path, '--mintozero']
    cmd += ['--kt', '2.5165']
    for d_ in range(len(mins)):
        cmd += ['--min', '%.6f' % mins[d_], '--max', '%.6f' % maxs[d_], '--bin', str(bins[d_])]
    cmd += ['--outfile', out_prefix]
    r = subprocess.run(cmd, capture_output=True, env=_plumed_env())
    msg = r.stdout.decode('utf-8', 'replace') + r.stderr.decode('utf-8', 'replace')
    cand = [out_prefix] + sorted(W.glob.glob(out_prefix + '.*'))
    for c in cand:
        if os.path.exists(c) and os.path.getsize(c) > 0:
            return True, c, msg[-400:]
    return False, '', msg[-400:]


def task_plumed_fes(ws, p):
    """重加权自由能面：plumed sum_hills 从 HILLS 出 FES + 分块收敛 + 盆地/势垒。

    well-tempered 的偏置按 (T+dT)/dT 缩放，sum_hills 会读 HILLS 的 biasf 列自动处理。
    这是**重加权 FES**，与 landscape 里的轨迹 KDE 估计不是一回事。
    分块收敛：把 HILLS 依次截断到 25/50/75/100%，在同一网格上比较 FES 的 RMS 差。

    结论里必须写清局限：单 CV 的 1D PMF 不等于结合自由能；收敛性看 RMS 差
    随块长的下降趋势，以及最后两块是否已趋稳。
    """
    h_file = W.find_plumed(ws, p.get('hills', ''), 'HILLS')
    if not h_file:
        raise RuntimeError('找不到 HILLS，无法做重加权 FES（元动力学数据必须先有 HILLS）')
    d = W.parse_plumed(h_file)
    fh, H = d['fields'], d['data']
    if not H.size:
        raise RuntimeError('HILLS 解析为空：' + h_file)
    ncv = W.hills_ncv(fh)
    dims = max(1, min(2, ncv))
    nb = int(p.get('bins', 150) or 150)
    bins = [nb] * dims
    mins = []
    maxs = []
    for k in range(dims):
        v = H[:, 1 + k]
        m = 0.05 * max(1e-9, float(v.max() - v.min()))
        mins.append(float(v.min()) - m)
        maxs.append(float(v.max()) + m)
    tmpdir = os.path.join(ws, 'plumed_fes_tmp')
    os.makedirs(tmpdir, exist_ok=True)
    out_full = os.path.join(ws, 'plumed_fes.dat')
    ok, real, log = _sum_hills(h_file, out_full, mins, maxs, bins)
    if not ok:
        raise RuntimeError('plumed sum_hills 失败：' + log)
    fdp = W.parse_plumed(real)
    FF = fdp['data']
    flds = list(fdp['fields'])
    if not flds:
        flds = ['cv1', 'F'] if FF.shape[1] == 2 else ['cv1', 'cv2', 'F']
    # sum_hills 的 1D 输出是 3 列：cv1, file.free, der_cv1 —— 最后那列是**导数**，
    # 绝不能当自由能用；按列名取，取不到再退回第 2 列。
    if 'file.free' in flds:
        fcol = flds.index('file.free')
    elif FF.shape[1] >= 2:
        fcol = 1
    else:
        fcol = FF.shape[1] - 1
    fv = FF[:, fcol]
    ref = fv - fv.min()
    write_csv(os.path.join(ws, 'plumed_fes.csv'), flds, list(FF))
    files = ['plumed_fes.csv']
    fcut = float(p.get('f_max_kJmol', 200.0) or 200.0)
    n_h = int(H.shape[0])
    conv = []
    for fr in [0.25, 0.5, 0.75]:
        cut = max(2, int(n_h * fr))
        tmp_h = os.path.join(tmpdir, 'subset_hills_%d' % int(fr * 100))
        _hills_subset(h_file, tmp_h, cut)
        o2 = os.path.join(tmpdir, 'fes_%d.dat' % int(fr * 100))
        ok2, real2, log2 = _sum_hills(tmp_h, o2, mins, maxs, bins)
        if not ok2:
            conv.append({'fraction': fr, 'n_hills': cut})
            conv[-1]['error'] = log2[-200:]
            continue
        f2 = W.parse_plumed(real2)['data'][:, fcol]
        nn = min(len(f2), len(ref))
        f2 = f2[:nn]
        r2 = ref[:nn]
        m2 = (r2 <= fcut) & (f2 <= fcut)
        if int(m2.sum()) < 3:
            m2 = r2 <= fcut
        if not m2.any():
            m2 = np.ones(nn, dtype=bool)
        df = f2[m2] - r2[m2]
        e = {'fraction': fr, 'n_hills': cut, 'n_grid_used': int(m2.sum())}
        e['rms_vs_full'] = float(np.sqrt((df ** 2).mean()))
        e['rms_centered'] = float(np.sqrt(((df - df.mean()) ** 2).mean()))
        e['max_vs_full'] = float(np.abs(df).max())
        conv.append(e)
    bas = {}
    if dims == 1:
        idx = np.where(ref <= fcut)[0]
        if idx.size < 5:
            idx = np.arange(len(ref))
        cvm = FF[idx, 0]
        fvm = ref[idx]
        win = np.ones(5) / 5.0
        sm = np.convolve(fvm, win, mode='same')
        loc = []
        for i in range(2, len(sm) - 2):
            if sm[i] <= sm[i - 1] and sm[i] <= sm[i + 1]:
                loc.append(i)
        loc.sort(key=lambda i: fvm[i])
        bas['f_cut_kJmol'] = fcut
        bas['n_local_minima'] = len(loc)
        bas['cv1_at_global_min'] = float(cvm[int(np.argmin(fvm))])
        bas['F_global_min_kJmol'] = float(fvm.min())
        bas['sampled_cv_range'] = [float(cvm[0]), float(cvm[-1])]
        bas['sampled_grid_fraction'] = float(idx.size / max(1, len(ref)))
        if len(loc) >= 2:
            i1, i2 = loc[0], loc[1]
            lo_i = min(i1, i2)
            hi_i = max(i1, i2)
            seg = fvm[lo_i:hi_i + 1]
            bas['cv1_at_second_min'] = float(cvm[i2])
            bas['dF_min2_minus_min1_kJmol'] = float(fvm[i2] - fvm[i1])
            bas['barrier_between_minima_kJmol'] = float(seg.max() - seg.min())
        bas['F_at_sampled_low_end_kJmol'] = float(fvm[0])
        bas['F_at_sampled_high_end_kJmol'] = float(fvm[-1])
    grid = {'mins': mins, 'maxs': maxs, 'bins': bins}
    prof = []
    if dims == 1:
        idxp = np.where(ref <= fcut)[0]
        if idxp.size < 4:
            idxp = np.arange(len(ref))
        n_show = min(12, int(idxp.size))
        sel = np.unique(idxp[np.linspace(0, idxp.size - 1, n_show).astype(int)])
        for i in sel:
            prof.append({'cv1': float(FF[i, 0]), 'F_kJmol': float(ref[i])})
    cav = ('单 CV 的 1D PMF 不等于结合自由能；必须先看 convergence 里 '
           'rms_centered 是否随块长趋稳，未收敛时 fes_profile 的深井不可当作定量结论')
    return {'ok': True, 'hills_file': os.path.basename(h_file), 'n_hills': n_h, 'n_cv': ncv,
            'fields': fh, 'dims': dims, 'grid': grid, 'convergence': conv,
            'basins': bas, 'fes_profile': prof, 'files': files, 'caveat': cav}


def _betweenness(adj, n):
    """Brandes 无权介数中心性。adj 为 (n,n) 布尔邻接矩阵。

    无 networkx 依赖，纯 numpy/列表实现；n~200 时 200 次 BFS 毫秒级。
    """
    bc = np.zeros(n)
    for s in range(n):
        S = []
        P = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1.0
        dist = np.full(n, -1)
        dist[s] = 0
        Q = [s]
        while Q:
            v = Q.pop(0)
            S.append(v)
            for w in np.where(adj[v])[0]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    Q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta = np.zeros(n)
        while S:
            w = S.pop()
            for v in P[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]
    return bc / 2.0


def _label_prop(adj, n, iters=100, seed=0):
    """标签传播社区检测（numpy，确定性：平局取最小标签）。返回 0..k-1 标签。"""
    rng = np.random.default_rng(seed)
    lab = np.arange(n)
    for _ in range(iters):
        changed = 0
        for v in rng.permutation(n):
            nb = np.where(adj[v])[0]
            if nb.size == 0:
                continue
            vals, cnt = np.unique(lab[nb], return_counts=True)
            best = int(vals[cnt == cnt.max()].min())
            if lab[v] != best:
                lab[v] = best
                changed += 1
        if changed == 0:
            break
    uniq = {v: i for i, v in enumerate(sorted(set(int(x) for x in lab)))}
    return np.array([uniq[int(x)] for x in lab], dtype=int)


def _match_labels(a, b, k):
    """把两套社区标签对齐（最大化重合），返回重标号后的 b。

    用 scipy 的匈牙利算法在重合度矩阵上做最优匹配，避免"标签号相同但社区不同"的假重排。
    """
    from scipy.optimize import linear_sum_assignment
    ka = int(a.max()) + 1 if a.size else 0
    kb = int(b.max()) + 1 if b.size else 0
    kk = max(ka, kb, 1)
    M = np.zeros((kk, kk))
    for i in range(len(a)):
        M[int(a[i]), int(b[i])] += 1.0
    r, c = linear_sum_assignment(-M)
    mp = {int(c[i]): int(r[i]) for i in range(len(c))}
    return np.array([mp.get(int(x), int(x)) for x in b], dtype=int)


def task_psn(ws, p):
    """蛋白结构网络（PSN）：接触图 + 介数中心性 + 社区检测 + 社区重排。

    Cα-Cα 距离 < cut 记一条边；先在整段轨迹的平均接触频率图上做介数与标签传播
    社区检测；再把轨迹分成前半/后半各自建图，用匈牙利算法对齐社区标签后找出
    "换社区"的残基 —— 这就是 community rewiring。纯 numpy/scipy，无 networkx 依赖。
    """
    s = Session(ws, p.get('top', ''), p.get('traj', ''))
    stride = _auto_stride(s, p, 1500)
    core = s.prof.get('tm_core_align', ['TM1', 'TM2', 'TM3', 'TM4'])
    fit_res = set(s.seg_residues(core))
    feat_res = s.tm_residues() or s.chain_residues()
    d = s.ca_tensor(feat_res, stride)
    X, T, res = d['x'], d['t'], d['res']
    fit_mask = np.array([r in fit_res for r in res])
    if fit_mask.sum() < 10:
        fit_mask = np.ones(len(res), dtype=bool)
    Y = superpose(X, fit_mask)
    F, N, _ = Y.shape
    cut = float(p.get('contact_cutoff_nm', 0.8) or 0.8)
    Q = np.zeros((N, N))
    Q1 = np.zeros((N, N))
    Q2 = np.zeros((N, N))
    half = max(1, F // 2)
    for f in range(F):
        P = Y[f]
        dd = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        c = (dd < cut).astype(float)
        np.fill_diagonal(c, 0.0)
        Q += c
        if f < half:
            Q1 += c
        else:
            Q2 += c
    Q /= float(F)
    Q1 /= float(half)
    Q2 /= float(max(1, F - half))
    eth = float(p.get('edge_freq', 0.5) or 0.5)
    adj = Q >= eth
    np.fill_diagonal(adj, False)
    deg = adj.sum(axis=1).astype(float)
    bc = _betweenness(adj, N)
    lab = _label_prop(adj, N)
    adj1 = Q1 >= eth
    adj2 = Q2 >= eth
    np.fill_diagonal(adj1, False)
    np.fill_diagonal(adj2, False)
    l1 = _label_prop(adj1, N)
    l2 = _label_prop(adj2, N)
    l2m = _match_labels(l1, l2, N)
    rewired = (l1 != l2m)
    segs = s.prof.get('segments', {})
    an = s.prof.get('anchors', {})
    seg_of = {}
    for kk in SEG_ORDER:
        if kk in segs:
            for rr in range(segs[kk][0], segs[kk][1] + 1):
                seg_of[rr] = kk
    rows = []
    for i in range(N):
        r = int(res[i])
        bwv = G.bw_label(r, an, segs) or ''
        rw = 1 if rewired[i] else 0
        row = [r, bwv, seg_of.get(r, 'other')]
        row += [float(deg[i]), float(bc[i]), int(lab[i]), int(l1[i]), int(l2m[i]), rw]
        rows.append(row)
    hd = ['resnum', 'bw', 'segment', 'degree', 'betweenness', 'community']
    hd += ['community_first_half', 'community_second_half', 'rewired']
    write_csv(os.path.join(ws, 'psn_residues.csv'), hd, rows)
    hrows = []
    iu = np.triu_indices(N, 1)
    for a_, b_ in zip(iu[0], iu[1]):
        if Q[a_, b_] >= eth:
            hrows.append([int(res[a_]), int(res[b_]), float(Q[a_, b_])])
    write_csv(os.path.join(ws, 'psn_contacts.csv'),
              ['resnum_i', 'resnum_j', 'contact_frequency'], hrows)
    rew = []
    for i in np.where(rewired)[0]:
        r = int(res[i])
        e = {'resnum': r, 'bw': G.bw_label(r, an, segs) or ''}
        e['segment'] = seg_of.get(r, 'other')
        e['from_community'] = int(l1[i])
        e['to_community'] = int(l2m[i])
        rew.append(e)
    top_bc = []
    for i in np.argsort(-bc)[:12]:
        r = int(res[i])
        e = {'resnum': r, 'bw': G.bw_label(r, an, segs) or ''}
        e['segment'] = seg_of.get(r, 'other')
        e['betweenness'] = float(bc[i])
        e['degree'] = float(deg[i])
        top_bc.append(e)
    ncom = {}
    for x in lab.tolist():
        k_ = str(int(x))
        ncom[k_] = ncom.get(k_, 0) + 1
    files = ['psn_residues.csv', 'psn_contacts.csv']
    cav = 'PSN 用整段平均接触图；重排是前半/后半粗时间分辨率对比，非逐帧'
    return {'ok': True, 'stride': stride, 'n_frames_used': int(F), 'n_ca': int(N),
            'contact_cutoff_nm': cut, 'edge_freq_threshold': eth,
            'n_edges': int(adj.sum() // 2), 'n_communities': int(lab.max()) + 1,
            'community_sizes': ncom, 'top_betweenness': top_bc,
            'n_rewired': len(rew), 'rewired_residues': rew[:40],
            'files': files, 'caveat': cav}


def _detrend_z(x):
    """去线性趋势再 z 分数化（lead-lag 之前必须做，否则共同漂移会伪装成相关）。"""
    n = x.size
    t = np.arange(n, dtype=float)
    A = np.vstack([t, np.ones(n)]).T
    coef, _r, _rk, _s = np.linalg.lstsq(A, x, rcond=None)
    r = x - A @ coef
    s = float(r.std())
    return (r - r.mean()) / (s if s > 1e-12 else 1.0)


def _xcorr(a, b, lag):
    """a(t) 与 b(t+lag) 的 Pearson 相关；lag>0 表示 a 领先 b。"""
    if lag > 0:
        aa, bb = a[:a.size - lag], b[lag:]
    elif lag < 0:
        aa, bb = a[-lag:], b[:b.size + lag]
    else:
        aa, bb = a, b
    if aa.size < 10:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def task_leadlag(ws, p):
    """激活坐标之间的先后顺序：去线性趋势 + z 分数后的互相关，取 argmax lag。

    正 lag 表示 A 领先 B（A 先动）。用来回答"谁先动"——例如 TM6 先外移还是
    DRY 盐桥先断、配体先动还是微开关先动。系列必须等长且来自同一 stride。
    """
    ser = p.get('series') or ''
    if isinstance(ser, str):
        ser = [x.strip() for x in ser.split(',') if x.strip()]
    if not ser:
        ser = ['microswitch_PIF.csv:3.40-6.44', 'microswitch_DRY.csv:3.49-3.50',
               'microswitch_CWxP.csv:6.48 chi1', 'microswitch_NPxxY.csv:7.50-7.53',
               'tm6_opening.csv:TM6ic_TM3ic_nm', 'ligand_retention.csv:ligand_rmsd_nm']
    items = []
    for spec in ser:
        items.append(tuple(spec.split(':', 1)) if ':' in spec else ('', spec))
    names = []
    cols = []
    dt_ns = 0.0
    for fname, col in items:
        path = _find_series(ws, fname)
        if not path:
            continue
        head, v = read_csv_column(path, col)
        if v.size < 20:
            continue
        if 'time_ns' in head and dt_ns == 0.0:
            _h2, tt = read_csv_column(path, 'time_ns')
            if tt.size > 1:
                dt_ns = float(np.mean(np.diff(tt)))
        names.append(col)
        cols.append(v)
    if len(cols) < 2:
        raise RuntimeError('至少需要两条可读序列；先跑 microswitches / tm6_opening 等生成 CSV')
    sizes = sorted(set(int(c.size) for c in cols))
    if len(sizes) != 1:
        raise RuntimeError('序列长度不一致 %s；请用相同 stride 重新导出后再比较' % sizes)
    L = sizes[0]
    M = np.vstack([c.astype(float) for c in cols])
    max_lag = int(p.get('max_lag', 20) or 20)
    max_lag = max(1, min(max_lag, L // 4))
    K = len(names)
    Z = np.vstack([_detrend_z(M[i]) for i in range(M.shape[0])])
    lagmat = np.zeros((K, K))
    cmat = np.zeros((K, K))
    pairs = []
    for i in range(K):
        for j in range(K):
            if i == j:
                cmat[i, j] = 1.0
                continue
            best_l, best_c = 0, 0.0
            for lg in range(-max_lag, max_lag + 1):
                cc = _xcorr(Z[i], Z[j], lg)
                if abs(cc) > abs(best_c):
                    best_c, best_l = cc, lg
            lagmat[i, j] = float(best_l)
            cmat[i, j] = best_c
            if i < j:
                pairs.append((abs(best_c), best_c, best_l, i, j))
    pairs.sort(reverse=True)
    top = []
    for ac, cc, lg, i, j in pairs[:12]:
        e = {'a': names[i], 'b': names[j], 'abs_corr': float(ac), 'corr': float(cc)}
        e['best_lag_frames'] = int(lg)
        e['best_lag_ns'] = float(lg * dt_ns)
        e['leader'] = names[i] if lg > 0 else (names[j] if lg < 0 else 'none')
        e['lag_at_boundary'] = bool(abs(lg) >= max_lag - 1)
        top.append(e)
    n_bound = sum(1 for x in top if x['lag_at_boundary'])
    lrows = []
    crows = []
    for i in range(K):
        lrows.append([names[i]] + [int(lagmat[i, j]) for j in range(K)])
        crows.append([names[i]] + [float(cmat[i, j]) for j in range(K)])
    write_csv(os.path.join(ws, 'leadlag_matrix.csv'), ['series'] + names, lrows)
    write_csv(os.path.join(ws, 'leadlag_corr.csv'), ['series'] + names, crows)
    cav = ('正 lag 表示 a 领先 b；lag 分辨率 = 一帧。lag_at_boundary=true 说明最优 lag '
           '顶在搜索窗边缘，真实先后顺序未辨识，必须加大 max_lag 再看')
    return {'ok': True, 'n_points': int(L), 'dt_ns': float(dt_ns),
            'max_lag_frames': int(max_lag), 'series': names, 'top_leadlag': top,
            'n_boundary_lags': int(n_bound), 'lag_matrix': lagmat,
            'corr_matrix': cmat, 'caveat': cav,
            'files': ['leadlag_matrix.csv', 'leadlag_corr.csv']}


def _wet_session(s, ws):
    """再抽一条含水的子轨迹（Protein_LIG_Water）用于水网络分析，返回 Universe 或 None。

    worker 的 write_index 已把 Protein_LIG_Water 写进 wc.ndx；老工作区可能没这个组，
    检测到缺组就按当前拓扑重写一次 index。含水子轨迹单独存为 analysis_sol.xtc。
    """
    import MDAnalysis as mda
    ndx = os.path.join(ws, 'wc.ndx')
    labels = list(getattr(s, 'ndx_labels', []) or [])
    if (not os.path.exists(ndx)) or ('Protein_LIG_Water' not in labels):
        u_full = mda.Universe(s.top_full)
        _p, labels = W.write_index(ws, u_full, W.classify(u_full))
        s.ndx_labels = labels
    if 'Protein_LIG_Water' not in labels:
        return None
    gi = labels.index('Protein_LIG_Water')
    xtc = os.path.join(ws, 'analysis_sol.xtc')
    gro = os.path.join(ws, 'analysis_sol.gro')
    if not (os.path.exists(xtc) and os.path.exists(gro)):
        tpr = W.find_tpr(ws)
        src, pbc = (tpr, '-pbc mol') if tpr else (s.top_full, '-pbc whole')
        base = W._q(W.GMX) + ' trjconv -s ' + W._q(src) + ' -f ' + W._q(s.traj)
        base += ' -n ' + W._q(ndx) + ' ' + pbc
        sel = (('%d' % gi) + W.NL).encode()
        tx, tg = xtc + '.tmp.xtc', gro + '.tmp.gro'
        for p_ in (tx, tg, xtc, gro):
            if os.path.exists(p_):
                os.remove(p_)
        subprocess.run(base + ' -o ' + W._q(tx), shell=True, cwd=ws, input=sel, capture_output=True, timeout=7200)
        subprocess.run(base + ' -dump 0 -o ' + W._q(tg), shell=True, cwd=ws, input=sel, capture_output=True, timeout=3600)
        if (os.path.exists(tx) and os.path.getsize(tx) > 0
                and os.path.exists(tg) and os.path.getsize(tg) > 0):
            os.replace(tx, xtc)
            os.replace(tg, gro)
    if not (os.path.exists(xtc) and os.path.exists(gro)):
        return None
    import MDAnalysis as mda
    return mda.Universe(gro, xtc)


def task_water_network(ws, p):
    """水网络：核心位点的水占据率 + 配体-水-残基 水桥。

    位点取 GPCR 档案的 BW 锚点（即微开关位点）；水只统计每残基首原子（TIP3P 的氧）。
    「占据率」= 该位点最近重原子 cut 内有水的帧比例；「水桥」= 同一个水同时在配体和
    位点 cut 内。含水的子轨迹单独抽一次（analysis_sol.xtc），只含 Protein+LIG+Water。
    """
    s = Session(ws, p.get('top', ''), p.get('traj', ''))
    u = _wet_session(s, ws)
    if u is None:
        raise RuntimeError('抽不出含水的子轨迹 analysis_sol.xtc；请确认 wc.ndx 与 .tpr 可用')
    cut = float(p.get('water_cutoff_nm', 0.35) or 0.35)
    stride = _auto_stride(s, p, 400)
    an = s.prof.get('anchors', {})
    segs = s.prof.get('segments', {})
    sites = []
    for bw in sorted(an):
        sites.append((bw, int(an[bw])))
    site_res = [r for _bw, r in sites]
    by_res = {}
    for a in u.atoms:
        if a.name[0] != 'H':
            by_res.setdefault(int(a.resid), []).append(a.index)
    wres = sorted(set(r.resname for r in u.residues) & W.WATER)
    watag = u.select_atoms('resname ' + ' '.join(wres)) if wres else u.atoms[:0]
    seen = set()
    oxy = []
    for a in watag:
        if a.resindex not in seen:
            seen.add(a.resindex)
            oxy.append(a.index)
    lres = set(s.info['lig_resnames'])
    lig_sel = [a.index for a in u.atoms if a.resname in lres]
    n_frames = len(u.trajectory)
    frames = list(range(0, n_frames, max(1, stride)))
    site_atoms = [by_res.get(r, []) for r in site_res]
    oxy_sel = u.atoms[oxy]
    ligag = u.atoms[lig_sel] if lig_sel else None
    cnt = np.zeros((len(frames), len(sites)))
    brg = np.zeros(len(sites))
    TT = np.zeros(len(frames))
    for k, f in enumerate(frames):
        u.trajectory[f]
        TT[k] = float(u.trajectory.time) / 1000.0
        Wp = oxy_sel.positions * W.A2NM
        Lc = ligag.positions * W.A2NM if ligag is not None else None
        for si, ii in enumerate(site_atoms):
            if not ii:
                continue
            S = u.atoms[ii].positions * W.A2NM
            com = S.mean(axis=0)
            rmax = float(np.sqrt(((S - com) ** 2).sum(axis=1)).max())
            dc = np.sqrt(((Wp - com) ** 2).sum(axis=1))
            cand = np.where(dc < rmax + cut)[0]
            if cand.size == 0:
                continue
            sub = Wp[cand]
            dd = np.sqrt(((sub[:, None, :] - S[None, :, :]) ** 2).sum(-1))
            near = dd.min(axis=1) < cut
            cnt[k, si] = float(near.sum())
            if Lc is not None and bool(near.any()):
                dl = np.sqrt(((sub[near][:, None, :] - Lc[None, :, :]) ** 2).sum(-1))
                brg[si] += float((dl.min(axis=1) < cut).sum())
    hdr = ['time_ns'] + ['water_n_' + bw for bw, _r in sites]
    rows = []
    for k in range(len(frames)):
        rows.append([float(TT[k])] + [float(cnt[k, si]) for si in range(len(sites))])
    write_csv(os.path.join(ws, 'water_occupancy.csv'), hdr, rows)
    occ = (cnt > 0).mean(axis=0)
    summ = []
    for si, (bw, r) in enumerate(sites):
        e = {'site': bw, 'resnum': r, 'segment': 'other'}
        for kk in SEG_ORDER:
            if kk in segs and segs[kk][0] <= r <= segs[kk][1]:
                e['segment'] = kk
        e['mean_water_count'] = float(cnt[:, si].mean())
        e['max_water_count'] = float(cnt[:, si].max())
        e['occupancy'] = float(occ[si])
        e['water_bridge_fraction'] = float(brg[si] / max(1, len(frames)))
        summ.append(e)
    summ.sort(key=lambda x: -x['occupancy'])
    srows = []
    for x in summ:
        r0 = [x['site'], x['resnum'], x['segment']]
        r0 += [x['mean_water_count'], x['max_water_count']]
        r0 += [x['occupancy'], x['water_bridge_fraction']]
        srows.append(r0)
    wh = ['site', 'resnum', 'segment', 'mean_water_count', 'max_water_count']
    wh += ['occupancy', 'water_bridge_fraction']
    write_csv(os.path.join(ws, 'water_sites.csv'), wh, srows)
    cav = ('水只统计每残基首原子(TIP3P 氧)；占据率 = 至少一个水在 cut 内的帧比例；'
           '水桥要求同一个水同时在配体与位点的 cut 内')
    return {'ok': True, 'stride': stride, 'n_frames_used': int(len(frames)),
            'water_cutoff_nm': cut, 'n_water_oxygen': len(oxy),
            'n_ligand_atoms': len(lig_sel), 'n_sites': len(sites),
            'sites': summ, 'caveat': cav,
            'files': ['water_occupancy.csv', 'water_sites.csv']}


def _delta_ci(a, b, n_block=6, n_boot=2000, seed=11):
    """两样本分块 bootstrap 求差值 (mean_b - mean_a) 的 95% CI。

    significant = CI 不含 0。双体系对比必须看这个，否则两个跑得都不够久的体系
    之间任何差别都能被说成"显著"。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 4 or b.size < 4:
        return {}
    ba = np.array_split(a, min(n_block, a.size))
    bb = np.array_split(b, min(n_block, b.size))
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, len(ba), len(ba))
        ib = rng.integers(0, len(bb), len(bb))
        vals[i] = np.concatenate([bb[j] for j in ib]).mean() - np.concatenate([ba[j] for j in ia]).mean()
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return {'delta': float(b.mean() - a.mean()), 'lo': lo, 'hi': hi,
            'sd': float(vals.std(ddof=1)), 'significant': bool(lo > 0 or hi < 0)}


def task_compare(ws, p):
    """A/B 双体系对比：同一套指标在两个工作区之间做差值 + 分块 bootstrap 置信区间。

    A 就是当前工作区；B 用 ws_b 指定。指标形如
    "microswitch_DRY.csv:3.49-3.50,microswitch_CWxP.csv:6.48 chi1"；不给就用默认一套。
    会校验两个工作区的 GPCR 编号档案是否一致（锚点不同则明确警告）。
    """
    wsb = p.get('ws_b') or p.get('workspace_b') or ''
    if not wsb or not os.path.isdir(wsb):
        raise RuntimeError('需要 ws_b 指向 B 工作区，如 {"ws_b": "/data2/JRK/md_workspaces/8XZI"}')
    warn = []
    pa = G.load_profile(ws)
    pb = G.load_profile(wsb)
    if pa and pb:
        if pa.get('anchors') != pb.get('anchors'):
            warn.append('两个工作区的 BW 锚点不一致；对比前必须确认编号口径相同')
        if pa.get('seq_len') != pb.get('seq_len'):
            warn.append('两个工作区受体长度不同：%s vs %s' % (pa.get('seq_len'), pb.get('seq_len')))
    if not pa or not pb:
        warn.append('有一侧缺少 gpcr.json，先跑 detect_numbering 再对比')
    mets = p.get('metrics') or ''
    if isinstance(mets, str):
        mets = [x.strip() for x in mets.split(',') if x.strip()]
    if not mets:
        mets = ['microswitch_DRY.csv:3.49-3.50', 'microswitch_CWxP.csv:6.48 chi1',
                'microswitch_NPxxY.csv:7.50-7.53', 'microswitch_NPxxY.csv:7.53 chi1',
                'microswitch_PIF.csv:3.40-6.44', 'tm6_opening.csv:TM6ic_TM3ic_nm',
                'ligand_retention.csv:ligand_rmsd_nm', 'ligand_retention.csv:n_contact_atoms']
    items = []
    for spec in mets:
        items.append(tuple(spec.split(':', 1)) if ':' in spec else ('', spec))
    n_block = int(p.get('n_block', 6) or 6)
    n_boot = int(p.get('n_boot', 2000) or 2000)
    res = []
    missing = []
    for fname, col in items:
        fpa = _find_series(ws, fname)
        fpb = _find_series(wsb, fname)
        if not fpa or not fpb:
            missing.append(fname + ':' + col)
            continue
        try:
            _h, va = read_csv_column(fpa, col)
            _h, vb = read_csv_column(fpb, col)
        except Exception as ex:
            missing.append(fname + ':' + col + ' -> ' + str(ex)[:60])
            continue
        dd = _delta_ci(va, vb, n_block, n_boot)
        if not dd:
            missing.append(fname + ':' + col + ' -> 有效样本太少')
            continue
        e = {'file': fname, 'column': col, 'n_a': int(va.size), 'n_b': int(vb.size)}
        e['mean_a'] = float(va.mean())
        e['mean_b'] = float(vb.mean())
        e['delta_b_minus_a'] = dd['delta']
        e['ci95_lo'] = dd['lo']
        e['ci95_hi'] = dd['hi']
        e['significant'] = dd['significant']
        res.append(e)
    rows = []
    for x in res:
        r0 = [x['file'], x['column'], x['n_a'], x['n_b']]
        r0 += [x['mean_a'], x['mean_b'], x['delta_b_minus_a']]
        r0 += [x['ci95_lo'], x['ci95_hi'], 1 if x['significant'] else 0]
        rows.append(r0)
    hd = ['file', 'column', 'n_a', 'n_b', 'mean_a', 'mean_b']
    hd += ['delta_b_minus_a', 'ci95_lo', 'ci95_hi', 'significant']
    write_csv(os.path.join(ws, 'compare_metrics.csv'), hd, rows)
    nsig = sum(1 for x in res if x['significant'])
    if any(min(x['n_a'], x['n_b']) < 50 for x in res):
        warn.append('有指标序列短于 50 帧：分块 bootstrap 的 CI 会明显偏乐观；'
                    '此时的 significant 只说明这一小段时间内不同，不代表构象层面的结论')
    cav = ('显著 = 差值 95% CI 不含 0（分块 bootstrap）；两体系必须等长时间尺度才可比，'
           '否则先把两段轨迹截到相同长度再比')
    return {'ok': True, 'workspace_a': os.path.abspath(ws), 'workspace_b': os.path.abspath(wsb),
            'n_metrics': len(res), 'n_significant': nsig, 'metrics': res,
            'missing': missing, 'warnings': warn, 'caveat': cav,
            'files': ['compare_metrics.csv']}


# __APPEND__
