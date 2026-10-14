"""GPCR / 元动力学 重分析后端 worker。

在【分析 env(gmxMMPBSA: MDAnalysis 2.7 + scipy + sklearn)】里运行，被 agent venv
的 md_tools._py_analysis() 以子进程方式调用。

为什么单独一个 worker：
  1) MDAnalysis 2.7 读不了 GROMACS 2026 的 tpr(tpx version 138)，所以拓扑一律
     走 .gro/.pdb，坐标走 .xtc；
  2) agent venv 里没有 scipy/sklearn/MDAnalysis。重的几何与统计放这里，agent
     venv 只保留 matplotlib 与编排逻辑。

CLI: python md_analysis_worker.py <task> --ws <workspace> [--json '<payload>']
输出: stdout 单行 JSON，形如 {"ok": true, ...} 或 {"ok": false, "error": "..."}
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import gpcr_defs as G

GMX = os.environ.get("GMX_BIN", "/opt/gromacs/bin/gmx")
PLUMED = os.environ.get("PLUMED_BIN", "/data/JRK/main/plumed-2.10.1-install/bin/plumed")

WATER = {"SOL", "HOH", "WAT", "TIP3", "TIP3P", "TIP4P", "SPC", "SPCE", "DOD"}
IONS = {"NA", "CL", "K", "MG", "ZN", "CA", "SOD", "CLA", "POT", "CAL", "MG2",
        "CES", "LIT", "RUB", "BAR", "IOD", "BR"}
LIPIDS = {"POPC", "POPE", "POPS", "POPG", "POPI", "DPPC", "DMPC", "DLPC", "DOPC",
          "DOPE", "DOPG", "SOPC", "CHOL", "CHL1", "PALM", "OLEO", "STEA", "MYRI",
          "PC", "PE", "PS", "PA", "PG", "PI", "PIP", "PIP2", "PIP3", "CER",
          "SM", "GM1", "SAPI", "CHL"}

# MDAnalysis 的默认长度单位是 Å，而 GROMACS 与我们的输出一律用 nm。
# 所有从 Universe 里读出来的坐标都必须乘这个系数，否则 RMSD/距离会大 10 倍。
A2NM = 0.1


def emit(obj):
    """输出单行 JSON 结果并退出。"""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, default=_jsonable))
    sys.stdout.write("\n")
    sys.exit(0)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def fail(msg):
    sys.stdout.write(json.dumps({"ok": False, "error": str(msg)}, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.exit(0)


# ---------------------------------------------------------------- 文件定位
def _pick(ws, explicit, names):
    if explicit:
        for c in (explicit, os.path.join(ws, explicit)):
            if os.path.exists(c):
                return c
    for n in names:
        p = os.path.join(ws, n)
        if os.path.exists(p):
            return p
    return ''


def _is_derived(name):
    """本 worker 生成的派生文件 / GROMACS 备份（#x.1#），绝不能当输入。

    不排除的话会"自己吃自己"：analysis_sol.xtc 比源轨迹新，会被 find_traj 选中，
    再拿去生成 analysis_sub.*，gmx 对已存在的输出先改名备份，中途失败就留下 0 字节文件。
    """
    b = os.path.basename(name)
    return (b.startswith('analysis_sub') or b.startswith('analysis_sol')
            or (b.startswith('#') and b.endswith('#')))


def find_topology(ws, explicit=''):
    """拓扑：显式 -> 常见 gro/pdb 名 -> 工作区最新 .gro/.pdb（GROMACS2026 的 tpr 读不了）。"""
    p = _pick(ws, explicit, ['step5_input.gro', 'step5_1.gro', 'md.gro'])
    if p:
        return p
    cands = [c for c in glob.glob(os.path.join(ws, '*.gro')) + glob.glob(os.path.join(ws, '*.pdb'))
             if os.path.getsize(c) > 1000 and not _is_derived(c)]
    return max(cands, key=os.path.getmtime) if cands else ''


def find_traj(ws, explicit=''):
    """源轨迹（输入）：显式 -> merged.xtc -> 最新的非派生 .xtc。"""
    p = _pick(ws, explicit, ['merged.xtc'])
    if p:
        return p
    cands = [c for c in glob.glob(os.path.join(ws, '*.xtc'))
             if not _is_derived(c) and os.path.getsize(c) > 0]
    return max(cands, key=os.path.getmtime) if cands else ''


def find_sub(ws):
    """已存在的 Protein_LIG 子轨迹对（产物）：.xtc 与 .gro 都在才返回。"""
    xtc = os.path.join(ws, 'analysis_sub.xtc')
    gro = os.path.join(ws, 'analysis_sub.gro')
    return (xtc, gro) if os.path.exists(xtc) and os.path.exists(gro) else ('', '')


def find_plumed(ws, explicit='', kind='COLVAR'):
    """找 PLUMED 输出：显式 -> 工作区(含子目录)里最新的 COLVAR*/HILLS*。"""
    if explicit:
        for c in (explicit, os.path.join(ws, explicit)):
            if os.path.exists(c):
                return c
    pats = ['COLVAR*', 'colvar*'] if kind == 'COLVAR' else ['HILLS*', 'hills*']
    hit = []
    for pat in pats:
        hit += glob.glob(os.path.join(ws, pat)) + glob.glob(os.path.join(ws, '**', pat), recursive=True)
    hit = [h for h in hit if os.path.isfile(h)
           and 'fes_tmp' not in h and not os.path.basename(h).startswith('subset_')]
    return max(hit, key=os.path.getmtime) if hit else ''


def list_plumed(ws, kind='COLVAR'):
    """列出工作区里所有 PLUMED 输出候选（一个目录常同时有 COLVAR 与 COLVAR_A100，
    分属不同 CV，必须让调用方知道有歧义）。"""
    pats = ['COLVAR*', 'colvar*'] if kind == 'COLVAR' else ['HILLS*', 'hills*']
    hit = []
    for pat in pats:
        hit += glob.glob(os.path.join(ws, pat)) + glob.glob(os.path.join(ws, '**', pat), recursive=True)
    hit = [h for h in hit if os.path.isfile(h)
           and 'fes_tmp' not in h and not os.path.basename(h).startswith('subset_')]
    out = []
    for h in sorted(set(hit)):
        out.append({'name': os.path.relpath(h, ws), 'size': os.path.getsize(h)})
    return out


# ---------------------------------------------------------------- 体系解析
def classify(u):
    """拆成 受体 / 配体 / 水，并统计 resname。

    受体 = 原子最多的那条 protein 段；配体 = (a) 既非氨基酸、也非水/离子/脂质的残基，
    外加 (b) 其余较短的 protein 段（肽类配体，合计 <100 残基时）。若配体本身就是短肽，
    只按 resname 过滤会把它当成受体的一部分，必须单独拎出来。
    """
    prot = u.select_atoms('protein')
    segs = list(prot.segments)
    rec = prot
    pep = u.atoms[:0]
    if len(segs) > 1:
        main = max(segs, key=lambda s: len(s.atoms))
        rec = main.atoms
        others = [s for s in segs if s is not main]
        if sum(len(s.residues) for s in others) < 100:
            for s in others:
                pep = pep + s.atoms
    resnames = {}
    for r in u.residues:
        resnames[r.resname] = resnames.get(r.resname, 0) + 1
    lig_res = [rn for rn in resnames
               if rn not in G.AA_SET and rn not in WATER and rn not in IONS and rn not in LIPIDS]
    lig = u.select_atoms('resname ' + ' '.join(sorted(lig_res))) if lig_res else u.atoms[:0]
    if len(pep):
        lig = lig + pep
    return {'receptor': rec, 'ligand': lig, 'resnames': resnames,
            'lig_resnames': sorted(lig_res), 'peptide_ligand': len(pep) > 0}


def receptor_sequence(rec):
    """受体一字母序列 + 残基号；若编号不连续则取最长连续段。"""
    seq, nums = [], []
    for r in rec.residues:
        seq.append(G.AA3to1.get(r.resname, 'X'))
        nums.append(int(r.resid))
    s = ''.join(seq)
    # 取最长连续残基段（跳过断裂/非标准）
    best = (0, 0)
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j + 1
    a, b = best
    return s[a:b + 1], nums[a:b + 1]


def write_index(ws, u, info, name='wc.ndx'):
    """自己写 .ndx（避免 gmx make_ndx 交互）。返回 (路径, 组名列表)。

    注意 GROMACS 的组号从 0 开始，且给了 -n 之后就只认本文件里的组，
    所以组号 == 下面的添加顺序（空组不写、不占号）。调用方要按返回的 labels 定位。
    """
    lines = []
    labels = []

    def add(label, atoms):
        idx = [str(a.index + 1) for a in atoms]
        if not idx:
            return
        labels.append(label)
        lines.append('[ %s ]' % label)
        for i in range(0, len(idx), 15):
            lines.append(' '.join(idx[i:i + 15]))

    add('System', u.atoms)
    add('Protein', info['receptor'])
    add('LIG', info['ligand'])
    add('Protein_LIG', info['receptor'] + info['ligand'])
    wres = sorted(set(info['resnames']) & WATER)
    wsel = u.select_atoms('resname ' + ' '.join(wres)) if wres else u.atoms[:0]
    add('Water', wsel)
    add('Protein_LIG_Water', info['receptor'] + info['ligand'] + wsel)
    p = os.path.join(ws, name)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(NL.join(lines) + NL)
    return p, labels


def find_tpr(ws, explicit=''):
    """找 .tpr。GROMACS 需要它（-pbc mol），而 MDAnalysis 读不了 GROMACS2026 的 tpr，
    两者用途不同，所以 tpr 与 gro 并存。"""
    if explicit:
        for c in (explicit, os.path.join(ws, explicit)):
            if os.path.exists(c):
                return c
    cands = glob.glob(os.path.join(ws, '*.tpr'))
    return max(cands, key=os.path.getmtime) if cands else ''


def ensure_subtraj(ws, top, traj, group_index, tpr='', force=False):
    """用 gmx trjconv 抽出 Protein_LIG 子轨迹 + 配套 .gro（大膜体系 9 万原子 -> 5 千）。

    -pbc mol 要求 -s 是 .tpr；没有 tpr 时退回 -pbc whole（对内部几何同样够用）。
    必须同时产出 .gro：子轨迹原子数远小于原拓扑，MDAnalysis 要用匹配的拓扑来读。
    返回 (xtc, gro)；任一失败返回 ('', '')。
    """
    xtc = os.path.join(ws, 'analysis_sub.xtc')
    gro = os.path.join(ws, 'analysis_sub.gro')
    if (os.path.exists(xtc) and os.path.exists(gro) and not force
            and os.path.getmtime(xtc) >= os.path.getmtime(traj)):
        return xtc, gro
    ndx = os.path.join(ws, 'wc.ndx')
    if not os.path.exists(ndx):
        return '', ''
    src, pbc = (tpr, '-pbc mol') if tpr else (top, '-pbc whole')
    sel = (('%d' % group_index) + NL).encode()
    base = '%s trjconv -s %s -f %s -n %s %s' % (_q(GMX), _q(src), _q(traj), _q(ndx), pbc)
    tx, tg = xtc + '.tmp.xtc', gro + '.tmp.gro'
    for p_ in (tx, tg, xtc, gro):
        if os.path.exists(p_):
            os.remove(p_)
    subprocess.run(base + ' -o ' + _q(tx), shell=True, cwd=ws, input=sel,
                   capture_output=True, timeout=7200)
    if not (os.path.exists(tx) and os.path.getsize(tx) > 0):
        return '', ''
    subprocess.run(base + ' -dump 0 -o ' + _q(tg), shell=True, cwd=ws, input=sel,
                   capture_output=True, timeout=3600)
    if not (os.path.exists(tg) and os.path.getsize(tg) > 0):
        return '', ''
    os.replace(tx, xtc)
    os.replace(tg, gro)
    return xtc, gro


# ---------------------------------------------------------------- 几何与统计
def _q(s):
    return '"%s"' % s


def kabsch_rot(P, Q):
    """求把 P 叠合到 Q 的最优旋转（含反射修正）。P/Q 均为 (N,3) 行向量。"""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, pc, qc


def apply_align(x, R, pc, qc):
    """与 kabsch_rot 配套：x 为 (...,3)，同样变换到 Q 的参考系。"""
    return (x - pc) @ R.T + qc


def rmsd(a, b, align_first=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if align_first:
        R, pc, qc = kabsch_rot(a, b)
        a = apply_align(a, R, pc, qc)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def dihedral(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w))))


def circ_stats(deg):
    """圆统计：均值角 / 集中度 R / 圆标准差(度)。"""
    a = np.radians(np.asarray(deg, dtype=float))
    C, S = float(np.mean(np.cos(a))), float(np.mean(np.sin(a)))
    Rbar = float(np.hypot(C, S))
    mean = float(np.degrees(np.arctan2(S, C)))
    sd = float(np.degrees(np.sqrt(max(0.0, -2.0 * np.log(max(Rbar, 1e-12))))))
    return {'mean_deg': mean, 'R': Rbar, 'circ_sd_deg': sd}


def rotamer_occupancy(deg, centers=(-60.0, 60.0, 180.0)):
    """最近邻 rotamer 占据率（默认 gauche+/gauche-/trans）。"""
    d = np.asarray(deg, dtype=float)
    if d.size == 0:
        return {}
    c = np.array(centers)
    dd = np.abs((d[:, None] - c[None, :] + 180.0) % 360.0 - 180.0)
    lab = np.argmin(dd, axis=1)
    return {('%+d' % int(c[k])): float(np.sum(lab == k) / len(d)) for k in range(len(c))}


def block_bootstrap_ci(x, n_block=6, n_boot=2000, seed=7):
    """分块 bootstrap —— 时间序列有自相关，普通 bootstrap 会低估误差。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return {'mean': float(x.mean()) if x.size else float('nan'),
                'lo': float('nan'), 'hi': float('nan'), 'n': int(x.size)}
    blocks = np.array_split(x, min(n_block, x.size))
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        vals[i] = np.concatenate([blocks[j] for j in pick]).mean()
    return {'mean': float(x.mean()), 'sd': float(vals.std(ddof=1)),
            'lo': float(np.percentile(vals, 2.5)), 'hi': float(np.percentile(vals, 97.5)),
            'n': int(x.size)}


def autocorr_ess(x, max_lag=None):
    """积分自相关时间 tau 与有效样本量 ESS。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return {'tau': float('nan'), 'ess': float(n)}
    y = x - x.mean()
    if np.allclose(y, 0):
        return {'tau': float('nan'), 'ess': float(n)}
    max_lag = max_lag or min(n // 2, 2000)
    ac = np.correlate(y, y, mode='full')[n - 1:n + max_lag]
    ac = ac / ac[0]
    cut = np.where(ac < 0.05)[0]
    m = int(cut[0]) if cut.size else len(ac)
    tau = 1.0 + 2.0 * float(np.sum(ac[1:m]))
    return {'tau': float(tau), 'ess': float(n / max(tau, 1.0))}


# ---------------------------------------------------------------- 输出
NL = chr(10)


def write_csv(path, header, rows):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + NL)
        for r in rows:
            cells = []
            for v in r:
                if isinstance(v, (float, np.floating)):
                    cells.append('%.6g' % float(v))
                elif isinstance(v, (int, np.integer)):
                    cells.append(str(int(v)))
                else:
                    cells.append(str(v))
            f.write(','.join(cells) + NL)
    return path


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_jsonable)
    return path


# ---------------------------------------------------------------- 分析会话
class Session(object):
    '''一次分析会话：拓扑 / 轨迹 / 受体 / 配体 / GPCR 档案 只解析一次。'''

    def __init__(self, ws, top='', traj='', profile_file='gpcr.json', build_sub=True):
        import MDAnalysis as mda
        self.ws = ws
        self.top_full = find_topology(ws, top)
        if not self.top_full:
            raise RuntimeError('找不到拓扑 .gro/.pdb；GROMACS2026 的 tpr MDAnalysis 读不了')
        self.traj = find_traj(ws, traj)
        u_full = mda.Universe(self.top_full)
        info_full = classify(u_full)
        self.sub, self.sub_gro = '', ''
        self.ndx_labels = []
        if build_sub and self.traj:
            _p, self.ndx_labels = write_index(ws, u_full, info_full)
            gi = self.ndx_labels.index('Protein_LIG') if 'Protein_LIG' in self.ndx_labels else 0
            self.sub, self.sub_gro = ensure_subtraj(
                ws, self.top_full, self.traj, gi, find_tpr(ws))
        if not (self.sub and self.sub_gro):
            self.sub, self.sub_gro = find_sub(ws)
        if self.sub and self.sub_gro:
            self.top = self.sub_gro
            self.u = mda.Universe(self.sub_gro, self.sub)
        elif self.traj:
            self.top = self.top_full
            self.u = mda.Universe(self.top_full, self.traj)
        else:
            self.top = self.top_full
            self.u = u_full
        self.info = classify(self.u)
        self.rec = self.info['receptor']
        self.lig = self.info['ligand']
        self.seq, self.nums = receptor_sequence(self.rec)
        self.prof = G.load_profile(ws, profile_file)
        if not self.prof:
            self.prof = G.build_profile(self.seq, self.nums[0], name='receptor')
        self.n_frames = len(self.u.trajectory)
        self.dt_ns = self._frame_dt_ns()
        self.by_res = {}
        for a in self.u.atoms:
            self.by_res.setdefault(int(a.resid), {})[a.name.upper()] = a.index
        self.ca = {}
        for a in self.rec:
            if a.name.upper() == 'CA':
                self.ca[int(a.resid)] = a.index

    def _frame_dt_ns(self):
        try:
            return float(self.u.trajectory.dt) / 1000.0
        except Exception:
            return float('nan')

    def pick(self, resnum, names):
        '''按残基号 + 原子名取原子索引（缺哪个自动跳过）。'''
        d = self.by_res.get(int(resnum), {})
        return [d[n] for n in names if n in d]

    def coords_nm(self, ag=None):
        '''当前帧坐标，换算成 nm（MDAnalysis 默认 Å，不换会大 10 倍）。'''
        return (ag if ag is not None else self.u.atoms).positions * A2NM

    def resname_of(self, resnum):
        '''残基号 -> 残基名（用于判断该位点是 D/E/K/R/F/W/Y 等）。'''
        d = getattr(self, '_rn', None)
        if d is None:
            d = {}
            for r in self.rec.residues:
                d[int(r.resid)] = r.resname
            self._rn = d
        return d.get(int(resnum), '')

    def res_of_bw(self, bw):
        return self.prof.get('anchors', {}).get(str(bw))

    def seg_residues(self, seg_names):
        return G.residues_of(self.prof.get('segments', {}), seg_names)

    def seg_ca(self, seg_names):
        res = self.seg_residues(seg_names)
        idx = [self.ca[r] for r in res if r in self.ca]
        return idx, [r for r in res if r in self.ca]

    def tm_residues(self):
        return self.seg_residues(['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7'])

    def chain_residues(self):
        '''受体全部残基（用于全局相关分析）。'''
        return [int(r.resid) for r in self.rec.residues]

    def ca_tensor(self, resnums=None, stride=1):
        '''把所有帧的 Cα 坐标读成 (F, N, 3) + 时间轴。受体 Cα 只有几百个，内存友好。'''
        cache = getattr(self, '_cache', None)
        if cache is None:
            cache = {}
            self._cache = cache
        res = [int(r) for r in (resnums or sorted(self.ca.keys()))]
        res = [r for r in res if r in self.ca]
        key = (tuple(res), int(stride))
        if key in cache:
            return cache[key]
        idx = [self.ca[r] for r in res]
        sel = self.u.atoms[idx]
        frames = list(range(0, self.n_frames, max(1, int(stride))))
        X = np.empty((len(frames), len(idx), 3), dtype=np.float32)
        T = np.empty(len(frames), dtype=float)
        for k, f in enumerate(frames):
            self.u.trajectory[f]
            X[k] = sel.positions * A2NM
            T[k] = float(self.u.trajectory.time) / 1000.0
        out = {'x': X, 't': T, 'res': res, 'stride': int(stride)}
        cache[key] = out
        return out


def superpose(X, fit_mask, ref=None):
    '''把 (F,N,3) 坐标逐帧叠合到 ref（默认第 0 帧），fit_mask 是长度 N 的布尔数组。'''
    ref = X[0] if ref is None else np.asarray(ref, dtype=float)
    P = np.asarray(X[:, fit_mask, :], dtype=float)
    Q = np.asarray(ref[fit_mask, :], dtype=float)
    F = X.shape[0]
    R = np.empty((F, 3, 3))
    pc = np.empty((F, 3))
    qc = np.empty((F, 3))
    for i in range(F):
        R[i], pc[i], qc[i] = kabsch_rot(P[i], Q)
    return np.einsum('fnj,fkj->fnk', np.asarray(X, dtype=float) - pc[:, None, :], R) + qc[:, None, :]


# ================================================================ 任务实现
SEG_ORDER = ['TM1', 'ICL1', 'TM2', 'ECL1', 'TM3', 'ICL2', 'TM4', 'ECL2',
             'TM5', 'ICL3', 'TM6', 'ECL3', 'TM7', 'H8']


def _auto_stride(s, p, cap=2000):
    if p.get('stride'):
        return max(1, int(p['stride']))
    return max(1, int(s.n_frames // cap))


def task_detect_numbering(ws, p):
    '''识别受体 GPCR 编号档案，写入 gpcr.json，返回摘要。'''
    s = Session(ws, p.get('top', ''), p.get('traj', ''),
                build_sub=bool(p.get('build_sub', True)))
    path = G.save_profile(ws, s.prof)
    return {'ok': True, 'profile_file': os.path.basename(path),
            'summary': G.profile_summary(s.prof), 'profile': s.prof,
            'n_frames': s.n_frames, 'dt_ns': s.dt_ns,
            'residue_range': [s.nums[0], s.nums[-1]], 'seq_len': len(s.seq),
            'ligand_resnames': s.info['lig_resnames'],
            'n_receptor_atoms': int(s.rec.n_atoms), 'n_ligand_atoms': int(s.lig.n_atoms),
            'subtraj': os.path.basename(s.sub) if s.sub else ''}


def task_segments(ws, p):
    '''分段 Cα RMSD/RMSF：以核心段(默认 TM1-TM4)叠合，逐段 RMSD 时间序列 + 逐残基 RMSF。'''
    s = Session(ws, p.get('top', ''), p.get('traj', ''))
    segs = dict(s.prof.get('segments', {}))
    if p.get('segments'):
        segs.update(G.parse_segments_str(p['segments']))
    names = [n for n in SEG_ORDER if n in segs]
    fit_names = [x.strip() for x in str(p.get('fit_segments', '')).split(',') if x.strip()]
    if not fit_names:
        fit_names = s.prof.get('tm_core_align', ['TM1', 'TM2', 'TM3', 'TM4'])
    stride = _auto_stride(s, p)
    d = s.ca_tensor(None, stride)
    X, T, res = d['x'], d['t'], d['res']
    pos = {r: i for i, r in enumerate(res)}
    fit_res = set(s.seg_residues(fit_names))
    fit_mask = np.array([r in fit_res for r in res])
    if fit_mask.sum() < 3:
        raise RuntimeError('叠合段 %s 找不到足够 Cα，请检查段定义' % fit_names)
    Y = superpose(X, fit_mask)
    ref = Y[0]
    series = {}
    stats = {}
    for n in names:
        a, b = segs[n]
        sel = [pos[r] for r in range(a, b + 1) if r in pos]
        if not sel:
            continue
        dev = Y[:, sel, :] - ref[None, sel, :]
        rt = np.sqrt(np.mean(np.sum(dev ** 2, axis=2), axis=1))
        series[n] = rt
        stats[n] = {'range': [a, b], 'n_ca': len(sel),
                    'rmsd_mean_nm': float(rt.mean()),
                    'rmsd_last_nm': float(rt[-1]),
                    'ci95': block_bootstrap_ci(rt)}
    cols = [n for n in names if n in series]
    write_csv(os.path.join(ws, 'segments_rmsd.csv'), ['time_ns'] + cols,
              list(zip(T, *[series[n] for n in cols])))
    dev_all = Y - ref[None]
    rmsf = np.sqrt(np.mean(np.sum(dev_all ** 2, axis=2), axis=0))
    seg_of = {}
    for n in names:
        a, b = segs[n]
        for r in range(a, b + 1):
            seg_of[r] = n
    rows = []
    res_rmsf = {}
    for i, r in enumerate(res):
        lab = G.bw_label(r, s.prof.get('anchors', {}), segs) or ''
        rows.append([r, lab, seg_of.get(r, 'other'), float(rmsf[i])])
        res_rmsf[r] = float(rmsf[i])
    write_csv(os.path.join(ws, 'segments_rmsf.csv'),
              ['resnum', 'bw', 'segment', 'rmsf_nm'], rows)
    for n in names:
        a, b = segs[n]
        vals = [res_rmsf[r] for r in range(a, b + 1) if r in res_rmsf]
        if vals:
            st = stats.setdefault(n, {'range': [a, b], 'n_ca': len(vals)})
            st['rmsf_mean_nm'] = float(np.mean(vals))
            st['rmsf_max_nm'] = float(np.max(vals))
    anchors = s.prof.get('anchors', {})
    flex = sorted((v, r) for r, v in res_rmsf.items())[::-1][:10]
    return {'ok': True, 'fit_segments': fit_names, 'stride': stride,
            'n_frames_used': int(len(T)),
            'time_span_ns': [float(T[0]), float(T[-1])] if len(T) else [],
            'segments': stats,
            'top_flexible_residues': [
                {'resnum': r, 'bw': G.bw_label(r, anchors, segs) or '', 'rmsf_nm': float(v)}
                for v, r in flex],
            'files': ['segments_rmsd.csv', 'segments_rmsf.csv'],
            'profile_source': s.prof.get('source', '')}


# ================================================================ CLI
def _tasks():
    """收集所有任务：本模块的 task_* + md_gpcr_tasks 的。

    md_gpcr_tasks 在模块级 import 本模块，所以这里必须延迟导入，否则循环导入。
    """
    t = {}
    for k, v in list(globals().items()):
        if k.startswith('task_') and callable(v):
            t[k[5:]] = v
    try:
        import md_gpcr_tasks as X
        for k, v in list(vars(X).items()):
            if k.startswith('task_') and callable(v):
                t.setdefault(k[5:], v)
    except Exception as e:
        sys.stderr.write('warn: md_gpcr_tasks 未加载: %s\n' % e)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', nargs='?', default='')
    ap.add_argument('--ws', default='.')
    ap.add_argument('--json', default='{}')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    T = _tasks()
    if a.list or not a.task:
        emit({'ok': True, 'tasks': sorted(T)})
    try:
        p = json.loads(a.json) if a.json.strip() else {}
    except Exception as e:
        fail('--json 解析失败: %s' % e)
    fn = T.get(a.task)
    if fn is None:
        fail('未知任务 %s；可用: %s' % (a.task, sorted(T)))
    try:
        emit(fn(os.path.abspath(a.ws), p))
    except Exception as e:
        import traceback
        fail('%s: %s%s%s' % (type(e).__name__, e, NL, traceback.format_exc()[-1500:]))


# 入口调用放在文件末尾：main() 里 _tasks() 要扫描全部 task_* 函数，必须等它们都定义完


def atoms_for(s, resnum, kind):
    '''按 BW 位点 + 原子类别取原子索引：sc 侧链末端 / ring 芳环 / bb 主链。'''
    if not resnum:
        return []
    rn = s.resname_of(resnum)
    aa = G.AA3to1.get(rn, '')
    if kind == 'ring':
        return s.pick(resnum, G.RING_ATOMS.get(aa, []))
    if kind == 'bb':
        return s.pick(resnum, ['N', 'CA', 'C', 'O'])
    if kind == 'sc':
        names = G.CHARGE_ATOMS.get(aa, [])
        fallback = ['CB', 'CG', 'SG', 'CD', 'CD1', 'CD2', 'CE', 'CZ', 'NZ']
        return s.pick(resnum, names if names else fallback)
    return s.pick(resnum, [kind.upper()])


def task_microswitches(ws, p):
    '''微开关几何：PIF / CWxP / NPxxY / DRY / 离子锁 的距离、chi 二面角、占据率。'''
    s = Session(ws, p.get('top', ''), p.get('traj', ''))
    stride = _auto_stride(s, p, 4000)
    sws = p.get('switches') or G.MICROSWITCH_ORDER
    if isinstance(sws, str):
        sws = [x.strip() for x in sws.split(',') if x.strip()]
    anchors = s.prof.get('anchors', {})
    dists = []
    dihs = []
    missing = []
    for sw in sws:
        spec = G.MICROSWITCH_SPEC.get(sw)
        if not spec:
            continue
        for a_bw, a_kind, b_bw, b_kind in spec['distances']:
            ia = atoms_for(s, anchors.get(a_bw), a_kind)
            ib = atoms_for(s, anchors.get(b_bw), b_kind)
            if ia and ib:
                dists.append((sw, a_bw + '-' + b_bw, ia, ib))
            else:
                missing.append(sw + ': ' + a_bw + '-' + b_bw)
        for bw, kind in spec.get('dihedrals', []):
            r = anchors.get(bw)
            rn = s.resname_of(r) if r else ''
            aa = G.AA3to1.get(rn, '')
            q = s.pick(r, G.CHI_ATOMS.get(kind, {}).get(aa, [])) if r else []
            if len(q) == 4:
                dihs.append((sw, bw + ' ' + kind, q))
            else:
                missing.append(sw + ': ' + bw + ' ' + kind)
    need = sorted({i for _s, _l, ia, ib in dists for i in ia + ib}
                  | {i for _s, _l, q in dihs for i in q})
    if not need:
        raise RuntimeError('没有任何可测量的微开关原子；请检查 gpcr.json 里的 BW 锚点')
    pm = {a: k for k, a in enumerate(need)}
    sel = s.u.atoms[need]
    labels = [l for _s, l, _a, _b in dists] + [l for _s, l, _q in dihs]
    n_used = len(range(0, s.n_frames, stride))
    T = np.empty(n_used)
    D = {l: np.empty(n_used) for l in labels}
    k = 0
    for f in range(0, s.n_frames, stride):
        s.u.trajectory[f]
        P = sel.positions * A2NM
        T[k] = float(s.u.trajectory.time) / 1000.0
        for _s, l, ia, ib in dists:
            A = P[[pm[i] for i in ia]]
            B = P[[pm[i] for i in ib]]
            D[l][k] = float(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min())
        for _s, l, q in dihs:
            D[l][k] = dihedral(P[pm[q[0]]], P[pm[q[1]]], P[pm[q[2]]], P[pm[q[3]]])
        k += 1
    summary = {}
    for sw in sws:
        spec = G.MICROSWITCH_SPEC.get(sw)
        if not spec:
            continue
        entry = {'summary': spec['summary'], 'pairs': {}, 'dihedrals': {}}
        for s2, l, _ia, _ib in dists:
            if s2 != sw:
                continue
            v = D[l]
            entry['pairs'][l] = {
                'mean_nm': float(v.mean()), 'min_nm': float(v.min()),
                'max_nm': float(v.max()),
                'sd_nm': float(v.std(ddof=1)) if v.size > 1 else 0.0,
                'occ_0.4nm': float(np.mean(v < 0.4)), 'occ_0.5nm': float(np.mean(v < 0.5)),
                'ci95': block_bootstrap_ci(v), 'ess': autocorr_ess(v)}
        for s2, l, _q in dihs:
            if s2 != sw:
                continue
            st = circ_stats(D[l])
            st['rotamer_occ'] = rotamer_occupancy(D[l])
            entry['dihedrals'][l] = st
        summary[sw] = entry
    files = []
    for sw in sws:
        cols = [l for s2, l, _a, _b in dists if s2 == sw]
        cols += [l for s2, l, _q in dihs if s2 == sw]
        if not cols:
            continue
        fn = 'microswitch_' + sw + '.csv'
        write_csv(os.path.join(ws, fn), ['time_ns'] + cols,
                  list(zip(T, *[D[c] for c in cols])))
        files.append(fn)
    motif = _motif_ca(s, sws, anchors, stride)
    used = {}
    for sw in sws:
        if sw in G.MICROSWITCH_SPEC:
            used[sw] = G.switch_residues(s.prof, sw)
    span = [float(T[0]), float(T[-1])] if n_used else []
    return {'ok': True, 'stride': stride, 'n_frames_used': int(n_used),
            'time_span_ns': span, 'residues_used': used, 'switches': summary,
            'missing_measurements': missing, 'files': files,
            'motif_ca_rmsd': motif, 'profile_source': s.prof.get('source', '')}


def _motif_ca(s, sws, anchors, stride):
    '''某开关声明的 motif_ca 位点自叠合后的 Cα RMSD（相对首帧），即基序自身构象变形。'''
    out = {}
    for sw in sws:
        spec = G.MICROSWITCH_SPEC.get(sw)
        if not spec or not spec.get('motif_ca'):
            continue
        rs = []
        for bw in spec['motif_ca']:
            r = anchors.get(bw)
            if r in s.ca:
                rs.append(r)
        if len(rs) < 3:
            continue
        d = s.ca_tensor(rs, stride)
        Y = superpose(d['x'], np.ones(len(d['res']), dtype=bool))
        v = np.sqrt(np.mean(np.sum((Y - Y[0]) ** 2, axis=2), axis=1))
        out[sw] = {'residues': d['res'], 'ca_rmsd_mean_nm': float(v.mean()),
                   'ca_rmsd_last_nm': float(v[-1]), 'ci95': block_bootstrap_ci(v)}
    return out


def task_tm6_opening(ws, p):
    '''TM6 胞内端外移：TM6 胞内段到 TM3/TM7 胞内端的 COM 距离 + 相对首帧位移。'''
    s = Session(ws, p.get('top', ''), p.get('traj', ''))
    stride = _auto_stride(s, p, 4000)
    segs = s.prof.get('segments', {})
    tm6, tm3, tm7 = segs.get('TM6'), segs.get('TM3'), segs.get('TM7')
    if not tm6:
        raise RuntimeError('档案里没有 TM6 段定义，无法计算激活位移')
    n_ic = int(p.get('ic_len', 6) or 6)
    # TM6 的方向与 TM3/TM7 相反：class A 里 TM6 自胞内侧(N 端)走向胞外(C 端)，
    # 所以胞内端是 TM6 的【低残基号】一侧。取错一侧会把跨膜束对角距离(~4.5 nm)
    # 当成"胞内开口"，数值看似合理却完全无意义。
    res_ic = [r for r in range(tm6[0], tm6[0] + n_ic) if r in s.ca]
    res_3 = [r for r in range(tm3[1] - 3, tm3[1] + 1) if r in s.ca] if tm3 else []
    res_7 = [r for r in range(tm7[1] - 3, tm7[1] + 1) if r in s.ca] if tm7 else []
    if len(res_ic) < 3 or not res_3:
        raise RuntimeError('TM6 胞内端或 TM3 参考位点在受体中取不到 Cα')
    core = s.prof.get('tm_core_align', ['TM1', 'TM2', 'TM3', 'TM4'])
    core_res = set(s.seg_residues(core))
    d = s.ca_tensor(None, stride)
    X, T, res = d['x'], d['t'], d['res']
    fit_mask = np.array([r in core_res for r in res])
    if fit_mask.sum() < 10:
        raise RuntimeError('叠合核心段 %s 取不到足够 Cα' % core)
    Y = superpose(X, fit_mask)
    idx = {r: i for i, r in enumerate(res)}
    def com(rlist):
        ii = [idx[r] for r in rlist if r in idx]
        return Y[:, ii, :].mean(axis=1)
    def dist(a, b):
        return np.sqrt(((a - b) ** 2).sum(axis=1))
    c6, c3, c7 = com(res_ic), com(res_3), com(res_7)
    d63 = dist(c6, c3)
    d67 = dist(c6, c7) if res_7 else np.zeros(len(T))
    shift = dist(c6, c6[0:1])
    write_csv(os.path.join(ws, 'tm6_opening.csv'),
              ['time_ns', 'TM6ic_TM3ic_nm', 'TM6ic_TM7ic_nm', 'TM6ic_shift_nm'],
              list(zip(T, d63, d67, shift)))
    m63 = {'first_nm': float(d63[0]), 'last_nm': float(d63[-1]),
           'mean_nm': float(d63.mean()), 'delta_nm': float(d63[-1] - d63[0]),
           'range_nm': float(d63.max() - d63.min()), 'ci95': block_bootstrap_ci(d63)}
    m67 = {'first_nm': float(d67[0]), 'last_nm': float(d67[-1]),
           'mean_nm': float(d67.mean()), 'delta_nm': float(d67[-1] - d67[0]),
           'range_nm': float(d67.max() - d67.min()), 'ci95': block_bootstrap_ci(d67)}
    dsh = {'last_nm': float(shift[-1]), 'max_nm': float(shift.max())}
    dres = {'TM6ic': res_ic, 'TM3ic': res_3, 'TM7ic': res_7}
    return {'ok': True, 'stride': stride, 'n_frames_used': int(len(T)),
            'fit_segments': core, 'residues': dres, 'TM6ic_TM3ic_nm': m63,
            'TM6ic_TM7ic_nm': m67, 'TM6ic_shift_nm': dsh,
            'files': ['tm6_opening.csv'], 'profile_source': s.prof.get('source', '')}


def _sel_tensor(s, atom_indices, stride):
    '''任意原子集合的坐标张量 (F, N, 3) nm + 时间轴（已按索引排序去重）。'''
    idx = sorted(set(int(i) for i in atom_indices))
    sel = s.u.atoms[idx]
    frames = list(range(0, s.n_frames, max(1, int(stride))))
    X = np.empty((len(frames), len(idx), 3))
    T = np.empty(len(frames))
    for k, f in enumerate(frames):
        s.u.trajectory[f]
        X[k] = sel.positions * A2NM
        T[k] = float(s.u.trajectory.time) / 1000.0
    return {'x': X, 't': T, 'idx': idx}


def task_ligand_retention(ws, p):
    '''配体保持：受体核心叠合后的 配体 RMSD / COM 位移 / 最近距离 / 接触数与逐残基接触占据率。'''
    s = Session(ws, p.get('top', ''), p.get('traj', ''))
    if s.lig.n_atoms == 0:
        raise RuntimeError('体系里没识别到配体（非氨基酸/水/离子/脂质残基）；若是共价或肽配体请检查')
    stride = _auto_stride(s, p, 3000)
    cut = float(p.get('contact_cutoff_nm', 0.45) or 0.45)
    core = s.prof.get('tm_core_align', ['TM1', 'TM2', 'TM3', 'TM4'])
    core_res = set(s.seg_residues(core))
    d = s.ca_tensor(None, stride)
    X, T, res = d['x'], d['t'], d['res']
    fit_mask = np.array([r in core_res for r in res])
    if fit_mask.sum() < 10:
        raise RuntimeError('叠合核心段 %s 取不到足够 Cα' % core)
    F = X.shape[0]
    R = np.empty((F, 3, 3)); pc = np.empty((F, 3)); qc = np.empty((F, 3))
    for i in range(F):
        R[i], pc[i], qc[i] = kabsch_rot(X[i, fit_mask, :], X[0, fit_mask, :])
    lig = _sel_tensor(s, [a.index for a in s.lig], stride)
    heavy = [a.index for a in s.rec if a.name[0] != 'H']
    rec = _sel_tensor(s, heavy, stride)
    LigT = np.einsum('fnj,fkj->fnk', lig['x'] - pc[:, None, :], R) + qc[:, None, :]
    RecT = np.einsum('fnj,fkj->fnk', rec['x'] - pc[:, None, :], R) + qc[:, None, :]
    lig_rmsd = np.sqrt(np.mean(np.sum((LigT - LigT[0]) ** 2, axis=2), axis=1))
    lc = LigT.mean(axis=1)
    com_shift = np.sqrt(((lc - lc[0]) ** 2).sum(axis=1))
    rres = np.array([int(a.resid) for a in s.rec if a.name[0] != 'H'])
    res_list, starts = np.unique(rres, return_index=True)
    n_contacts = np.zeros(F)
    min_dist = np.zeros(F)
    occ = np.zeros((F, len(res_list)))
    for f in range(F):
        dd = np.sqrt(((LigT[f][:, None, :] - RecT[f][None, :, :]) ** 2).sum(-1))
        am = dd.min(axis=0)
        min_dist[f] = am.min()
        n_contacts[f] = float((dd.min(axis=1) < cut).sum())
        occ[f] = np.minimum.reduceat(am, starts)
    write_csv(os.path.join(ws, 'ligand_retention.csv'),
              ['time_ns', 'ligand_rmsd_nm', 'ligand_com_shift_nm', 'min_dist_nm', 'n_contact_atoms'],
              list(zip(T, lig_rmsd, com_shift, min_dist, n_contacts)))
    segs = s.prof.get('segments', {})
    an = s.prof.get('anchors', {})
    seg_of = {}
    for k in SEG_ORDER:
        if k in segs:
            for r in range(segs[k][0], segs[k][1] + 1):
                seg_of[r] = k
    occ_hit = occ < cut
    occ_mean = occ_hit.mean(axis=0)
    rows = []
    for k, r in enumerate(res_list):
        r = int(r)
        hit = occ[:, k] < cut
        md = float(occ[hit, k].min()) if hit.any() else -1.0
        rows.append([r, G.bw_label(r, an, segs) or '', seg_of.get(r, 'other'), float(occ_mean[k]), md])
    write_csv(os.path.join(ws, 'ligand_contacts.csv'),
              ['resnum', 'bw', 'segment', 'occupancy', 'min_dist_nm_when_contact'], rows)
    order = np.argsort(-occ_mean)
    top = []
    for k in order[:15]:
        if occ_mean[k] <= 0:
            break
        r = int(res_list[k])
        top.append({'resnum': r, 'bw': G.bw_label(r, an, segs) or '',
                    'segment': seg_of.get(r, 'other'), 'occupancy': float(occ_mean[k])})
    s_rmsd = {'mean_nm': float(lig_rmsd.mean()), 'last_nm': float(lig_rmsd[-1]),
              'max_nm': float(lig_rmsd.max()), 'ci95': block_bootstrap_ci(lig_rmsd)}
    s_com = {'last_nm': float(com_shift[-1]), 'max_nm': float(com_shift.max())}
    s_min = {'mean_nm': float(min_dist.mean()), 'last_nm': float(min_dist[-1])}
    s_nc = {'mean': float(n_contacts.mean()), 'min': float(n_contacts.min()),
            'last': float(n_contacts[-1])}
    return {'ok': True, 'stride': stride, 'n_frames_used': int(F), 'contact_cutoff_nm': cut,
            'ligand_resnames': s.info['lig_resnames'], 'n_ligand_atoms': int(s.lig.n_atoms),
            'ligand_rmsd_nm': s_rmsd, 'ligand_com_shift_nm': s_com,
            'min_dist_nm': s_min, 'n_contact_atoms': s_nc, 'persistent_contacts': top,
            'files': ['ligand_retention.csv', 'ligand_contacts.csv'],
            'profile_source': s.prof.get('source', '')}


def parse_plumed(path, max_rows=0):
    '''解析 PLUMED 输出（COLVAR / HILLS）：读 #! FIELDS / #! SET 头，返回字段与数值数组。'''
    fields = []
    sets = {}
    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('#!'):
                toks = s[2:].split()
                if toks and toks[0] == 'FIELDS':
                    fields = toks[1:]
                elif toks and toks[0] == 'SET' and len(toks) >= 3:
                    sets[toks[1]] = ' '.join(toks[2:])
                continue
            if s.startswith('#'):
                continue
            vals = s.split()
            try:
                rows.append([float(v) for v in vals])
            except ValueError:
                continue
            if max_rows and len(rows) >= max_rows:
                break
    if not rows:
        return {'fields': fields, 'sets': sets, 'data': np.zeros((0, max(1, len(fields))))}
    ncol = min(len(fields), len(rows[0]))
    data = np.array([r[:ncol] for r in rows if len(r) >= ncol])
    return {'fields': fields[:ncol], 'sets': sets, 'data': data}


def monotonic_segments(t, min_len=2):
    '''重启拼接会带来时间回跳；返回单调递增段 [(i0, i1)]（闭区间，含端点）。'''
    segs = []
    i0 = 0
    for i in range(1, len(t)):
        if t[i] < t[i - 1]:
            if i - i0 >= min_len:
                segs.append((i0, i - 1))
            i0 = i
    if len(t) - i0 >= min_len:
        segs.append((i0, len(t) - 1))
    return segs


def hills_ncv(fields):
    '''从 HILLS 字段名推 CV 个数：time + cv*N + sigma_*N + height [+ biasf]。'''
    n = len(fields)
    if n >= 3 and fields[-1] == 'biasf':
        return max(1, (n - 2) // 2)
    return max(1, (n - 1) // 2)


def task_plumed_parse(ws, p):
    '''解析 PLUMED 的 COLVAR / HILLS：字段、时间回跳分段、每列统计、CV 探索范围，导出 CSV。'''
    cv_file = find_plumed(ws, p.get('colvar', ''), 'COLVAR')
    h_file = find_plumed(ws, p.get('hills', ''), 'HILLS')
    out = {'ok': True, 'colvar_file': os.path.basename(cv_file) if cv_file else '',
           'hills_file': os.path.basename(h_file) if h_file else '', 'files': []}
    out['colvar_candidates'] = list_plumed(ws, 'COLVAR')
    out['hills_candidates'] = list_plumed(ws, 'HILLS')
    if not cv_file and not h_file:
        raise RuntimeError('工作区里找不到 COLVAR*/HILLS*；元动力学分析需要 PLUMED 输出')
    if cv_file:
        d = parse_plumed(cv_file)
        f, A = d['fields'], d['data']
        if A.size:
            t = A[:, 0]
            segs = monotonic_segments(t)
            stats = {}
            for k, nm in enumerate(f):
                v = A[:, k]
                stats[nm] = {'mean': float(v.mean()), 'min': float(v.min())}
                stats[nm]['max'] = float(v.max())
                stats[nm]['first'] = float(v[0])
                stats[nm]['last'] = float(v[-1])
            cv = {}
            cv['fields'] = f
            cv['sets'] = d['sets']
            cv['n_points'] = int(A.shape[0])
            cv['n_time_segments'] = len(segs)
            cv['time_span'] = [float(t[0]), float(t[-1])]
            cv['stats'] = stats
            out['colvar'] = cv
            step = max(1, A.shape[0] // 5000)
            write_csv(os.path.join(ws, 'plumed_colvar.csv'), f, list(A[::step, :]))
            out['files'].append('plumed_colvar.csv')
            if 'metad.bias' in f:
                b = A[:, f.index('metad.bias')]
                bg = {}
                bg['first_kjmol'] = float(b[0])
                bg['last_kjmol'] = float(b[-1])
                bg['max_kjmol'] = float(b.max())
                out['bias_growth'] = bg
    if h_file:
        dh = parse_plumed(h_file)
        fh, H = dh['fields'], dh['data']
        if H.size:
            ncv = hills_ncv(fh)
            hs = {}
            hs['fields'] = fh
            hs['sets'] = dh['sets']
            hs['n_cv'] = ncv
            hs['n_hills'] = int(H.shape[0])
            out['hills'] = hs
            hi = fh.index('height') if 'height' in fh else H.shape[1] - 1
            hs['total_height_kjmol'] = float(H[:, hi].sum())
            hs['first_height_kjmol'] = float(H[0, hi])
            hs['last_height_kjmol'] = float(H[-1, hi])
            hs['height_ratio_last_first'] = float(H[-1, hi] / H[0, hi]) if H[0, hi] else float('nan')
            cvs = []
            for k in range(ncv):
                v = H[:, 1 + k]
                e = {}
                e['name'] = fh[1 + k]
                e['min'] = float(v.min())
                e['max'] = float(v.max())
                e['first'] = float(v[0])
                e['last'] = float(v[-1])
                e['explored_range'] = float(v.max() - v.min())
                cvs.append(e)
            hs['cvs'] = cvs
            write_csv(os.path.join(ws, 'plumed_hills.csv'), fh, list(H))
            out['files'].append('plumed_hills.csv')
    return out


if __name__ == '__main__':
    main()
