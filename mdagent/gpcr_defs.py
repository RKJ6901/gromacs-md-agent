"""GPCR 领域知识层：Ballesteros-Weinstein(BW) 通用编号、跨膜段定义、微开关残基。

纯标准库实现（不 import numpy / MDAnalysis），因此可被两个不同 env 同时导入：
  - agent venv 的 tools.py（出图 / 结论组织）
  - 分析 env(gmxMMPBSA) 的 worker.py（几何计算）

编号来源优先级：
  1) 工作区 gpcr.json —— 人工或 agent 显式给出，可事后修正，最高优先级
  2) 序列 motif 自动识别 —— DRY / CWxP / NPxxY / TM5 脯氨酸
  3) 内置 APLNR 档案 —— 取自 7W0M CHARMM-GUI 实测序列（残基 26-315），已逐条核对

BW 约定要点（易错）：
  DRY   : D3.49 / R3.50 / Y3.51
  CWxP  : C6.47 / W6.48 / x6.49 / P6.50
  NPxxY : N7.49 / P7.50 / x7.51 / x7.52 / Y7.53   <- N 是 7.49 不是 7.39
  PIF   : x3.40 / P5.50 / F6.44（空间三元组，序列上无固定 motif）
  离子锁: D/E3.49 - R3.50 - E6.30（部分受体缺失，如 APLNR 的 6.30 位为 R）
"""
import json
import os
import re

# ---------------------------------------------------------------- 常量表
AA3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # CHARMM 质子化态命名
    "HSD": "H", "HSE": "H", "HSP": "H", "HID": "H", "HIE": "H", "HIP": "H",
    "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K",
}
AA_SET = set(AA3to1)

# Kyte-Doolittle 疏水标度
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# 侧链二面角原子名（按残基类型），用于在 .gro/.pdb 里选原子算 chi1/chi2
CHI_ATOMS = {
    "chi1": {
        "I": ["N", "CA", "CB", "CG1"], "V": ["N", "CA", "CB", "CG1"],
        "L": ["N", "CA", "CB", "CG"], "F": ["N", "CA", "CB", "CG"],
        "W": ["N", "CA", "CB", "CG"], "Y": ["N", "CA", "CB", "CG"],
        "M": ["N", "CA", "CB", "CG"], "C": ["N", "CA", "CB", "SG"],
        "D": ["N", "CA", "CB", "CG"], "E": ["N", "CA", "CB", "CG"],
        "N": ["N", "CA", "CB", "CG"], "Q": ["N", "CA", "CB", "CG"],
        "K": ["N", "CA", "CB", "CG"], "R": ["N", "CA", "CB", "CG"],
        "H": ["N", "CA", "CB", "CG"], "S": ["N", "CA", "CB", "OG"],
        "T": ["N", "CA", "CB", "OG1"], "P": ["N", "CA", "CB", "CG"],
    },
    "chi2": {
        "F": ["CA", "CB", "CG", "CD1"], "W": ["CA", "CB", "CG", "CD1"],
        "Y": ["CA", "CB", "CG", "CD1"], "H": ["CA", "CB", "CG", "ND1"],
        "M": ["CA", "CB", "CG", "SD"], "D": ["CA", "CB", "CG", "OD1"],
        "N": ["CA", "CB", "CG", "OD1"], "E": ["CA", "CB", "CG", "CD"],
        "Q": ["CA", "CB", "CG", "CD"], "K": ["CA", "CB", "CG", "CD"],
        "R": ["CA", "CB", "CG", "CD"], "I": ["CA", "CB", "CG1", "CD1"],
        "L": ["CA", "CB", "CG", "CD1"], "P": ["CA", "CB", "CG", "CD"],
    },
}

# 带电/极性侧链末端原子（用于盐桥与氢键几何）
CHARGE_ATOMS = {
    "D": ["OD1", "OD2"], "E": ["OE1", "OE2"],
    "R": ["NH1", "NH2", "NE"], "K": ["NZ"],
    "N": ["OD1", "ND2"], "Q": ["OE1", "NE2"],
    "Y": ["OH"], "S": ["OG"], "T": ["OG1"], "H": ["ND1", "NE2"],
    "W": ["NE1"], "C": ["SG"],
}

# 芳香环原子（用于堆积/CWxP 门控）；P 的吡咯烷环也放进来，PIF 里 P5.50 要按环取原子
RING_ATOMS = {
    "F": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "Y": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "W": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "H": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "P": ["CB", "CG", "CD"],
}


# ---------------------------------------------------------------- motif 识别
# name -> (正则, 相对 BW 标签列表)。class A 每个 motif 唯一，取首个匹配。
# 注意：全部正则刻意不含反斜杠，便于跨工具传递。
MOTIFS = {
    'DRY':   ('D[RE][YF]',          ['3.49', '3.50', '3.51']),
    'CWxP':  ('C[WYF][^P]P',        ['6.47', '6.48', '6.49', '6.50']),
    'NPxxY': ('N[PGA][^P]{2}[YF]',  ['7.49', '7.50', '7.51', '7.52', '7.53']),
}

# 有公认序列 motif 锚点的螺旋（TM3/TM6/TM7）；其余螺旋靠疏水预测定位
ANCHORED_HELICES = (3, 6, 7)


def find_motifs(seq, seq_start=1):
    """查找 class A 保守 motif，位置一律换算成【残基号】（非序列下标）。

    返回 {motif: {'span': (first_resnum, last_resnum), 'anchors': {bw: resnum}}}，
    只保留最靠前的匹配（class A 每个 motif 唯一）。
    """
    out = {}
    for name, (pat, labels) in MOTIFS.items():
        hits = list(re.finditer(pat, seq))
        if not hits:
            continue
        m = hits[0]
        anchors = {}
        for k, bw in enumerate(labels):
            anchors[bw] = m.start() + k + seq_start
        out[name] = {
            'span': (m.start() + seq_start, m.start() + len(m.group()) - 1 + seq_start),
            'anchors': anchors,
        }
    return out


# ---------------------------------------------------------------- TM 螺旋预测
def kd_profile(seq, win=19):
    """Kyte-Doolittle 滑动平均疏水曲线（窗宽 win，中心对齐）。"""
    n = len(seq)
    half = win // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = seq[lo:hi]
        out.append(sum(KD.get(c, 0.0) for c in w) / max(1, len(w)))
    return out


def predict_tm_helices(seq, thr=0.9, max_gap=6, min_len=12, n_tm=7, target_len=26):
    """疏水预测 7 条跨膜螺旋，返回 (螺旋 0-based 闭区间, 额外短螺旋, 疏水曲线)。

    只依赖序列，不需要 DSSP 外部二进制。步骤：
      阈值切分 -> 合并小间隙(<=max_gap, 且合并后 <=40 残基) -> 取最疏水的 7 条
      -> 各扩到约 target_len（典型跨膜段长度）并防止相邻重叠。
    """
    n = len(seq)
    prof = kd_profile(seq)
    raw = []
    i = 0
    while i < n:
        if prof[i] > thr:
            j = i
            while j + 1 < n and prof[j + 1] > thr:
                j += 1
            raw.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for r in raw:
        if (merged and r[0] - merged[-1][1] - 1 <= max_gap
                and (r[1] - merged[-1][0] + 1) <= 40):
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    merged = [r for r in merged if r[1] - r[0] + 1 >= min_len]
    extras = []
    if len(merged) > n_tm:
        ranked = sorted(merged, key=lambda r: -sum(prof[r[0]:r[1] + 1]))
        keep = {tuple(r) for r in ranked[:n_tm]}
        extras = sorted(r for r in merged if tuple(r) not in keep)
        merged = sorted(ranked[:n_tm])
    ext = []
    for idx, (a, b) in enumerate(merged):
        grow = max(0, target_len - (b - a + 1))
        left = grow // 2
        right = grow - left
        lo_bound = 0 if idx == 0 else ext[-1][1] + 2
        hi_bound = (n - 1) if idx == len(merged) - 1 else merged[idx + 1][0] - 2
        na = max(lo_bound, a - left)
        nb = min(hi_bound, b + right)
        if nb - na + 1 < (b - a + 1):
            na, nb = a, b
        ext.append([na, nb])
    return ext, extras, prof


def build_segments(seq_start, helices):
    """由 7 条 TM 螺旋推 TM1-7 / ICL1-3 / ECL1-3（残基号，1-based 闭区间）。"""
    names = ['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7']
    seg = {}
    for k, (a, b) in enumerate(helices[:7]):
        seg[names[k]] = (a, b)
    loops = ['ICL1', 'ECL1', 'ICL2', 'ECL2', 'ICL3', 'ECL3']
    for k in range(min(6, len(helices) - 1)):
        a = helices[k][1] + 1
        b = helices[k + 1][0] - 1
        if b >= a:
            seg[loops[k]] = (a, b)
    if helices and helices[0][0] > seq_start:
        seg['Nterm'] = (seq_start, helices[0][0] - 1)
    return seg


def derive_anchors(motifs, seq, seq_start, helices):
    """motif 锚点 + 由锚点推得的位置（3.40 / 6.44 / 7.39 / 5.50 / 6.30）。"""
    anchors = {}
    for info in motifs.values():
        anchors.update(info['anchors'])
    notes = []
    if '3.49' in anchors:
        anchors['3.40'] = anchors['3.49'] - 9
    if '6.48' in anchors:
        anchors['6.44'] = anchors['6.48'] - 4
    if '7.49' in anchors:
        anchors['7.39'] = anchors['7.49'] - 10
    n = len(seq)
    p550 = None
    if len(helices) >= 5:
        a, b = helices[4]
        pros = [r for r in range(a, b + 1)
                if 0 <= r - seq_start < n and seq[r - seq_start] == 'P']
        if len(pros) == 1:
            p550 = pros[0]
        elif len(pros) > 1:
            mid = (a + b) / 2.0
            p550 = min(pros, key=lambda r: abs(r - mid))
            notes.append('TM5 内多个脯氨酸，P5.50 取最靠近螺旋中点者 %d，建议人工确认' % p550)
    if p550:
        anchors['5.50'] = p550
    else:
        notes.append('TM5 内未找到脯氨酸，P5.50 缺失')
    if '6.48' in anchors:
        c30 = anchors['6.48'] - 18
        for off in (0, -1, 1, -2, 2, -3, 3):
            r = c30 + off
            if 0 <= r - seq_start < n and seq[r - seq_start] in 'DE':
                anchors['6.30'] = r
                break
        if '6.30' not in anchors:
            notes.append('6.30 位附近 %d 处无 D/E，该受体缺少经典离子锁，报告标注 absent' % c30)
    return anchors, notes


# ---------------------------------------------------------------- 微开关量测规格
# kind: 'sc' 侧链带电/极性原子端，'ring' 芳香环原子，'bb' 主链 N/CA/C/O
MICROSWITCH_SPEC = {
    'PIF': {
        'summary': 'PIF 三元组 x3.40/P5.50/F6.44 的闭合度（激活时崩解）',
        'distances': [('3.40', 'sc', '6.44', 'ring'), ('5.50', 'ring', '6.44', 'ring')],
        'dihedrals': [],
        'pairs': ['x3.40-F6.44', 'P5.50-F6.44'],
    },
    'CWxP': {
        'summary': 'CWxP toggle：W6.48 吲哚取向(chi1/chi2) 与 C6.47 邻近度',
        'distances': [('6.47', 'sc', '6.48', 'ring')],
        'dihedrals': [('6.48', 'chi1'), ('6.48', 'chi2')],
        'pairs': ['C6.47-W6.48'],
    },
    'NPxxY': {
        'summary': 'NPxxY：Y7.53 酚羟基取向(chi1/chi2) 与基序 C-alpha 构象',
        'distances': [('7.50', 'sc', '7.53', 'sc')],
        'dihedrals': [('7.53', 'chi1'), ('7.53', 'chi2')],
        'pairs': ['P7.50-Y7.53'],
        'motif_ca': ['7.49', '7.50', '7.51', '7.52', '7.53'],
    },
    'DRY': {
        'summary': 'DRY：D3.49-R3.50 盐桥与 R3.50-Y3.51 相互作用',
        'distances': [('3.49', 'sc', '3.50', 'sc'), ('3.50', 'sc', '3.51', 'sc')],
        'dihedrals': [('3.50', 'chi1'), ('3.50', 'chi2')],
        'pairs': ['D3.49-R3.50', 'R3.50-Y3.51'],
    },
    'ionic_lock': {
        'summary': '胞内离子锁：D/E3.49 - R3.50 (- E6.30)',
        'distances': [('3.49', 'sc', '3.50', 'sc'), ('3.50', 'sc', '6.30', 'sc')],
        'dihedrals': [],
        'pairs': ['D3.49-R3.50', 'R3.50-E6.30'],
    },
}

MICROSWITCH_ORDER = ['PIF', 'CWxP', 'NPxxY', 'DRY', 'ionic_lock']


def bw_label(resnum, anchors, segments):
    """残基号 -> BW 通用编号（如 127 -> '3.50'）；所在螺旋无锚点时返回 None。"""
    for bw, r in anchors.items():
        h = int(bw.split('.')[0])
        key = 'TM%d' % h
        if key in segments:
            a, b = segments[key]
            if a <= resnum <= b:
                idx = int(bw.split('.')[1]) + (resnum - r)
                if 1 <= idx <= 99:
                    return '%d.%02d' % (h, idx)
    return None


# 内置档案：取自 7W0M CHARMM-GUI GROMACS 拓扑实测序列（残基 26-315），逐条核对过
APLNR_PROFILE = {
    'name': 'APLNR',
    'aliases': ['7W0M', '8XZI', 'APJ', 'apelin receptor', 'APLNR'],
    'source': 'builtin: 7W0M CHARMM-GUI 实测序列 26-315',
    'seq_start': 26,
    'seq_end': 315,
    'seq_len': 290,
    'signature': {'DRY': 126, 'CWxP': 260, 'NPxxY': 305},
    'segments': {
        'Nterm': (26, 29), 'TM1': (30, 54), 'ICL1': (55, 64), 'TM2': (65, 95),
        'ECL1': (96, 99), 'TM3': (100, 130), 'ICL2': (131, 144), 'TM4': (145, 167),
        'ECL2': (168, 182), 'TM5': (183, 220), 'ICL3': (221, 247), 'TM6': (248, 280),
        'ECL3': (281, 288), 'TM7': (289, 313), 'H8': (314, 315),
    },
    'anchors': {
        '3.40': 117, '3.49': 126, '3.50': 127, '3.51': 128,
        '5.50': 213,
        '6.44': 257, '6.47': 260, '6.48': 261, '6.50': 263,
        '7.39': 295, '7.49': 305, '7.50': 306, '7.53': 309,
    },
    'tm_core_align': ['TM1', 'TM2', 'TM3', 'TM4'],
    'notes': [
        '已核对：DRY=126/127/128, CWxP(CWMP)=260/261/262/263, NPxxY(NPFLY)=305..309',
        '6.30 位实测为 R243（非 D/E），该受体缺少经典离子锁，报告标注 absent',
    ],
    'verified': True,
}

BUILTIN_PROFILES = {'APLNR': APLNR_PROFILE}


def match_builtin(seq, seq_start):
    """motif 的【残基号】与内置档案完全一致就命中（不要求构建体长度相同）。

    同一受体常有不同长度的构建体（N/C 端延伸不同）：例如 7W0M 是 26-315 而 8XZI
    是 19-327，但 motif 残基号相同，说明全局编号口径一致，可以直接套用已核对过的
    段定义与 BW 锚点。若强求长度相等，就会掉到粗糙的疏水预测上，把 TM 边界算偏。
    """
    motifs = find_motifs(seq, seq_start)
    for name, prof in BUILTIN_PROFILES.items():
        sig = prof.get('signature', {})
        if not sig:
            continue
        ok = True
        for mname, pos in sig.items():
            if mname not in motifs:
                ok = False
                break
            if min(motifs[mname]['anchors'].values()) != pos:
                ok = False
                break
        if ok:
            return name
    return None


def build_profile(seq, seq_start, name='receptor', prefer_builtin=True):
    """由受体序列构造完整档案：段定义 + BW 锚点 + 微开关残基 + 置信说明。"""
    tag = match_builtin(seq, seq_start) if prefer_builtin else None
    if tag:
        prof = json.loads(json.dumps(BUILTIN_PROFILES[tag]))
        prof['matched_builtin'] = tag
        lo, hi = seq_start, seq_start + len(seq) - 1
        prof['reference_range'] = [prof.get('seq_start'), prof.get('seq_end')]
        if lo != prof.get('seq_start') or hi != prof.get('seq_end'):
            prof.setdefault('notes', []).append(
                '构建体区间 %d-%d 与参考区间 %s-%s 不同；motif 编号一致，段定义与 BW 锚点可直接套用'
                % (lo, hi, prof['reference_range'][0], prof['reference_range'][1]))
        prof['seq_start'] = lo
        prof['seq_end'] = hi
        prof['seq_len'] = len(seq)
        return prof
    helices0, extras, _ = predict_tm_helices(seq)
    helices = [(a + seq_start, b + seq_start) for a, b in helices0]
    segments = build_segments(seq_start, helices)
    if extras:
        first_extra = min(extras, key=lambda r: r[0])
        a, b = first_extra[0] + seq_start, first_extra[1] + seq_start
        tm7 = segments.get('TM7')
        if tm7 and a > tm7[1] and (b - a + 1) >= 8:
            segments['H8'] = (a, b)
    motifs = find_motifs(seq, seq_start)
    anchors, notes = derive_anchors(motifs, seq, seq_start, helices)
    return {
        'name': name,
        'source': 'auto: 序列 motif + Kyte-Doolittle 跨膜预测',
        'seq_start': seq_start,
        'seq_end': seq_start + len(seq) - 1,
        'seq_len': len(seq),
        'signature': {k: min(v['anchors'].values()) for k, v in motifs.items()},
        'segments': segments,
        'anchors': anchors,
        'tm_core_align': ['TM1', 'TM2', 'TM3', 'TM4'],
        'notes': notes,
        'verified': False,
    }


# ---------------------------------------------------------------- 读写与检索
def save_profile(ws, prof, fname='gpcr.json'):
    p = os.path.join(ws, fname)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(prof, f, indent=2, ensure_ascii=False)
    return p


def load_profile(ws, fname='gpcr.json'):
    p = os.path.join(ws, fname)
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    return None


def switch_residues(prof, switch):
    """微开关用到的 {BW: 残基号}；缺位点（如无 6.30）自动略过。"""
    spec = MICROSWITCH_SPEC[switch]
    anchors = prof.get('anchors', {})
    out = {}
    for a, _ka, b, _kb in spec['distances']:
        for bw in (a, b):
            if bw in anchors:
                out[bw] = anchors[bw]
    for bw, _d in spec.get('dihedrals', []):
        if bw in anchors:
            out[bw] = anchors[bw]
    for bw in spec.get('motif_ca', []):
        if bw in anchors:
            out[bw] = anchors[bw]
    return out


def segments_to_str(segments):
    """{'TM1': (30,54), ...} -> 'TM1:30-54,TM2:65-95'（按标准顺序）。"""
    order = ['Nterm', 'TM1', 'ICL1', 'TM2', 'ECL1', 'TM3', 'ICL2', 'TM4', 'ECL2',
             'TM5', 'ICL3', 'TM6', 'ECL3', 'TM7', 'H8', 'Cterm']
    parts = []
    for k in order:
        if k in segments:
            a, b = segments[k]
            parts.append('%s:%d-%d' % (k, a, b))
    for k in sorted(segments):
        if k not in order:
            a, b = segments[k]
            parts.append('%s:%d-%d' % (k, a, b))
    return ','.join(parts)


def parse_segments_str(s):
    """'TM1:30-54,TM2:65-95' -> {'TM1': (30,54), ...}（用于用户覆盖默认段定义）。"""
    out = {}
    for tok in str(s).split(','):
        tok = tok.strip()
        if not tok or ':' not in tok or '-' not in tok:
            continue
        name, rng = tok.split(':', 1)
        a, b = rng.split('-', 1)
        try:
            out[name.strip()] = (int(a), int(b))
        except ValueError:
            continue
    return out


def residues_of(segments, names, include_loops=False):
    """取若干段的残基号并集（默认只取名中列出的段）。"""
    res = []
    for n in names:
        if n in segments:
            a, b = segments[n]
            res.extend(range(a, b + 1))
    return sorted(set(res))


def profile_summary(prof):
    """给 LLM 看的一行式档案摘要。"""
    segs = prof.get('segments', {})
    anch = prof.get('anchors', {})
    verified = '已人工核对' if prof.get('verified') else '自动识别（建议人工确认）'
    lines = [
        '受体：%s（%s，残基 %s-%s，来源 %s）' % (
            prof.get('name', '?'), verified, prof.get('seq_start'), prof.get('seq_end'),
            prof.get('source', '?')),
        '段定义：' + segments_to_str(segs),
        'BW 锚点：' + ', '.join('%s=%s' % (k, v) for k, v in sorted(anch.items())),
    ]
    for note in prof.get('notes', []):
        lines.append('提示：' + note)
    return '\n'.join(lines)
