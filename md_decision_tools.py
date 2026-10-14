# 决策层·证据与执行：把真实文件解析成数字，交给策略层判定，落进决策账本。
#
# 与 md_decision_defs.py 的分工：
#   defs 里只有阈值表与判定逻辑（纯标准库，谁都能复算）；
#   这里负责「把文件读成证据」和「把判定写下来 / 执行下去」。
#
# 三条实测逼出来的约束：
#  1. 两者不能互相 import：本模块 import md_prep_tools，md_prep_tools 绝不能再
#     import 本模块，否则循环。策略层只认 evidence 字典，与本层解耦。
#  2. 证据必须从文件解析出来，宁可缺项（返回「没找到文件」并跳过落账），
#     也不能凭感觉编一个数写进账本 —— 编出来的数后面没人能反证。
#  3. 判定不是终点：ESCALATE 的判定要在返回串最前面打「需要你定夺」标记，
#     图靠这个前缀决定停下来问人。所以账本与返回串用同一套 marker。
#
# 纯标准库 import（numpy/MDAnalysis 一律走 md_tools 的 _py_tool 子进程），
# 与 gpcr_defs.py 同定位，两个 env 都能加载。
import os
import re
import json
import glob

from langchain_core.tools import tool

from md_tools import GMX, _abs, current_workspace, get_workspace_path, _py_tool
import md_prep_tools as PT
import md_decision_defs as D


_HERE = os.path.dirname(os.path.abspath(__file__))
FORCEFIELDS_ROOT = os.path.join(_HERE, "forcefields")

PREP_CFG = "prep_config.json"
PREP_LOG = "prep_log.json"
DEC_LOG = "decision_log.json"

# 残基名集合。写法跨力场家族不通用（CHARMM 的 HSD vs Amber 的 HID），全收。
HIS_NAMES = ("HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP")
CYS_NAMES = ("CYS", "CYX")
ASP_NAMES = ("ASP", "ASH")
GLU_NAMES = ("GLU", "GLH")
WATER_RESNAMES = ("SOL", "TIP3", "TIP3P", "TIP4P", "TIP5P", "SPC", "SPCE")
ION_MOLNAMES = ("NA", "CL", "SOD", "CLA", "K", "MG", "CA")


# ---------------------------------------------------------------------------
# 一、共享助手（与 md_prep_tools 同一套工作区机制）。
# ---------------------------------------------------------------------------
def _ws():
    return os.path.abspath(current_workspace.get())


def _save_json(name, obj):
    p = os.path.join(_ws(), name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return p


def _load_json(name):
    p = os.path.join(_ws(), name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _tail(text, n=1600):
    text = str(text or "")
    return text[-n:] if len(text) > n else text


def _escalate_on():
    # 用户要求「有判断就停下来问」时置 true；缺省不打扰。
    cfg = _load_json(PREP_CFG) or {}
    return bool(cfg.get("escalate", False))


def _append_decision(rec):
    # 追加一条判定到 decision_log.json，返回累计条数。
    data = _load_json(DEC_LOG) or {"decisions": []}
    if not isinstance(data, dict):
        data = {"decisions": []}
    recs = data.get("decisions")
    if not isinstance(recs, list):
        recs = []
    recs.append(rec)
    _save_json(DEC_LOG, {"decisions": recs})
    return len(recs)


def _load_decisions():
    data = _load_json(DEC_LOG) or {}
    recs = data.get("decisions") if isinstance(data, dict) else None
    return list(recs) if isinstance(recs, list) else []


def _marker(rec):
    # 图靠这个前缀决定停下来问人：ESCALATE 必须以它开头。
    return "【需要你定夺】" if rec.get("verdict") == "ESCALATE" else "【已判定】"


# ---------------------------------------------------------------------------
# 二、结构解析（纯文本，不碰 numpy）。
# ---------------------------------------------------------------------------
def _read_pdb_residues(path):
    # 按 PDB 固定列解析 ATOM/HETATM，返回 (residues, atoms)。
    # residues 按 resseq 归并；同一 resseq 保留第一次出现的 resname。
    # 坐标是 PDB 原生的埃，调用方负责换算成 nm。
    residues, order, atoms = [], {}, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
                continue
            name = ln[12:16].strip()
            resname = ln[17:21].strip()
            try:
                resseq = int(ln[22:26])
                pos = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            except ValueError:
                continue
            atoms.append(dict(resname=resname, resseq=resseq, name=name, x=pos[0], y=pos[1], z=pos[2]))
            if resseq in order:
                residues[order[resseq]]["atoms"][name] = pos
            else:
                order[resseq] = len(residues)
                residues.append(dict(resname=resname, resseq=resseq, atoms=dict([(name, pos)])))
    return residues, atoms


def _dist(a, b):
    if a is None or b is None:
        return None
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# 三、质子化态与二硫键证据（工具 1）。
# ---------------------------------------------------------------------------
def _resolve_protein_pdb(protein_pdb=""):
    # 决策产物优先：已写出的 protein_decided.pdb 必须盖过上游 oriented / 传参。
    for name in ("protein_decided.pdb", "protein_oriented.pdb"):
        p = os.path.join(_ws(), name)
        if os.path.exists(p):
            return p
    if protein_pdb:
        p = _abs(protein_pdb)
        if p and os.path.exists(p):
            return p
    cands = sorted(glob.glob(os.path.join(_ws(), "*.pdb")))
    return cands[0] if cands else ""


def _carboxylate_oxygens(residues):
    # ASP/ASH 的 OD1/OD2、GLU/GLH 的 OE1/OE2，供 His 找最近的羧基氧。
    out = []
    for r in residues:
        rn = r["resname"].upper()
        if rn in ASP_NAMES:
            for on in ("OD1", "OD2"):
                out.append((r["resseq"], rn, on, r["atoms"].get(on)))
        elif rn in GLU_NAMES:
            for on in ("OE1", "OE2"):
                out.append((r["resseq"], rn, on, r["atoms"].get(on)))
    return out


@tool
def decide_protonation_scan(protein_pdb="") -> str:
    """【决策·质子化】扫描 His 的羧基近邻与 Cys 的 SG-SG 距离，判定互变异构体与二硫键。

    证据落 decision_evidence_protonation.json，判定追加到 decision_log.json。
    His 的 Nδ1/Nε2 距羧基氧 <= 0.30 nm 判 HSP，否则取中性 HSD；
    Cys 的 SG-SG <= 0.25 nm 判二硫键，0.25-0.35 nm 落灰区时停下来问人。

    Args:
        protein_pdb: 蛋白结构（.pdb）；留空则按 decided > oriented > 工作区首个 .pdb 解析
    """
    src = _resolve_protein_pdb(protein_pdb)
    if not src or not os.path.exists(src):
        return "【已判定】未找到蛋白 PDB，无法扫描质子化态（先跑 prep_orient_protein）。"
    residues, _atoms = _read_pdb_residues(src)
    oxy = _carboxylate_oxygens(residues)

    his_sites = []
    for r in residues:
        rn = r["resname"].upper()
        if rn not in HIS_NAMES:
            continue
        best = None  # (dist_nm, acceptor_res, acceptor_atom)
        for (_rs, arn, an, opos) in oxy:
            for hn in ("ND1", "NE2"):
                d = _dist(r["atoms"].get(hn), opos)
                if d is None:
                    continue
                dnm = d / 10.0
                if best is None or dnm < best[0]:
                    best = (dnm, arn, an)
        his_sites.append(dict(
            resnum=r["resseq"], resname=rn,
            acceptor_res=best[1] if best else None,
            acceptor_atom=best[2] if best else None,
            acceptor_dist_nm=round(best[0], 3) if best else None))

    cys = [r for r in residues if r["resname"].upper() in CYS_NAMES]
    ss_pairs, ss_ambiguous = [], []
    paired = set()
    for i in range(len(cys)):
        for j in range(i + 1, len(cys)):
            d = _dist(cys[i]["atoms"].get("SG"), cys[j]["atoms"].get("SG"))
            if d is None:
                continue
            dnm = d / 10.0
            item = dict(resnum_a=cys[i]["resseq"], resnum_b=cys[j]["resseq"], dist_nm=round(dnm, 3))
            if dnm <= D.SS_BOND_MAX_NM:
                ss_pairs.append(item)
                paired.add(i)
                paired.add(j)
            elif dnm <= 0.35:
                ss_ambiguous.append(item)
                paired.add(i)
                paired.add(j)
    n_free_cys = len(cys) - len(paired)

    ev = dict(n_his=len(his_sites), his_sites=his_sites, ss_pairs=ss_pairs,
              ss_ambiguous=ss_ambiguous, n_cys=len(cys), n_free_cys=n_free_cys)
    _save_json("decision_evidence_protonation.json", ev)
    rec = D.evaluate("protonation", ev, escalate_on=_escalate_on())
    _append_decision(rec)

    lines = [_marker(rec) + "质子化态与二硫键：%s（置信度 %s）" % (rec.get("verdict"), rec.get("confidence"))]
    his_choice = (rec.get("choice") or {}).get("his", {}) or {}
    if his_sites:
        lines.append("组氨酸（%d 个）：" % len(his_sites))
        for s in his_sites:
            got = his_choice.get(s["resnum"], s["resname"])
            if s["acceptor_dist_nm"] is not None:
                near = "最近羧基氧 %s %s %.2f nm" % (s["acceptor_res"], s["acceptor_atom"], s["acceptor_dist_nm"])
            else:
                near = "周围无羧基"
            lines.append("  His %d（%s -> %s）：%s" % (s["resnum"], s["resname"], got, near))
    else:
        lines.append("没有组氨酸。")
    if ss_pairs:
        lines.append("二硫键（%d 对）：" % len(ss_pairs))
        for p in ss_pairs:
            lines.append("  Cys %d - Cys %d，SG-SG %.2f nm" % (p["resnum_a"], p["resnum_b"], p["dist_nm"]))
    else:
        lines.append("未发现二硫键。")
    if ss_ambiguous:
        lines.append("灰区判不准（%d 对，落账为需要你定夺）：" % len(ss_ambiguous))
        for p in ss_ambiguous:
            lines.append("  Cys %d - Cys %d，SG-SG %.2f nm" % (p["resnum_a"], p["resnum_b"], p["dist_nm"]))
    lines.append("游离 Cys %d 个。" % n_free_cys)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 四、净电荷与离子数证据（工具 2）。
# ---------------------------------------------------------------------------
def _parse_top_charges(top_path, charges=None, seen=None):
    # 递归跟 #include，按 moleculetype 归并 atoms 的电荷列（字段 6）。
    if charges is None:
        charges = {}
    if seen is None:
        seen = set()
    ap = os.path.abspath(top_path)
    if ap in seen:
        return charges
    seen.add(ap)
    base = os.path.dirname(ap)
    cur_mol = None
    cur_sect = None
    try:
        f = open(ap, encoding="utf-8", errors="replace")
    except OSError:
        return charges
    with f:
        for ln in f:
            s = ln.strip()
            if s.startswith("#" + "inc" + "lude"):
                parts = s.split()
                tok = parts[-1] if parts else ""
                if tok:
                    _parse_top_charges(os.path.join(base, tok[1:-1]), charges, seen)
                continue
            if s.startswith("[") and s.endswith("]"):
                cur_sect = s.strip("[]").strip().lower()
                continue
            if not s or s.startswith(";"):
                continue
            if cur_sect == "moleculetype":
                cur_mol = s.split()[0]
                charges.setdefault(cur_mol, 0.0)
            elif cur_sect == "atoms" and cur_mol is not None:
                toks = s.split()
                if len(toks) >= 7:
                    try:
                        charges[cur_mol] += float(toks[6])
                    except ValueError:
                        pass
    return charges


def _parse_top_molecules(top_path):
    # 从 topol.top 的 [ molecules ] 段取每个分子 type 的计数。
    mols = []
    in_mols = False
    with open(top_path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            s = ln.strip()
            if s.lower().startswith("[ molecules"):
                in_mols = True
                continue
            if in_mols:
                if s.startswith("["):
                    break
                if s and not s.startswith(";"):
                    w = s.split()
                    if len(w) > 1 and w[1].lstrip("-").isdigit(): mols.append((w[0], int(w[1])))
    return mols


@tool
def decide_charge_audit() -> str:
    """【决策·净电荷】解析 topol.top（递归跟 include）求溶质净电荷，判定中和与离子数。

    证据落 decision_evidence_charge.json，判定追加到 decision_log.json。
    电荷从 atoms 段的电荷列逐原子求和，按 moleculetype 归并，再乘 molecules 段计数；
    水与 Na/Cl 等离子不计入溶质。判定结果用于决定 genion 该补几个离子。
    """
    top = os.path.join(_ws(), "topol.top")
    if not os.path.exists(top):
        return "【已判定】未找到 topol.top，无法核对净电荷。"
    charges = _parse_top_charges(top)
    mols = _parse_top_molecules(top)
    n_water, solute = 0, 0.0
    ions_added, charge_by_molecule, lines = dict(), dict(), []
    for name, cnt in mols:
        u = name.upper()
        q = float(charges.get(name, 0.0))
        if u in WATER_RESNAMES: n_water += cnt; continue
        if u in ("NA", "SOD"): ions_added["NA"] = ions_added.get("NA", 0) + cnt; continue
        if u in ("CL", "CLA"): ions_added["CL"] = ions_added.get("CL", 0) + cnt; continue
        if u in ION_MOLNAMES: continue
        charge_by_molecule[name] = round(q, 4)
        solute += q * cnt
        lines.append("  %-16s x%-6d 电荷/分子 %+g 贡献 %+g" % (name, cnt, q, q * cnt))
    cfg = _load_json(PREP_CFG) or {}
    conc = cfg.get("target_conc_M", cfg.get("salt_conc_M"))
    ev = dict(charge_by_molecule=charge_by_molecule, solute_charge=round(solute, 4),
              n_water=n_water, target_conc_M=conc)
    if ions_added:
        ev["ions_added"] = ions_added
    _save_json("decision_evidence_charge.json", ev)
    rec = D.evaluate("net_charge", ev, escalate_on=_escalate_on())
    _append_decision(rec)
    head = _marker(rec) + "净电荷与离子数：%s（溶质 %+g）" % (rec.get("verdict"), solute)
    tail = "离子计数：%s；水 %d 个。" % (ions_added or "未加离子", n_water)
    return "\n".join([head, "逐分子 type 电荷："] + lines + [tail])


# ---------------------------------------------------------------------------
# 五、配体参数来源证据（工具 3）。
# ---------------------------------------------------------------------------
def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _cgenff_penalty_max():
    log = _load_json(PREP_LOG) or {}
    for w in log.get("warnings", []):
        m = re.search(r"惩罚分：最大\s*([0-9.]+)", str(w))
        if m: return _safe_float(m.group(1))
    return None


def _cgenff_cache_exists():
    return bool(glob.glob(os.path.join(FORCEFIELDS_ROOT, "cgenff_cache", "*.str")))


def _count_ligand_heavy_atoms():
    p = os.path.join(_ws(), "LIG.gro")
    if os.path.exists(p):
        try:
            L = open(p, encoding="utf-8", errors="replace").read().splitlines()
            n = int(L[1].strip())
            return len([x for x in L[2:2 + n] if x[10:15].strip() and not x[10:15].strip().upper().startswith("H")]) or None
        except Exception:
            return None
    p = os.path.join(_ws(), "out.mol2")
    if os.path.exists(p):
        cnt, in_atoms = 0, False
        for ln in open(p, encoding="utf-8", errors="replace"):
            if ln.startswith("@<TRIPOS>ATOM"): in_atoms = True; continue
            if ln.startswith("@<TRIPOS>"): in_atoms = False; continue
            t = ln.split()
            if in_atoms and len(t) > 1 and not t[1].upper().startswith("H"): cnt += 1
        return cnt or None
    return None


def _ligand_present():
    # 本次是否真的带了配体：配置里的 ligand_file、工作区里的 LIG.itp / LIG.acpype 目录，
    # 或 prep_log 里跑成功过的配体参数化步骤，任一命中即算有配体。
    # 纯蛋白体系里这条判定根本无从谈起，必须先挡下来，否则会误报「需要你定夺」。
    cfg = _load_json(PREP_CFG) or {}
    lig = str(cfg.get("ligand_file") or "").strip()
    if lig:
        cand = lig if os.path.isabs(lig) else os.path.join(_ws(), lig)
        if os.path.exists(cand):
            return True
    ws = _ws()
    if os.path.exists(os.path.join(ws, "LIG.itp")):
        return True
    if os.path.isdir(os.path.join(ws, "LIG.acpype")):
        return True
    log = _load_json(PREP_LOG) or {}
    for s in log.get("steps", []):
        if s.get("step") in ("ligand_cgenff", "ligand_gaff") and s.get("ok"):
            return True
    return False


@tool
def decide_ligand_params() -> str:
    """【决策·配体参数】判定配体参数来源（CGenFF 罚分 / GAFF）是否可信。

    无配体时直接返回不适用、不落账。证据落 decision_evidence_ligand_params.json，
    判定追加到 decision_log.json。CGenFF 路线从 prep_log 的告警串解析最大罚分，
    罚分 > 100 停下来问人；GAFF 路线本地全自动、无罚分概念。另统计配体重原子数与
    CGenFF 缓存命中。
    """
    if not _ligand_present():
        return "【已判定】本次未带配体，配体参数判定不适用，流程继续。"
    cfg = _load_json(PREP_CFG) or {}
    route = str(cfg.get("ligand_route") or "cgenff")
    ev = dict(route=route)
    if route == "cgenff":
        ev["penalty_max"] = _cgenff_penalty_max()
        ev["cached"] = _cgenff_cache_exists()
        ev["n_heavy_atoms"] = _count_ligand_heavy_atoms()
    _save_json("decision_evidence_ligand_params.json", ev)
    rec = D.evaluate("ligand_params", ev, escalate_on=_escalate_on())
    _append_decision(rec)
    if route == "cgenff":
        info = "CGenFF 最大罚分 %s；重原子 %s；缓存命中 %s。" % (
            ev.get("penalty_max"), ev.get("n_heavy_atoms"), ev.get("cached"))
    else:
        info = "GAFF 路线，无罚分。"
    return _marker(rec) + "配体参数来源：%s\n%s" % (rec.get("verdict"), info)


# ---------------------------------------------------------------------------
# 六、膜质量证据（工具 4）。
# ---------------------------------------------------------------------------
def _regex_num(pat, text):
    m = re.search(pat, str(text or ""))
    return _safe_float(m.group(1)) if m else None


def _gro_box_z_nm(path):
    L = open(path, encoding="utf-8", errors="replace").read().splitlines()
    t = L[-1].split() if L else []
    if len(t) >= 3: return _safe_float(t[2])
    return None


def _popc_leaflet_counts(path):
    # 直接扫 PDB 行、只认 resname==POPC 的原子，按 (resseq, icode, chain) 归并残基。
    # 不能走 _read_pdb_residues：那个按 resseq 单独归并，而膜 PDB 里 POPC 的残基号
    # 与蛋白各自独立编号、必然撞车，POPC 会被并进/遮在蛋白条目下，计数就取决于
    # 蛋白的 z，实测会假报「上 24 / 下 0」（真值每叶 62、上下对称）。
    if not os.path.exists(path):
        return 0, 0
    zs_by_res = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
                continue
            if ln[17:21].strip() != "POPC":
                continue
            key = (ln[22:26].strip(), ln[26], ln[21])
            try:
                z = float(ln[46:54])
            except ValueError:
                continue
            zs_by_res.setdefault(key, []).append(z)
    up, lo = 0, 0
    for zs in zs_by_res.values():
        if not zs: continue
        if sum(zs) / len(zs) > 0: up += 1
        else: lo += 1
    return up, lo


def _protein_z_span_nm(path):
    res, _a = _read_pdb_residues(path)
    zs = [c[2] for r in res for c in r["atoms"].values()]
    return round((max(zs) - min(zs)) / 10.0, 3) if zs else None


def _tm_and_tilt(path):
    try:
        import numpy as np
        import gpcr_defs as G
        keys, seq, ca = PT._read_pdb_ca(path)
        helices, _ex, _pr = G.predict_tm_helices(seq)
        n_tm = len(helices)
        axes = []
        for a, b in helices[:7]:
            idxs = [i for i in range(a, min(b + 1, len(keys)))]
            if len(idxs) < 6: continue
            pts = np.array([ca[keys[i]] for i in idxs])
            pts = pts - pts.mean(axis=0)
            _u, _s, vt = np.linalg.svd(pts, full_matrices=False)
            v = vt[0]
            if v[2] < 0: v = -v
            axes.append(v)
        if len(axes) < 5: return n_tm, None
        A = np.array(axes)
        A = A - A.mean(axis=0)
        _u, _s, vt = np.linalg.svd(A, full_matrices=False)
        tilt = float(np.degrees(np.arccos(min(1.0, abs(float(vt[0][2]))))))
        return n_tm, round(tilt, 2)
    except Exception:
        return None, None


@tool
def decide_membrane_check() -> str:
    """【决策·膜质量】核对本地铺的 POPC 双层几何（无重叠/上下对称/水层够），判定能不能跑。

    未建膜时直接返回不适用、不落账。证据落 decision_evidence_membrane.json，
    判定追加到 decision_log.json。这是几何摆放检查，不代表膜已经平衡。
    """
    cfg = _load_json(PREP_CFG) or {}
    if not cfg.get("membrane"):
        return "【已判定】本次未建膜，膜质量判定不适用。"
    prot = os.path.join(_ws(), "protein_oriented.pdb")
    if not os.path.exists(prot):
        return "【已判定】未找到 protein_oriented.pdb，跳过膜质量判定。"
    log = _load_json(PREP_LOG) or {}
    detail = ""
    for s in log.get("steps", []):
        if s.get("step") == "build_membrane": detail = s.get("detail", "")
    box_A = _regex_num(r"盒子\s*([0-9.]+)", detail)
    min_A = _regex_num(r"最小脂质-蛋白距离\s*([0-9.]+)", detail)
    n_tm, tilt = _tm_and_tilt(prot)
    memp = os.path.join(_ws(), "membrane.pdb")
    boxp = os.path.join(_ws(), "box.gro")
    up, lo = _popc_leaflet_counts(memp) if os.path.exists(memp) else (0, 0)
    box_z = _gro_box_z_nm(boxp) if os.path.exists(boxp) else None
    span = _protein_z_span_nm(prot)
    ev = dict(n_tm=n_tm, tilt_deg=tilt, min_lipid_protein_A=min_A,
              n_leaflet_upper=up, n_leaflet_lower=lo, box_z_nm=box_z,
              protein_z_span_nm=span, box_side_A=box_A)
    _save_json("decision_evidence_membrane.json", ev)
    rec = D.evaluate("membrane_quality", ev, escalate_on=_escalate_on())
    _append_decision(rec)
    probs = (rec.get("choice") or {}).get("problems", [])
    info = "盒子 %s A；最小脂质-蛋白距离 %s A；上下叶 %d/%d；盒子 z %s nm；TM %s 条，倾角 %s 度。" % (
        box_A, min_A, up, lo, box_z, n_tm, tilt)
    wtxt = ("问题：" + "；".join(probs)) if probs else "几何检查无问题。"
    return _marker(rec) + "膜质量：%s\n%s\n%s" % (rec.get("verdict"), info, wtxt)


# ---------------------------------------------------------------------------
# 七、配体解离证据（工具 5）。
# ---------------------------------------------------------------------------
def _is_derived(name):
    b = os.path.basename(name)
    return b.startswith("analysis_sub") or b.startswith("analysis_sol") or b.startswith("#")


def _find_tpr():
    ws = _ws()
    p = os.path.join(ws, "step5_1.tpr")
    if os.path.exists(p): return p
    c = sorted(glob.glob(os.path.join(ws, "*.tpr")))
    return c[0] if c else ""


def _find_traj():
    ws = _ws()
    p = os.path.join(ws, "merged.xtc")
    if os.path.exists(p): return p
    cands = [c for c in glob.glob(os.path.join(ws, "*.xtc"))
             if os.path.basename(c) != "merged.xtc" and not _is_derived(c)]
    return max(cands, key=os.path.getmtime) if cands else ""


def _read_retention_csv(path):
    rows, idx = [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        h = f.readline().strip().lstrip("#").strip()
        for i, c in enumerate(h.split(",")): idx[c.strip()] = i
        for ln in f:
            s = ln.strip()
            if s and not s.startswith("#"): rows.append([t.strip() for t in s.split(",")])
    return idx, rows


def _col(rows, idx, name):
    j = idx.get(name)
    if j is None: return []
    out = []
    for r in rows:
        if j < len(r): out.append(_safe_float(r[j]))
    return [x for x in out if x is not None]


def _head_mean(v, frac):
    k = max(1, int(round(len(v) * frac)))
    return sum(v[:k]) / k if v else 0.0


def _tail_mean(v, frac):
    k = max(1, int(round(len(v) * frac)))
    return sum(v[-k:]) / k if v else 0.0


def _max_xvg_y(path):
    mx = None
    for ln in open(path, encoding="utf-8", errors="replace"):
        if not ln or ln[0] in "#@&": continue
        t = ln.split()
        if len(t) >= 2:
            y = _safe_float(t[1])
            if y is not None: mx = y if mx is None else max(mx, y)
    return mx


@tool
def decide_dissociation_check() -> str:
    """【决策·配体解离】读 ligand_retention.csv，判定配体是跑掉了还是力场假象。

    没有轨迹/tpr 时不落账。调既有分析 worker 跑出 ligand_retention.csv，
    取接触保留率首尾与末段配体 RMSD/最小距离作证据，判定追加到 decision_log.json。
    """
    traj = _find_traj()
    tpr = _find_tpr()
    if not traj or not tpr:
        return "【已判定】没有轨迹/tpr，配体解离判定跳过。"

    def sm(r):
        return "配体保持分析：%s 帧（stride %s）" % (r.get("n_frames_used"), r.get("stride"))
    _py_tool("ligand_retention", sm, top="", traj=traj)

    csvp = os.path.join(_ws(), "ligand_retention.csv")
    if not os.path.exists(csvp):
        return "【已判定】ligand_retention.csv 未生成（分析环境可能不可用），跳过解离判定。"
    idx, rows = _read_retention_csv(csvp)
    nc = _col(rows, idx, "n_contact_atoms")
    rmsd = _col(rows, idx, "ligand_rmsd_nm")
    mind = _col(rows, idx, "min_dist_nm")
    denom = max(_head_mean(nc, 0.1), 1.0)
    ev = dict(
        retention_start=round(_head_mean(nc, 0.1) / denom, 3),
        retention_end=round(_tail_mean(nc, 0.1) / denom, 3),
        lig_rmsd_final_nm=round(_tail_mean(rmsd, 0.1), 3),
        min_dist_final_nm=round(_tail_mean(mind, 0.1), 3),
        penalty_max=_cgenff_penalty_max(),
        n_frames=len(rows))
    xvg = os.path.join(_ws(), "rmsd.xvg")
    ev["protein_rmsd_max_nm"] = round(_max_xvg_y(xvg), 3) if os.path.exists(xvg) else None
    _save_json("decision_evidence_dissociation.json", ev)
    rec = D.evaluate("ligand_dissociation", ev, escalate_on=_escalate_on())
    _append_decision(rec)
    info = "接触保留率 起始 %.2f -> 末段 %.2f；末段配体 RMSD %.2f nm；最小距离 %.2f nm；%d 帧。" % (
        ev["retention_start"], ev["retention_end"], ev["lig_rmsd_final_nm"], ev["min_dist_final_nm"], len(rows))
    return _marker(rec) + "配体解离：%s\n%s\n依据：%s" % (rec.get("verdict"), info, rec.get("reason"))


# ---------------------------------------------------------------------------
# 八、执行：把质子化决策落到 PDB（工具 6）。
# ---------------------------------------------------------------------------
def _safe_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _latest_decision(did):
    last = None
    for r in _load_decisions():
        if r.get("id") == did: last = r
    return last


@tool
def decide_apply_protonation() -> str:
    """【决策·执行】按最近的质子化决策改写 PDB 的组氨酸残基名（HSD/HSE/HSP）。

    拷贝 protein_decided.pdb（无则 oriented / 工作区首个 .pdb），只改残基名列；
    只写 ATOM 记录（HETATM/配体等一律丢弃，pdb2gmx 只认标准氨基酸）。
    不增删氢，pdb2gmx 会按新名字重建氢。二硫键交给 pdb2gmx 按 SG-SG 自动成键。
    """
    rec = _latest_decision("protonation")
    if rec is None:
        return "【执行报错】还没有质子化决策记录，请先跑 decide_protonation_scan。"
    choice = rec.get("choice") or {}
    his_map = choice.get("his") or {}
    norm = {}
    for k, v in his_map.items():
        ik = _safe_int(k)
        if ik is not None: norm[ik] = str(v)
    src = _resolve_protein_pdb("")
    if not src or not os.path.exists(src):
        return "【执行报错】未找到可改写的蛋白 PDB。"
    dst = os.path.join(_ws(), "protein_decided.pdb")
    renamed = set()
    dropped = 0
    with open(src, encoding="utf-8", errors="replace") as fi, open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            # pdb2gmx 只认标准氨基酸，HETATM（配体/水/离子）会让它因「residue not found」中止，故只写 ATOM。
            if not line.startswith("ATOM"):
                if line.startswith("HETATM"):
                    dropped += 1
                continue
            ik = _safe_int(line[22:26].strip())
            if ik is not None and ik in norm:
                line = line[:17] + ("%-4s" % norm[ik])[:4] + line[21:]
                renamed.add(ik)
            fo.write(line)
    return ("【已判定】已按决策写出 protein_decided.pdb：重命名 %d 个组氨酸（HSD/HSE/HSP）。"
            "仅改残基名，不增删氢；pdb2gmx -ignh 会按新名字重建氢。"
            "二硫键交给 pdb2gmx 按 SG-SG 距离自动成键。"
            "另有 %d 条非标准 HETATM 记录（配体/水/离子）未写入该文件：pdb2gmx 只接受标准氨基酸，"
            "配体由配体工具单独参数化、稍后由 prep_assemble 合并。" % (len(renamed), dropped))


# ---------------------------------------------------------------------------
# 九、账本渲染（工具 7）与导出清单。
# ---------------------------------------------------------------------------
@tool
def decide_review() -> str:
    """【决策·账本】把 decision_log.json 渲染成 summary/decisions.md 并返回 markdown。

    账本里每条判定都带判定结果、选择、依据、反证条件与重做代价；
    ESCALATE 的条目就是需要人定夺的地方。空账本返回「没有需要判断的地方」。
    """
    recs = _load_decisions()
    if not recs:
        return "【已判定】本次没有需要判断的地方（输入已经把关键量都定死了）。"
    md = D.render_ledger(recs)
    d = get_workspace_path("summary")
    with open(os.path.join(d, "decisions.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return md


DECISION_TOOLS = [
    decide_protonation_scan,
    decide_charge_audit,
    decide_ligand_params,
    decide_membrane_check,
    decide_dissociation_check,
    decide_apply_protonation,
    decide_review,
]
