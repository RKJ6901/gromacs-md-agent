# MD 体系准备工具层：蛋白-配体复合物 -> 可运行的 .tpr。
#
# 与 tools.py 的分工：tools 管"已有体系的模拟与分析"，
# 这里管"把体系造出来"。共用同一套工作区机制，产物直接喂给 run_gromacs_md。
#
# 三个实测撞出来的硬约束：
#  1. GMXLIB 是"追加"不是"替换"：设了它 GROMACS 仍能找到自带的 amber/gromos。
#  2. pdb2gmx 对部分力场会弹交互式端基菜单，必须喂 stdin，否则 rc=1 且产物 0 字节。
#  3. 残基命名跨力场家族不通用（CHARMM 的 HSD vs Amber 的 HID），必须改名。
import os
import json
import glob
import tarfile
import subprocess

from langchain_core.tools import tool

from tools import GMX, _abs, current_workspace, get_workspace_path
import prep_defs as P


_HERE = os.path.dirname(os.path.abspath(__file__))
FORCEFIELDS_ROOT = os.path.join(_HERE, "forcefields")

PREP_CFG = "prep_config.json"
PREP_LOG = "prep_log.json"

# CGenFF 的 .str -> GROMACS 转换器需要 networkx，只有 base env 有；
# gmx26 / gmx / AmberTools23 都没有，必须用这个解释器跑。
CONVERTER_PY = os.getenv("PREP_CONVERTER_PY", "/opt/conda/bin/python")
CONVERTER = os.getenv(
    "PREP_CGENFF_CONVERTER",
    "/data/opt-bk/opt/PyAutoFEP/Tools/cgenff_charmm2gmx_py3.py",
)
ACPYPE = os.getenv("PREP_ACPYPE", "/opt/conda/envs/ambertools/bin/acpype")
# acpype 内部要调 antechamber -> sqm（半经验量化，一个配体要跑几分钟）。
# 直接调会因缺 AmberTools 的环境变量而秒退（Cannot properly run sqm），
# 必须用 conda run 把整个 env 的环境带上才跑得动。
CONDA = os.getenv("PREP_CONDA", "/opt/conda/bin/conda")
ACPYPE_ENV = os.getenv("PREP_ACPYPE_ENV", "gmx")



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


def _log(step, ok, detail):
    # 记每一步的结果，供 review_preparation 汇总；不依赖 LLM 记忆。
    rec = _load_json(PREP_LOG) or {"steps": [], "warnings": []}
    rec["steps"].append({"step": step, "ok": bool(ok), "detail": str(detail)[:2000]})
    _save_json(PREP_LOG, rec)
    return rec


def _warn(text):
    rec = _load_json(PREP_LOG) or {"steps": [], "warnings": []}
    rec.setdefault("warnings", []).append(str(text))
    _save_json(PREP_LOG, rec)


def _gmx_env(extra_path=""):
    # GMXLIB 指向项目自带力场目录；GROMACS 会同时保留内置力场的搜索，
    # 所以设它不会让 amber/gromos 找不到。
    # extra_path 用于 acpype：它依赖同环境的 obabel/antechamber，
    # 而 agent 跑在自己的 venv，默认 PATH 里没有这些。
    env = dict(os.environ)
    if os.path.isdir(FORCEFIELDS_ROOT):
        env["GMXLIB"] = FORCEFIELDS_ROOT
    if extra_path:
        env["PATH"] = extra_path + os.pathsep + env.get("PATH", "")
    return env


def _run_prep(cmd, stdin="", timeout=1800, cwd=None, extra_path=""):
    try:
        p = subprocess.run(
            cmd, shell=True, cwd=os.path.abspath(cwd or _ws()),
            input=(stdin.encode() if stdin else b""),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, env=_gmx_env(extra_path),
        )
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, "命令超时 (%ds): %s" % (timeout, cmd)
    except Exception as e:
        return 1, "【系统异常】%s" % e


def _tail(text, n=1200):
    text = str(text or "")
    return text[-n:] if len(text) > n else text


@tool
def prep_set_config(forcefield="charmm36-jul2022", water="tip3p",
                    ligand_route="cgenff", box="dodecahedron",
                    box_padding_nm=1.0, salt="NaCl", salt_conc_M=0.15,
                    membrane=False, lipid="POPC") -> str:
    """【准备·配置】写入本工作区的体系准备参数，并返回兼容性告警。

    必须在其它 prep_* 工具之前调用。后续工具从 prep_config.json 读参数，
    所以只需在这一处指定。告警不是错误：不兼容的组合会照常执行，但会被
    如实写进审查报告，由使用者判断是否接受。

    Args:
        forcefield: 蛋白力场（charmm36-jul2022 为本机唯一自带脂质与 CGenFF 的）
        water: 水模型
        ligand_route: 配体参数路线（cgenff 半自动 / gaff 全本地）
        box: 盒型
        box_padding_nm: 盒子边距（nm）
        salt / salt_conc_M: 盐类型与浓度
        membrane / lipid: 是否建膜与脂质类型
    """
    raw = {
        "forcefield": forcefield, "water": water, "ligand_route": ligand_route,
        "box": box, "box_padding_nm": box_padding_nm, "salt": salt,
        "salt_conc_M": salt_conc_M, "membrane": membrane, "lipid": lipid,
    }
    cfg, notes = P.resolve_config(raw)
    warns = P.compat_check(cfg)
    # 保留已有的输入路径：这两个字段不是本工具的参数，
    # 若直接覆盖会把上游写好的蛋白/配体路径丢掉。
    _old = _load_json(PREP_CFG) or {}
    for _k in ("protein_pdb", "ligand_file"):
        if _old.get(_k):
            cfg[_k] = _old[_k]
    _save_json(PREP_CFG, cfg)
    # 新一轮开始，清掉上一轮的步骤记录与告警，否则同一工作区重跑时
    # 旧的失败项会被算进本次判定，判定就不对了。
    _save_json(PREP_LOG, {"steps": [], "warnings": []})
    for n in notes:
        _warn(n)
    for w in warns:
        _warn(w)
    lines = ["【准备配置已写入 %s】" % PREP_CFG, P.describe(cfg)]
    if notes:
        lines.append("参数修正：")
        lines += ["  - " + n for n in notes]
    lines.append("兼容性告警 %d 条：" % len(warns) if warns else "无兼容性告警。")
    lines += ["  - " + w for w in warns]
    return "\n".join(lines)


def _rename_residues(src_pdb, mapping):
    # 按列改写残基名（PDB 第 18-20 列），返回新路径；mapping 为空则原样返回。
    if not mapping:
        return src_pdb
    dst = os.path.join(_ws(), "protein_renamed.pdb")
    n = 0
    with open(src_pdb, encoding="utf-8", errors="replace") as fi, \
            open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            if line[:6] in ("ATOM  ", "HETATM"):
                r = line[17:20].strip().upper()
                if r in mapping:
                    line = line[:17] + ("%-3s" % mapping[r]) + line[20:]
                    n += 1
            fo.write(line)
    return dst if n else src_pdb


@tool
def prep_protein_topology(protein_pdb="") -> str:
    """【准备·蛋白】pdb2gmx 生成蛋白拓扑与坐标（topol_protein.top / protein.gro）。

    会自动处理两件实测踩过的事：
      1. 残基命名归一化：CHARMM-GUI 的结构用 HSD/HSE，喂 Amber 力场会
         Fatal error: Residue 'HSD' not found —— 这里按目标力场家族自动改名。
      2. pdb2gmx 的交互式端基菜单：统一喂默认应答，避免 rc=1 且产物 0 字节。

    Args:
        protein_pdb: 蛋白结构路径（.pdb）；留空则在工作区里找
    """
    cfg = _load_json(PREP_CFG)
    if not cfg:
        return "【执行报错】尚未配置：请先调用 prep_set_config。"
    _decided = os.path.join(_ws(), "protein_decided.pdb")
    _oriented = os.path.join(_ws(), "protein_oriented.pdb")
    if not protein_pdb and os.path.exists(_decided):
        src = _decided
    elif not protein_pdb and os.path.exists(_oriented):
        src = _oriented
    else:
        src = _abs(protein_pdb or cfg.get("protein_pdb") or "")
    if not src or not os.path.exists(src):
        cands = sorted(glob.glob(os.path.join(_ws(), "*.pdb")))
        src = cands[0] if cands else ""
    if not src or not os.path.exists(src):
        return "【执行报错】未找到蛋白结构 .pdb。"

    ff = P.get_forcefield(cfg["forcefield"]) or {}
    names = P.scan_resnames(src)
    mapping = P.rename_map_for(ff.get("family"), names)
    used = _rename_residues(src, mapping)
    renamed_note = "；残基改名 %s" % mapping if mapping else ""

    cmd = ('"%s" pdb2gmx -f "%s" -ff %s -water %s -o protein.gro '
           '-p topol_protein.top -i posre.itp -ignh') % (
        GMX, used, cfg["forcefield"], cfg["water"])
    rc, out = _run_prep(cmd, stdin="\n" * 30, timeout=1800)

    top = os.path.join(_ws(), "topol_protein.top")
    ok = rc == 0 and os.path.exists(top) and os.path.getsize(top) > 0
    if not ok:
        _log("protein_topology", False, out)
        hint = ""
        if "not found in residue topology database" in out:
            hint = ("\n提示：这是残基命名与本力场不匹配。若输入是 CHARMM-GUI 的结构，"
                    "请确认所选力场家族；本流程只会自动转换组氨酸/半胱氨酸的命名。")
        return "【执行报错】pdb2gmx 失败（rc=%d）%s\n%s" % (rc, hint, _tail(out))
    _log("protein_topology", True, "pdb2gmx ok" + renamed_note)
    return ("【蛋白拓扑已生成】topol_protein.top + protein.gro\n"
            "力场=%s 水=%s%s\n原子数见 protein.gro 首行。"
            % (cfg["forcefield"], cfg["water"], renamed_note))


def _find_str():
    for pat in ("LIG.str", "lig.str", "*.str"):
        hits = sorted(glob.glob(os.path.join(_ws(), pat)))
        if hits:
            return hits[0]
    return ""


# 这些是本流程自己写出的派生结构文件，绝不能当成配体：
# 膜/蛋白/gro/盒/溶剂化产物都属于「体系的其它部分」，不是小分子配体。
# 实测踩过：无配体时兜底取了「工作区第一个 *.pdb」，把 lipids.pdb
# （125 个 POPC）喂给 acpype，一个配体会跑到天荒地老。
_LIGAND_DENY = frozenset([
    "lipids.pdb", "membrane.pdb", "protein.pdb", "protein_oriented.pdb",
    "protein_decided.pdb", "protein.gro", "complex.gro", "box.gro",
    "solv.gro", "ionized.gro", "popc_single.pdb",
])


def _find_ligand():
    # 只认工作区里的 *.mol2：*.pdb / *.sdf 太容易撞上本流程的中间产物。
    for h in sorted(glob.glob(os.path.join(_ws(), "*.mol2"))):
        if os.path.basename(h).lower() in _LIGAND_DENY:
            continue
        return h
    return ""


def _resolve_ligand(ligand_file, cfg):
    # 配体只来自显式来源：函数入参 / prep_config.json 的 ligand_file /
    # 工作区里的 *.mol2；三者都没有就判定为纯蛋白体系。
    # 显式路径同样要剔掉派生中间产物，且不再有「随便拿第一个 *.pdb 兜底」。
    cand = _abs(ligand_file or (cfg or {}).get("ligand_file") or "")
    if cand and os.path.basename(cand).lower() in _LIGAND_DENY:
        cand = ""
    if cand and os.path.exists(cand):
        return cand
    return _find_ligand()


_NO_LIGAND_MSG = ("【跳过配体】未提供配体文件（函数参数与 prep_config.json 的 ligand_file "
                  "均为空，工作区也没有可用的 *.mol2），本次按纯蛋白体系处理，"
                  "跳过配体参数化，链路继续往下走。")


@tool
def prep_ligand_cgenff(ligand_file="", str_file="") -> str:
    """【准备·配体】CGenFF 路线：把 paramchem 返回的 .str 转成 LIG.itp / LIG.gro。

    半自动：赋参那一步在 app.cgenff.com 完成（无免费接口）。本工具负责之后的一切。
    若工作区还没有 .str，会明确停下并给出上传指引，不会假装成功。

    Args:
        ligand_file: 配体文件（.mol2，需与上传给 paramchem 的是同一个）
        str_file: paramchem 返回的 .str；留空则在工作中找 LIG.str / *.str
    """
    cfg = _load_json(PREP_CFG)
    if not cfg:
        return "【执行报错】尚未配置：请先调用 prep_set_config。"
    # 先定有没有配体：没有就按纯蛋白体系跳过（不跑转换器，也不停链）；
    # 有配体但缺 .str 才是需要人上传的那一步。
    lig = _resolve_ligand(ligand_file, cfg)
    if not lig or not os.path.exists(lig):
        return _NO_LIGAND_MSG
    st = _abs(str_file) if str_file else _find_str()
    if not st or not os.path.exists(st):
        return ("【需要人工一步】工作区里没有 CGenFF 的 .str 文件。\n"
                "请在 https://app.cgenff.com 上传配体 mol2，把返回的 .str "
                "保存到工作区后再次触发本步骤。\n"
                "配体文件：%s\n工作区：%s" % (lig or "(未提供)", _ws()))

    ff_dir = os.path.join(FORCEFIELDS_ROOT, (P.get_forcefield(cfg["forcefield"]) or {}).get("ff_dir") or "")
    if not os.path.isdir(ff_dir):
        return "【执行报错】找不到 CHARMM36 力场目录 %s，CGenFF 转换器需要它。" % ff_dir

    cmd = ('"%s" "%s" --mol2 "%s" --str "%s" --forcefield "%s" '
           '--mol_name LIG --output_mol2 out.mol2') % (
        CONVERTER_PY, CONVERTER, lig, st, ff_dir)
    rc, out = _run_prep(cmd, timeout=1800)
    itp = os.path.join(_ws(), "LIG.itp")
    if rc != 0 or not os.path.exists(itp):
        _log("ligand_cgenff", False, out)
        return "【执行报错】CGenFF 转换失败（rc=%d）\n%s" % (rc, _tail(out))

    pen = _parse_penalty(st)
    if pen:
        _warn("CGenFF 惩罚分：%s" % pen)
    _log("ligand_cgenff", True, "LIG.itp 生成；惩罚分 %s" % (pen or "(未解析到)"))
    return ("【配体拓扑已生成】LIG.itp + LIG.gro（力场 %s）\n"
            "CGenFF 惩罚分：%s\n"
            "注意：惩罚分高（>50）表示该分子有参数不理想的原子类型，"
            "结论解释时需要谨慎。" % (cfg["forcefield"], pen or "(未解析到)"))


def _parse_penalty(str_path):
    # 从 CGenFF 的 .str 里收集 PENALTY 数值，返回最大值（最大值决定分子是否可信）。
    vals = []
    try:
        fh = open(str_path, encoding="utf-8", errors="replace")
    except OSError:
        return ""
    with fh:
        for line in fh:
            if "PENALTY" not in line.upper():
                continue
            for tok in line.replace(":", " ").split():
                try:
                    v = float(tok)
                except ValueError:
                    continue
                if 0.0 <= v < 1000.0:
                    vals.append(v)
    if not vals:
        return ""
    return "最大 %.3f（共 %d 个数值）" % (max(vals), len(vals))


@tool
def prep_ligand_gaff(ligand_file="") -> str:
    """【准备·配体】GAFF 路线：用 acpype 全本地生成配体拓扑（免账号）。

    与 CGenFF 互斥：这是一条备选路线。GAFF 属 Amber 家族，配 CHARMM36 蛋白时
    力场不自洽（会在报告里告警）。

    Args:
        ligand_file: 配体文件（.mol2 / .pdb / .sdf）
    """
    cfg = _load_json(PREP_CFG)
    if not cfg:
        return "【执行报错】尚未配置：请先调用 prep_set_config。"
    lig = _resolve_ligand(ligand_file, cfg)
    if not lig or not os.path.exists(lig):
        # 没有配体就不再兜底抓 *.pdb：曾经把 lipids.pdb 当配体喂给 acpype 跑不停。
        return _NO_LIGAND_MSG
    if not os.path.exists(ACPYPE):
        return "【执行报错】未找到 acpype：%s" % ACPYPE

    cmd = '"%s" run --no-capture-output -n %s acpype -i "%s" -b LIG -n 0 -a gaff2 -o gmx' % (CONDA, ACPYPE_ENV, lig)
    rc, out = _run_prep(cmd, timeout=3600)
    itp = os.path.join(_ws(), "LIG.acpype", "LIG_GMX.itp")
    gro = os.path.join(_ws(), "LIG.acpype", "LIG_GMX.gro")
    if rc != 0 or not os.path.exists(itp):
        _log("ligand_gaff", False, out)
        return "【执行报错】acpype 失败（rc=%d）\n%s" % (rc, _tail(out))
    _log("ligand_gaff", True, "acpype ok")
    return ("【配体拓扑已生成】%s\n坐标：%s\n（GAFF/Amber 家族）"
            % (os.path.relpath(itp, _ws()), os.path.relpath(gro, _ws())))



def _split_ligand_itp(ligand_itp):
    # GROMACS 要求 [ atomtypes ] 出现在任何 [ moleculetype ] 之前。
    # acpype 生成的 itp 常把 atomtypes 和 moleculetype 放在同一个文件里，
    # 直接 include 到蛋白之后会报 "Invalid order for directive atomtypes"。
    # 所以拆成两个文件：atomtypes 单独一个，插到力场 include 之后。
    with open(ligand_itp, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    types, rest, cur = [], [], None
    for ln in lines:
        st = ln.strip().lower()
        if st.startswith("[") and st.endswith("]"):
            cur = st
        if cur in ("[ atomtypes ]", "[atomtypes]"):
            types.append(ln)
        else:
            rest.append(ln)
    at_p = os.path.join(_ws(), "LIG_atomtypes.itp")
    body_p = os.path.join(_ws(), "LIG_body.itp")
    with open(at_p, "w", encoding="utf-8") as f:
        f.write("; ligand atomtypes (由 prep_assemble 拆出)\n")
        f.write("".join(types) if types else "; (无 atomtypes 段)\n")
    with open(body_p, "w", encoding="utf-8") as f:
        f.write("".join(rest))
    return at_p, body_p, bool(types)


@tool
def prep_assemble(ligand_resname="LIG", ligand_itp="", ligand_gro="") -> str:
    """【准备·组装】把蛋白拓扑与配体拓扑拼成完整的 topol.top。

    做法：在 pdb2gmx 产出的 topol_protein.top 里插入 LIG 的 #include，
    并在 [ molecules ] 段加一行配体。不重写力场段落。

    Args:
        ligand_resname: 配体残基名（默认 LIG）
        ligand_itp / ligand_gro: 配体拓扑与坐标；留空则按路线自动找
    """
    cfg = _load_json(PREP_CFG)
    if not cfg:
        return "【执行报错】尚未配置：请先调用 prep_set_config。"
    pt = os.path.join(_ws(), "topol_protein.top")
    if not os.path.exists(pt):
        return "【执行报错】缺少 topol_protein.top，请先跑 prep_protein_topology。"

    route = cfg.get("ligand_route")
    if not ligand_itp:
        ligand_itp = os.path.join(_ws(), "LIG.itp") if route == "cgenff" else \
            os.path.join(_ws(), "LIG.acpype", "LIG_GMX.itp")
    if not ligand_gro:
        ligand_gro = os.path.join(_ws(), "LIG.gro") if route == "cgenff" else \
            os.path.join(_ws(), "LIG.acpype", "LIG_GMX.gro")
    has_lig = os.path.exists(ligand_itp)
    if not has_lig and (cfg.get("ligand_file") or "").strip():
        return "【执行报错】指定了配体文件，但没有生成出拓扑：%s" % ligand_itp

    at_p, body_p, has_types = _split_ligand_itp(ligand_itp) if has_lig else ("", "", False)
    with open(pt, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    at_inc = ('#include "%s"' % os.path.relpath(at_p, _ws())) + chr(10) if at_p else ""
    body_inc = ('#include "%s"' % os.path.relpath(body_p, _ws())) + chr(10) if body_p else ""
    n_lip = 0
    if cfg.get("membrane"):
        _lip = os.path.join(FORCEFIELDS_ROOT, "lipids", "POPC.itp")
        if not os.path.exists(_lip):
            return "【执行报错】缺少 POPC.itp：%s" % _lip
        body_inc += '#include "%s"\n' % _lip
        _lp = os.path.join(_ws(), "lipids.pdb")
        _mp = os.path.join(_ws(), "membrane.pdb")
        if not os.path.exists(_mp) or not os.path.exists(_lp):
            return "【执行报错】配置开了建膜，但缺 membrane.pdb / lipids.pdb，请先跑 prep_build_membrane。"
        _nl = sum(1 for _l in open(_lp, encoding="utf-8", errors="replace")
                  if _l.startswith("ATOM") and _l[17:21].strip() == "POPC")
        n_lip = _nl // 134
        n_lip = _nl // 134
    out, in_mols = [], False
    put_types, put_body = (not has_types) or not has_lig, not (has_lig or n_lip)
    for ln in lines:
        out.append(ln)
        low = ln.strip().lower()
        if (not put_types) and low.startswith("#include") and "forcefield.itp" in low:
            out.append("; 配体原子类型：必须早于任何 moleculetype，否则 GROMACS 报 Invalid order\n")
            out.append(at_inc)
            put_types = True
        if (not put_body) and low.startswith("[ molecules"):
            out.insert(len(out) - 1, "\n; 配体拓扑（由 prep_assemble 插入）\n")
            out.insert(len(out) - 1, body_inc)
            put_body = True
            in_mols = True
    if not put_body:
        return "【执行报错】topol_protein.top 里没找到 [ molecules ] 段，无法插入配体。"
    # 2) 在 [ molecules ] 末尾追加配体
    txt = "".join(out)
    if not txt.endswith("\n"):
        txt += "\n"
    if has_lig:
        txt += "%-20s %d\n" % (ligand_resname, 1)
    if n_lip:
        txt += "%-20s %d\n" % ("POPC", n_lip)
    with open(os.path.join(_ws(), "topol.top"), "w", encoding="utf-8") as f:
        f.write(txt)

    _pg = os.path.join(_ws(), "protein.gro")
    # 3) 坐标来源
    if n_lip:
        # 膜体系：坐标直接用 membrane.pdb（复合物 + 脂质，同一参考系）。
        # 注意：这一路要求输入的 protein_pdb 本身就是复合物，否则蛋白与配体
        # 会各自停在各自的坐标系里 —— 定向只旋转了它拿到的那个文件。
        rc, out = _run_prep('"%s" editconf -f lipids.pdb -o lipids.gro' % GMX, timeout=900)
        if rc != 0 or not os.path.exists(os.path.join(_ws(), "lipids.gro")):
            _log("assemble", False, out)
            return "【执行报错】lipids.pdb 转 gro 失败（rc=%d）\n%s" % (rc, _tail(out))
        if not os.path.exists(_pg):
            return "【执行报错】缺少 protein.gro，请先跑 prep_protein_topology。"
        merged = [_read_gro(_pg)]
        if has_lig and os.path.exists(ligand_gro):
            merged.append(_read_gro(ligand_gro))
        merged.append(_read_gro(os.path.join(_ws(), "lipids.gro")))
        atoms, box = [], None
        for a, b in merged:
            atoms += a
            box = b
        with open(os.path.join(_ws(), "complex.gro"), "w", encoding="utf-8") as f:
            f.write("membrane complex\n%5d\n" % len(atoms))
            f.write("\n".join(atoms) + "\n" + box + "\n")
        _log("assemble", True, "topol.top + complex.gro，含 %d 个 POPC" % n_lip)
        return ("【体系已组装·膜】topol.top + complex.gro\n"
                "蛋白 + 配体 + POPC %d 个分子\n"
                "下一步：prep_build_box -> prep_solvate（会自动剔除膜内的水）" % n_lip)
    # 3) 合并蛋白与配体坐标 -> complex.gro
    pg = os.path.join(_ws(), "protein.gro")
    if not os.path.exists(pg) or (has_lig and not os.path.exists(ligand_gro)):
        _log("assemble", True, "topol.top 写好，但坐标未合并（缺 gro）")
        return "【已写 topol.top】但缺 protein.gro 或配体 gro，坐标未合并。"

    # 合并坐标：gro 是固定格式（标题/原子数/逐原子行/盒子行），直接拼接。
    # pdb2gmx 不会平移坐标，所以蛋白与配体保持同一参考系，拼接即得复合物。

    pa, box = _read_gro(pg)
    la, _ = _read_gro(ligand_gro)
    merged = ["complex: protein + ligand", "%5d" % (len(pa) + len(la))] + pa + la + [box]
    with open(os.path.join(_ws(), "complex.gro"), "w", encoding="utf-8") as f:
        f.write("\n".join(merged) + "\n")
    _log("assemble", True, "topol.top + complex.gro (%d 原子)" % (len(pa) + len(la)))
    return ("【体系已组装】topol.top + complex.gro\n"
            "蛋白 %d 原子 + 配体 %d 原子 = %d\n"
            "下一步：prep_build_box -> prep_solvate -> prep_ionize -> prep_grompp"
            % (len(pa), len(la), len(pa) + len(la)))


@tool
def prep_build_box() -> str:
    """【准备·建盒】editconf 给复合物建盒（形状与边距取自 prep_config.json）。"""
    cfg = _load_json(PREP_CFG)
    if not cfg:
        return "【执行报错】尚未配置：请先调用 prep_set_config。"
    src = os.path.join(_ws(), "complex.gro")
    if not os.path.exists(src):
        return "【执行报错】缺少 complex.gro，请先跑 prep_assemble。"
    _b = _membrane_box()
    if _b:
        cmd = '"%s" editconf -f complex.gro -o box.gro -c -box %s %s %s' % (
            GMX, _b[0], _b[1], _b[2])
    else:
        cmd = '"%s" editconf -f complex.gro -o box.gro -c -d %s -bt %s' % (
            GMX, cfg.get("box_padding_nm", 1.0), cfg.get("box", "dodecahedron"))
    rc, out = _run_prep(cmd, timeout=600)
    if rc != 0 or not os.path.exists(os.path.join(_ws(), "box.gro")):
        _log("build_box", False, out)
        return "【执行报错】editconf 失败（rc=%d）\n%s" % (rc, _tail(out))
    _log("build_box", True, "box.gro")
    return "【盒子已建】box.gro（%s, 边距 %s nm）\n%s" % (
        cfg.get("box"), cfg.get("box_padding_nm"), _tail(out, 500))


@tool
def prep_solvate() -> str:
    """【准备·溶剂化】solvate 往盒子里加水（水模型取自 prep_config.json）。"""
    cfg = _load_json(PREP_CFG)
    if not cfg:
        return "【执行报错】尚未配置：请先调用 prep_set_config。"
    if not os.path.exists(os.path.join(_ws(), "box.gro")):
        return "【执行报错】缺少 box.gro，请先跑 prep_build_box。"
    cmd = ('"%s" solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top') % GMX
    rc, out = _run_prep(cmd, timeout=1200)
    if rc != 0 or not os.path.exists(os.path.join(_ws(), "solv.gro")):
        _log("solvate", False, out)
        return "【执行报错】solvate 失败（rc=%d）\n%s" % (rc, _tail(out))
    if _membrane_on():
        kept, dropped = _strip_water_in_membrane()
        _log("solvate", True, "solv.gro；膜内剔除 %d 个水，保留 %d" % (dropped, kept))
        return ("【已溶剂化·膜】solv.gro\n"
                "膜内剔除 %d 个水（双层内部不能有水），保留 %d 个\n"
                "水模型 %s" % (dropped, kept, cfg.get("water")))
    _log("solvate", True, "solv.gro")
    return "【已溶剂化】solv.gro（水模型 %s）\n%s" % (cfg.get("water"), _tail(out, 500))


# CHARMM-GUI 标准的力场设置（来自其 step5.0_minimization.mdp 实测值）。
# CHARMM 力场必须用 Force-switch 截断 + 1.2 nm，不能沿用默认的 cutoff。
_MDP_TMPL = """integrator              = steep
nsteps                  = 5000
emtol                   = 1000.0
cutoff-scheme           = Verlet
nstlist                 = 10
rlist                   = 1.2
vdwtype                 = Cut-off
vdw-modifier            = Force-switch
rvdw_switch             = 1.0
rvdw                    = 1.2
coulombtype             = PME
rcoulomb                = 1.2
constraints             = h-bonds
constraint_algorithm    = LINCS
"""


def _write_mdp(name="prep_minim.mdp"):
    p = os.path.join(_ws(), name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(_MDP_TMPL)
    return p

@tool
def prep_ionize() -> str:
    """【准备·加离子】genion 按目标浓度加盐并中和净电荷（默认 -neutral）。

    中和是必须的：净电荷不为零的体系在 PME 下会出现人为的静电伪影。
    """
    cfg = _load_json(PREP_CFG)
    if not cfg:
        return "【执行报错】尚未配置：请先调用 prep_set_config。"
    if not os.path.exists(os.path.join(_ws(), "solv.gro")):
        return "【执行报错】缺少 solv.gro，请先跑 prep_solvate。"
    mdp = _write_mdp()

    _g = '"%s" grompp -f prep_minim.mdp -c solv.gro -p topol.top -o ions.tpr' % GMX
    rc, out = _run_prep(_g, timeout=900)
    if rc != 0 and "Too many warnings" in out:
        # 加离子前体系必然带净电荷，grompp 必然报这一条。
        # 只放行"仅此一条 net charge"的情况 —— 这是它与盲开 -maxwarn 的区别：
        # 只要还有别的警告、或不止一条，就照旧失败。
        _wl = [x for x in out.splitlines() if x.strip().startswith("WARNING")]
        if len(_wl) == 1 and "net charge" in out:
            _warn("加离子前的 grompp 接受了 net charge 警告（构造上必然，随后由 genion 中和）")
            rc, out = _run_prep(_g + " -maxwarn 1", timeout=900)
    if rc != 0 or not os.path.exists(os.path.join(_ws(), "ions.tpr")):
        _log("ionize", False, out)
        return "【执行报错】加离子前的 grompp 失败（rc=%d）\n%s" % (rc, _tail(out))

    cmd = ('"%s" genion -s ions.tpr -o ionized.gro -p topol.top -pname NA -nname CL '
           '-conc %s -neutral') % (GMX, cfg.get("salt_conc_M", 0.15))
    rc, out = _run_prep(cmd, stdin="SOL\n", timeout=900)
    if rc != 0 or not os.path.exists(os.path.join(_ws(), "ionized.gro")):
        _log("ionize", False, out)
        return "【执行报错】genion 失败（rc=%d）\n%s" % (rc, _tail(out))
    _log("ionize", True, "ionized.gro")
    return "【已加离子并中和】ionized.gro（%s %s M）\n%s" % (
        cfg.get("salt"), cfg.get("salt_conc_M"), _tail(out, 600))

@tool
def prep_grompp() -> str:
    """【准备·出 tpr】grompp 产出 step5_1.tpr（可直接跑的最小化输入）。

    产物特意命名为 step5_1.tpr —— 与 CHARMM-GUI 惯例一致，
    run_gromacs_md 默认就找这个名字，所以准备完能无缝接上模拟。
    """
    cfg = _load_json(PREP_CFG)
    if not cfg:
        return "【执行报错】尚未配置：请先调用 prep_set_config。"
    if not os.path.exists(os.path.join(_ws(), "ionized.gro")):
        return "【执行报错】缺少 ionized.gro，请先跑 prep_ionize。"
    _write_mdp()
    cmd = ('"%s" grompp -f prep_minim.mdp -c ionized.gro -p topol.top '
           '-o step5_1.tpr') % GMX
    rc, out = _run_prep(cmd, timeout=900)
    tpr = os.path.join(_ws(), "step5_1.tpr")
    if rc != 0 or not os.path.exists(tpr) or os.path.getsize(tpr) == 0:
        _log("grompp", False, out)
        return "【执行报错】grompp 失败（rc=%d）\n%s" % (rc, _tail(out))
    warns = [l for l in out.splitlines() if "WARNING" in l.upper()]
    for w in warns:
        _warn("grompp: " + w.strip())
    _log("grompp", True, "step5_1.tpr（%d 警告）" % len(warns))
    return ("【可运行输入已生成】step5_1.tpr（%.1f MB）\n"
            "grompp 警告 %d 条（已记入审查报告）\n"
            "下一步：prep_review 出审查报告，之后可直接交给 run_gromacs_md。"
            % (os.path.getsize(tpr) / 1e6, len(warns)))


def _count_molecules():
    # 从 topol.top 的 [ molecules ] 段统计分子数与名称。
    p = os.path.join(_ws(), "topol.top")
    out, in_mols = [], False
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8", errors="replace") as f:
        for ln in f:
            s = ln.strip()
            if s.lower().startswith("[ molecules"):
                in_mols = True
                continue
            if in_mols:
                if s.startswith("["):
                    break
                if s and not s.startswith(";"):
                    out.append(s)
    return out


@tool
def prep_review() -> str:
    """【准备·审查】把本次准备的全部判断依据写成 summary/prep_review.md。

    这是本流程的产物之一，不是可选的装饰：准备阶段的错误（力场不自洽、
    残基名不匹配、膜未平衡、盒子过小）都不会让程序崩，只会让后续模拟
    悄悄偏掉。这份报告的作用是把它们摆到明面上，让使用者能判断能不能用。
    """
    cfg = _load_json(PREP_CFG)
    if not cfg:
        return "【执行报错】尚未配置：请先调用 prep_set_config。"
    log = _load_json(PREP_LOG) or {"steps": [], "warnings": []}
    # 判定档位：PASS / PASS WITH LIMITATIONS / FAIL。
    # 单看"流程跑完了"不算通过 —— 兼容性告警与失败步骤都降级。
    _steps = log.get("steps", [])
    _failed = [x for x in _steps if not x.get("ok")]
    _done = any(x.get("step") == "grompp" and x.get("ok") for x in _steps)
    _warns_all = [w for w in log.get("warnings", [])] + P.compat_check(cfg)
    if _failed or not _done:
        _verdict = "FAIL"
    elif _warns_all:
        _verdict = "PASS WITH LIMITATIONS"
    else:
        _verdict = "PASS"
    L = ["# 体系准备审查报告", "", "**判定：%s**" % _verdict, ""]
    L.append("## 一、本次配置")
    L.append(P.describe(cfg))
    L.append("")
    L.append("## 二、兼容性告警")
    # 兼容性告警自己算一遍，不依赖"有没有人先调过 prep_set_config"：
    # 图里的准备链条是直接读 prep_config.json 的，那条路不会经过配置工具。
    warns = list(log.get("warnings", []))
    for _w in P.compat_check(cfg):
        if _w not in warns:
            warns.append(_w)
    L += (["- " + w for w in warns] if warns else ["（无）"])
    L.append("")
    L.append("## 三、执行记录")
    for s in log.get("steps", []):
        L.append("- [%s] %s: %s" % ("OK" if s.get("ok") else "失败", s.get("step"), s.get("detail", "")[:300]))
    L.append("")

    L.append("## 四、体系组成")
    mols = _count_molecules()
    if mols:
        L += ["- " + m for m in mols]
    else:
        L.append("- （未找到 topol.top 或 [ molecules ] 段）")
    tpr = os.path.join(_ws(), "step5_1.tpr")
    if os.path.exists(tpr):
        L.append("- 产物 step5_1.tpr：%.1f MB" % (os.path.getsize(tpr) / 1e6))
    L.append("")
    L.append("## 五、这次准备不能保证什么")
    L.append("- 蛋白质子化状态由 pdb2gmx 按 pH 默认值判定，**未做 pKa 计算**；")
    L.append("  对活性位点组氨酸/半胱氨酸敏感的项目需要人工核对。")
    L.append("- 未做能量最小化与平衡；step5_1.tpr 是最小化的**输入**，不是平衡后的体系。")
    route = (P.get_ligand_route(cfg.get("ligand_route")) or {})
    if route.get("family") != (P.get_forcefield(cfg.get("forcefield")) or {}).get("family"):
        L.append("- 蛋白与配体来自**不同力场家族**，混合力场的定量结论需谨慎。")
    if cfg.get("membrane"):
        L.append("- **膜未平衡**：只做了几何摆放，不含多轮平衡/取向优化/脂质对称化。")
    if not os.path.exists(tpr):
        L.append("- 本次没有产出 step5_1.tpr（流程未走完），上述组成仅供参考。")
    L.append("")
    text = "\n".join(L)
    d = get_workspace_path("summary")
    p = os.path.join(d, "prep_review.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    _log("review", True, "summary/prep_review.md")
    return "【审查报告已生成】summary/prep_review.md\n\n" + text




# ---------------------------------------------------------------
# 膜插入：本地建 POPC 双层。
#
# 先说清楚达不到什么：CHARMM-GUI Membrane Builder 会做多轮平衡、蛋白取向优化、
# 脂质分布对称化，本流程只做几何摆放 + 最小化输入生成。所以产出的是"待平衡的
# 初始结构"，不是可用的膜体系 —— 审查报告里会把它写成硬结论。
#
# 设计参照 gromacs-electrolyte-workflow 的门禁做法：
# 固定随机种子、明确分子数、记录请求盒子与最小距离、输入输出留痕。
# ---------------------------------------------------------------

LIPID_LIB_NAME = "lipid_lib.tar.gz"


def _find_lipid_lib():
    # 找一个可用的 CHARMM-GUI 脂质库（不同体系各带一份，内容相同）。
    env = os.getenv("PREP_LIPID_LIB", "")
    if env and os.path.exists(env):
        return env
    roots = ["/data/JRK/MD", "/data/JRK/Otherwork", "/data/JRK"]
    for r in roots:
        if not os.path.isdir(r):
            continue
        for root, _d, files in os.walk(r):
            if LIPID_LIB_NAME in files:
                return os.path.join(root, LIPID_LIB_NAME)
    return ""




def _build_bilayer_grid(popc_pdb, n_per_leaflet, box_xy, seed):
    import math
    import random
    rnd = random.Random(int(seed))
    tmpl = []
    txt = open(popc_pdb, encoding="utf-8", errors="replace").read().splitlines()
    tmpl = [(l[12:16].strip(), float(l[30:38]), float(l[38:46]), float(l[46:54]))
            for l in txt if l.startswith("ATOM")]
    z_n = tmpl[0][3]
    z_head = 22.0
    dz = z_head - z_n
    sp = 9.2  # 目标脂质面积约 85 A^2；低于此相邻头基会重叠
    side = max(1, int(float(box_xy) / sp))
    out = []
    pts = [(a, b) for a in range(side) for b in range(side)][:int(n_per_leaflet)]
    if not pts:
        return []
    for leaf in (1, -1):
        for (i, j) in pts:
                x0 = -box_xy / 2.0 + sp * (i + 0.5)
                y0 = -box_xy / 2.0 + sp * (j + 0.5)
                x0 += rnd.uniform(-1.0, 1.0)
                y0 += rnd.uniform(-1.0, 1.0)
                for (nm, tx, ty, tz) in tmpl:
                    zz = leaf * (tz + dz)
                    out.append((nm, x0 + tx, y0 + ty, zz))
    return out


def _membrane_on():
    cfg = _load_json(PREP_CFG) or {}
    return bool(cfg.get("membrane")) and os.path.exists(os.path.join(_ws(), "membrane.pdb"))


def _membrane_box():
    # 膜体系盒子：XY 按复合物实际跨度留边距，Z 按脂质跨度再加两侧水层。
    p = os.path.join(_ws(), "complex.gro")
    if not os.path.exists(p):
        return None
    L = open(p, encoding="utf-8", errors="replace").read().splitlines()
    try:
        n = int(L[1].strip())
    except Exception:
        return None
    xs, ys, zs = [], [], []
    for ln in L[2:2 + n]:
        xs.append(float(ln[20:28]))
        ys.append(float(ln[28:36]))
        zs.append(float(ln[36:44]))
    if not xs:
        return None
    return (round(max(xs) - min(xs) + 0.8, 3),
            round(max(ys) - min(ys) + 0.8, 3),
            round(max(zs) - min(zs) + 5.0, 3))
def _strip_water_in_membrane(z_cut=2.2):
    # 膜内不能有水。gmx solvate 会把整个盒子灌满，包括双层内部，必须剔除。
    # 判据：水的氧原子 |z| 小于脂质头基平面（约 2.2 nm）就认为在膜内。
    p = os.path.join(_ws(), "solv.gro")
    if not os.path.exists(p):
        return 0, 0
    L = open(p, encoding="utf-8", errors="replace").read().splitlines()
    try:
        n = int(L[1].strip())
    except Exception:
        return 0, 0
    body = L[2:2 + n]
    box = L[2 + n] if len(L) > 2 + n else "   1.0   1.0   1.0"
    out, n_sol, dropped = [], 0, 0
    i = 0
    while i < len(body):
        ln = body[i]
        if ln[5:10].strip() != "SOL":
            out.append(ln)
            i += 1
            continue
        rid = ln[0:5]
        j = i
        while j < len(body) and body[j][5:10].strip() == "SOL" and body[j][0:5] == rid:
            j += 1
        chunk = body[i:j]
        zz = None
        for c in chunk:
            if c[10:15].strip() in ("OW", "O"):
                zz = float(c[36:44])
                break
        if zz is None:
            zz = float(chunk[0][36:44])
        if abs(zz) < z_cut:
            dropped += 1
        else:
            n_sol += 1
            out.extend(chunk)
        i = j
    fixed = []
    res_id = 0
    atom_id = 0
    cur = None
    for ln in out:
        key = ln[0:5] + ln[5:10]
        if key != cur:
            cur = key
            res_id += 1
        atom_id += 1
        fixed.append("%5d%-5s%-5s%5d" % (res_id, ln[5:10], ln[10:15], atom_id) + ln[20:])
    with open(p, "w", encoding="utf-8") as f:
        f.write("membrane solvated\n")
        f.write("%5d\n" % len(fixed))
        f.write("\n".join(fixed) + "\n")
        f.write(box + "\n")

    # 同步 [ molecules ] 里的 SOL 数，否则 grompp 会因原子数对不上而失败。
    tp = os.path.join(_ws(), "topol.top")
    lines2 = []
    for ln in open(tp, encoding="utf-8", errors="replace"):
        s = ln.split()
        if len(s) == 2 and s[0] == "SOL" and s[1].isdigit():
            lines2.append("%-20s %d\n" % ("SOL", n_sol))
        else:
            lines2.append(ln)
    open(tp, "w", encoding="utf-8").write("".join(lines2))
    return n_sol, dropped


def _read_gro(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        L = f.read().splitlines()
    n = int(L[1].strip())
    return L[2:2 + n], L[2 + n] if len(L) > 2 + n else "   1.0   1.0   1.0"


def _pdb_atom(serial, name, resname, resseq, x, y, z):
    # 按列定位写 PDB 行。不能用 %3s 拼：Python 的字符串宽度只补齐不截断，
    # 4 字符的 POPC 会把后面的列整体推歪，坐标列错位会让下游解析不了。
    s = [" "] * 66
    s[0:6] = list("ATOM  ")
    s[6:11] = list("%5d" % int(serial))
    s[12:16] = list(("%-4s" % str(name))[:4])
    s[17:21] = list(("%-4s" % str(resname))[:4])
    s[22:26] = list("%4d" % int(resseq))
    s[30:38] = list("%8.3f" % float(x))
    s[38:46] = list("%8.3f" % float(y))
    s[46:54] = list("%8.3f" % float(z))
    s[54:60] = list("  1.00")
    s[60:66] = list("  0.00")
    return "".join(s)
def _popc_pdb(dest):
    # 从脂质库里取一个 POPC 构象，转成 PDB（resname 用 POPC，与 CHARMM36 一致）。
    # 构象选择是确定性的：取排序后的第一个，保证同样输入得到同样结果。
    lib = _find_lipid_lib()
    if not lib:
        return None, "未找到 %s（可用 PREP_LIPID_LIB 指定）" % LIPID_LIB_NAME
    want = "lipid_lib/popc/conf2/"
    lines = []
    with tarfile.open(lib, "r:gz") as tf:
        names = sorted(n for n in tf.getnames() if n.startswith(want) and n.endswith(".crd"))
        if not names:
            return None, "脂质库里没有 %s 下的 .crd" % want
        fh = tf.extractfile(names[0])
        for raw in fh.read().decode("utf-8", "replace").splitlines():
            f = raw.split()
            if len(f) < 7 or not f[0].isdigit():
                continue
            lines.append(_pdb_atom(int(f[0]), f[3], "POPC", 1, float(f[4]), float(f[5]), float(f[6])))
    if not lines:
        return None, "解析 %s 得到 0 个原子" % names[0]
    with open(dest, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\nEND\n")
    return dest, "取自 %s，%d 个原子" % (os.path.basename(names[0]), len(lines))


_AA3 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
    'HID': 'H', 'HIE': 'H', 'HIP': 'H',
}


def _read_pdb_ca(path):
    # 返回 (残基键列表, 序列, 每个残基的 CA 坐标)。
    order, seq, ca = [], [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if not ln.startswith("ATOM"):
                continue
            rn = ln[17:20].strip().upper()
            key = (int(ln[22:26]), ln[26].strip())
            if key not in ca:
                order.append(key)
                seq.append(_AA3.get(rn, "X"))
                ca[key] = None
            if ln[12:16].strip() == "CA":
                ca[key] = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    return [k for k in order if ca[k] is not None], "".join(seq), ca


def _rot_axis_to_z(axis):
    # 构造把 axis 旋到 +z 的旋转矩阵（Rodrigues）。
    import numpy as np
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return np.eye(3)
    a = a / n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, z)
    c = float(np.dot(a, z))
    if float(np.linalg.norm(v)) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx.dot(vx) * (1.0 / (1.0 + c))


def _orient_protein(src_pdb, dst_pdb):
    # 用跨膜螺旋束的主轴把蛋白摆正（轴对齐 z），替代 CHARMM-GUI 调用的 PPM 网页服务。
    import numpy as np
    import gpcr_defs as G
    keys, seq, ca = _read_pdb_ca(src_pdb)
    if len(keys) < 60:
        return None, "读到 %d 个残基，不足以判断跨膜螺旋" % len(keys)
    helices, extras, _prof = G.predict_tm_helices(seq)
    if len(helices) < 5:
        return None, "只预测到 %d 条跨膜段，不足 5 条，不敢据此定向" % len(helices)
    axes = []
    for a, b in helices[:7]:
        idxs = list(range(a, min(b + 1, len(keys))))
        if len(idxs) < 6:
            continue
        pts = np.array([ca[keys[i]] for i in idxs])
        pts = pts - pts.mean(axis=0)
        _u, _s, vt = np.linalg.svd(pts, full_matrices=False)
        v = vt[0]
        if v[2] < 0:
            v = -v
        axes.append(v)
    if len(axes) < 5:
        return None, "有效螺旋轴只有 %d 条" % len(axes)
    A = np.array(axes)
    A = A - A.mean(axis=0)
    _u, _s, vt = np.linalg.svd(A, full_matrices=False)
    R = _rot_axis_to_z(vt[0])
    tilt = float(np.degrees(np.arccos(min(1.0, abs(float(vt[0][2]))))))
    return R, "跨膜束主轴与 z 轴夹角 %.1f°，已对齐；用了 %d 条螺旋" % (tilt, len(axes))


@tool
def prep_orient_protein(protein_pdb="") -> str:
    """【准备·膜·定向】用跨膜螺旋束主轴把蛋白摆正（轴对齐 z），供插膜使用。

    替代 CHARMM-GUI 调用的 PPM 网页服务：只依赖序列疏水预测 + 螺旋轴 PCA，
    不需要外部服务。非膜蛋白会被识别出来（预测不到 5 条以上跨膜段就拒绝）。

    Args:
        protein_pdb: 蛋白结构；留空则读 prep_config.json 里的 protein_pdb
    """
    cfg = _load_json(PREP_CFG)
    if not cfg:
        return "【执行报错】尚未配置：请先调用 prep_set_config。"
    src = _abs(protein_pdb or cfg.get("protein_pdb") or "")
    if not src or not os.path.exists(src):
        return "【执行报错】未找到蛋白结构 .pdb。"
    R, msg = _orient_protein(src, os.path.join(_ws(), "protein_oriented.pdb"))
    if R is None:
        _log("orient_protein", False, msg)
        return "【执行报错】定向失败：%s" % msg
    import numpy as np
    coords = []
    for ln in open(src, encoding="utf-8", errors="replace"):
        if ln.startswith(("ATOM", "HETATM")):
            coords.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    arr = np.array(coords)
    # 先把质心移到原点再旋转：否则蛋白还停在原始坐标处，
    # 后面按原点为中心铺的脂质网格就跟它错开了（实测差了 187 A）。
    ctr = arr.mean(axis=0)
    out = os.path.join(_ws(), "protein_oriented.pdb")
    n = 0
    with open(src, encoding="utf-8", errors="replace") as fi, \
            open(out, "w", encoding="utf-8") as fo:
        for ln in fi:
            if ln.startswith(("ATOM", "HETATM")):
                p3 = R.dot(np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]) - ctr)
                ln = ln[:30] + "%8.3f%8.3f%8.3f" % (p3[0], p3[1], p3[2]) + ln[54:]
                n += 1
            fo.write(ln)
    _log("orient_protein", True, msg)
    return "【蛋白已定向】protein_oriented.pdb（%d 原子）\n%s" % (n, msg)


@tool
def prep_build_membrane(n_per_leaflet=0, seed=12345, clash_cutoff=1.8) -> str:
    """【准备·膜·建膜】用晶格摆放 POPC 双层，并剔除与蛋白冲突的脂质。

    产出 membrane.pdb（蛋白 + 脂质）。**这是待平衡的初始结构，不是可用的膜体系**：
    本流程只做几何摆放，不做 CHARMM-GUI 那样的多轮平衡、取向优化与脂质对称化。
    这两个事实会写进审查报告，不会藏在日志里。

    这里的做法参照 gromacs-electrolyte-workflow 的门禁做法：
    固定随机种子、明确分子数、记录实际盒子和最小距离。

    Args:
        n_per_leaflet: 每层脂质数；0 表示按蛋白横截面自动估算
        seed: 晶格摆位随机种子（固定以保证可复现）
        clash_cutoff: 脂质-蛋白最小距离阈值（Å），低于它视为冲突并剔除该脂质
    """
    cfg = _load_json(PREP_CFG)
    if not cfg:
        return "【执行报错】尚未配置：请先调用 prep_set_config。"
    if not cfg.get("membrane"):
        return "【执行报错】配置里 membrane=False，请先开启建膜。"
    prot = os.path.join(_ws(), "protein_oriented.pdb")
    if not os.path.exists(prot):
        return "【执行报错】缺少 protein_oriented.pdb，请先跑 prep_orient_protein。"


    popc = os.path.join(_ws(), "popc_single.pdb")
    got, note = _popc_pdb(popc)
    if not got:
        _log("build_membrane", False, note)
        return "【执行报错】取 POPC 构象失败：%s" % note

    # 盒子 XY 由蛋白横截面决定；z 固定为双层 + 水层。
    xs, ys, zs = [], [], []
    for ln in open(prot, encoding="utf-8", errors="replace"):
        if ln.startswith(("ATOM", "HETATM")):
            xs.append(float(ln[30:38])); ys.append(float(ln[38:46])); zs.append(float(ln[46:54]))
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    box_xy = max(span_x, span_y) + 2 * 12.0          # 四周各留 12 A
    if not n_per_leaflet:
        area = box_xy * box_xy
        n_per_leaflet = int(area / 64.0)             # 约 64 A^2 每个脂质
    atoms = _build_bilayer_grid(popc, n_per_leaflet, box_xy, seed)
    n_at = sum(1 for l in open(popc, encoding="utf-8", errors="replace")
               if l.startswith("ATOM"))
    residues = [atoms[i:i + n_at] for i in range(0, len(atoms), n_at)]

    import numpy as np
    P = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                  for l in open(prot, encoding="utf-8", errors="replace")
                  if l.startswith(("ATOM", "HETATM"))], dtype=float)
    kept, dropped = [], 0
    for res in residues:
        R = np.array([[c[1], c[2], c[3]] for c in res], dtype=float)
        dmin = 1e9
        for i in range(0, len(P), 2000):
            blk = P[i:i + 2000]
            d = np.sqrt(((R[:, None, :] - blk[None, :, :]) ** 2).sum(axis=2)).min()
            if d < dmin:
                dmin = float(d)
            if dmin < clash_cutoff:
                break
        if dmin < clash_cutoff:
            dropped += 1
        else:
            kept.append((res, dmin))


    # 脂质之间也要去重叠：晶格 + 抖动会让相邻脂质的头基撞在一起。
    # 实测不去重叠时 mdrun 直接报 Maximum force = inf（原子重叠），结构不可用。
    _cents = []
    _kept2 = []
    for res, dmin in kept:
        c = np.array([[x[1], x[2], x[3]] for x in res], dtype=float).mean(axis=0)
        if any(float(np.linalg.norm(c - o)) < 7.0 for o in _cents):
            dropped += 1
            continue
        _cents.append(c)
        _kept2.append((res, dmin))
    kept = _kept2

    # 写 membrane.pdb（蛋白在前、脂质在后，与 topol 的 [ molecules ] 顺序一致）
    with open(os.path.join(_ws(), "membrane.pdb"), "w", encoding="utf-8") as fo:
        for l in open(prot, encoding="utf-8", errors="replace"):
            if l.startswith(("ATOM", "HETATM")):
                fo.write(l)
        fo.write("TER\n")
        idx = 0
        for res, _d in kept:
            idx += 1
            for k, (nm, x, y, z) in enumerate(res):
                fo.write(_pdb_atom((idx - 1) * len(res) + k + 1, nm, "POPC", idx, x, y, z) + "\n")
        fo.write("END\n")
    # 另存一份"只有脂质"的坐标：组装拓扑时用的是 pdb2gmx 的输出（已加氢），
    # 原子集与这里的输入结构不同，所以不能直接用 membrane.pdb 当坐标；
    # 只能用它的脂质部分，按同一参考系拼到 protein.gro 后面。
    with open(os.path.join(_ws(), "lipids.pdb"), "w", encoding="utf-8") as fl:
        idx2 = 0
        for res, _d in kept:
            idx2 += 1
            for k, (nm, x, y, z) in enumerate(res):
                fl.write(_pdb_atom((idx2 - 1) * len(res) + k + 1, nm, "POPC", idx2, x, y, z) + "\n")
        fl.write("END\n")

    dmin_all = min([d for _r, d in kept], default=0.0)
    _log("build_membrane", True,
         "请求每层 %d，实际保留 %d，剔除 %d；盒子 %.1f A；最小脂质-蛋白距离 %.2f A"
         % (n_per_leaflet, len(kept), dropped, box_xy, dmin_all))
    _warn("膜为本地晶格几何摆放：未做多轮平衡、蛋白取向优化、脂质对称化。"
          "这是待平衡的初始结构，不是可用的膜体系。")
    return ("【膜已搭建】membrane.pdb\n"
            "每层请求 %d，实际保留 %d，与蛋白冲突剔除 %d\n"
            "盒子边长 %.1f Å，最小脂质-蛋白距离 %.2f Å\n"
            "**注意**：这只是几何摆放的初始结构，插膜后必须做平衡，"
            "本流程不含平衡。" % (n_per_leaflet, len(kept), dropped, box_xy, dmin_all))


PREP_TOOLS = [
    prep_set_config, prep_protein_topology, prep_ligand_cgenff,
    prep_ligand_gaff, prep_assemble, prep_build_box,
    prep_solvate, prep_ionize, prep_grompp, prep_review,
    prep_orient_protein, prep_build_membrane,
]
