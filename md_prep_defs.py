"""MD 体系准备领域层：力场 / 水模型 / 盒型 / 配体路线 / 脂质的注册表与兼容性检查。

与 gpcr_defs.py 同一套定位：纯标准库（只 import os），不依赖 numpy/MDAnalysis，
两个 conda env 都能安全加载。这里只做"查表 + 判断"，不执行任何 gmx 命令。

设计要点：本模块的核心不是列举选项，而是 **compat_check()**。
力场、水模型、配体参数路线三者之间存在真实的物理约束
（如 GAFF 配体参数属于 Amber 家族，配 CHARMM36 蛋白是不自洽的）。
选错不会报错，只会让后续模拟结果悄悄偏掉 —— 所以宁可多出告警，也不静默放行。
"""
import os

# ============================================================
# 力场注册表
#   key            : 用户/CLI 传的名字，同时也是 pdb2gmx -ff 的值
#   ff_dir         : .ff 目录名；None 表示用 GROMACS 自带（在 gmx 安装目录里）
#   waters         : 该力场官方推荐/可接受的水模型
#   default_water  : 默认水模型
#   ligand_routes  : 与该力场"家族一致"的配体参数路线
#   lipids         : 可用的脂质（膜体系用）；None 表示该力场不带脂质参数
# ============================================================
FORCEFIELDS = {
    "charmm36-jul2022": {
        "label": "CHARMM36 (July 2022)",
        "ff_dir": "charmm36-jul2022.ff",
        "family": "charmm",
        "waters": ["tip3p", "tip3p_original", "tip4p", "tip4pew", "tip5p", "spc", "spce"],
        "default_water": "tip3p",
        "ligand_routes": ["cgenff"],
        "lipids": ["POPC", "POPE", "DPPC", "POPG", "POPS", "CHL1"],
        "recommended": True,
        "note": "含蛋白 + 脂质 + CGenFF，膜体系与配体体系都用它",
    },
    "amber99sb-ildn": {
        "label": "Amber99SB-ILDN",
        "ff_dir": None,
        "family": "amber",
        "waters": ["tip3p", "tip4p", "tip4pew", "spc", "spce"],
        "default_water": "tip3p",
        "ligand_routes": ["gaff"],
        "lipids": None,
        "recommended": False,
        "note": "可溶蛋白常用；配体需 GAFF（acpype）",
    },
    "amber14sb": {
        "label": "Amber14SB",
        "ff_dir": None,
        "family": "amber",
        "waters": ["tip3p", "tip4p", "tip4pew", "spc", "spce"],
        "default_water": "tip3p",
        "ligand_routes": ["gaff"],
        "lipids": None,
        "recommended": False,
        "note": "可溶蛋白常用；配体需 GAFF（acpype）",
    },
    "amber19sb": {
        "label": "Amber19SB",
        "ff_dir": None,
        "family": "amber",
        "waters": ["tip3p", "tip4pew", "spc", "spce"],
        "default_water": "tip4pew",
        "ligand_routes": ["gaff"],
        "lipids": None,
        "recommended": False,
        "note": "较新；官方建议配 OPC/TIP4PEW 水",
    },
    "oplsaa": {
        "label": "OPLS-AA",
        "ff_dir": None,
        "family": "opls",
        "waters": ["tip3p", "tip4p", "spc", "spce"],
        "default_water": "tip3p",
        "ligand_routes": [],
        "lipids": None,
        "recommended": False,
        "note": "配体参数需 OPLS 一致的路线（本机未装 LigParGen），配体体系不推荐",
    },
    "gromos54a7": {
        "label": "GROMOS-54A7",
        "ff_dir": None,
        "family": "gromos",
        "waters": ["spc", "spce", "tip3p", "tip4p"],
        "default_water": "spc",
        "ligand_routes": [],
        "lipids": None,
        "recommended": False,
        "note": "传统上与 SPC/SPC-E 搭配；配体参数路线本机未装",
    },
    "charmm27": {
        "label": "CHARMM27（旧版，仅为兼容保留）",
        "ff_dir": None,
        "family": "charmm",
        "waters": ["tip3p", "tip3p_original"],
        "default_water": "tip3p",
        "ligand_routes": ["cgenff"],
        "lipids": None,
        "recommended": False,
        "note": "已被 CHARMM36 取代，除非复现旧工作否则不要选",
    },
}

# 水模型：families 记录它被哪些力场家族接受（用于兼容性检查）
WATER_MODELS = {
    "tip3p": {"label": "TIP3P", "families": ["charmm", "amber", "opls", "gromos"]},
    "tip4p": {"label": "TIP4P", "families": ["charmm", "amber", "opls"]},
    "tip4pew": {"label": "TIP4P/EW", "families": ["charmm", "amber"]},
    "tip5p": {"label": "TIP5P", "families": ["charmm", "amber"]},
    "spc": {"label": "SPC", "families": ["charmm", "amber", "opls", "gromos"]},
    "spce": {"label": "SPC/E", "families": ["charmm", "amber", "opls", "gromos"]},
}


# 盒子形状
BOX_SHAPES = {
    "dodecahedron": {"label": "十二面体", "note": "同体积下最省水，常规首选"},
    "cubic": {"label": "立方体", "note": "最简单，但水分子数最多"},
    "triclinic": {"label": "三斜", "note": "适合晶体/各向异性体系"},
}

# 配体参数路线
LIGAND_ROUTES = {
    "cgenff": {
        "label": "CGenFF",
        "family": "charmm",
        "local_tools": ["cgenff_charmm2gmx_py3.py"],
        "automatic": False,
        "note": "与 CHARMM36 配套；赋参需在 app.cgenff.com 上传取 .str（半自动）",
    },
    "gaff": {
        "label": "GAFF (acpype)",
        "family": "amber",
        "local_tools": ["acpype", "antechamber"],
        "automatic": True,
        "note": "全本地免账号；属 Amber 家族，配 CHARMM36 蛋白时力场不自洽",
    },
}



DEFAULT_CONFIG = {
    "forcefield": "charmm36-jul2022",
    "water": "tip3p",
    "ligand_route": "cgenff",
    "box": "dodecahedron",
    "box_padding_nm": 1.0,
    "salt": "NaCl",
    "salt_conc_M": 0.15,
    "membrane": False,
    "lipid": "POPC",
    "protein_pdb": "",
    "ligand_file": "",
}


def get_forcefield(key):
    return FORCEFIELDS.get(str(key or "").strip())


def get_water(key):
    return WATER_MODELS.get(str(key or "").strip())


def get_ligand_route(key):
    return LIGAND_ROUTES.get(str(key or "").strip())


def get_box(key):
    return BOX_SHAPES.get(str(key or "").strip())


_ENUM_TABLES = {"forcefield": FORCEFIELDS, "water": WATER_MODELS, "ligand_route": LIGAND_ROUTES, "box": BOX_SHAPES}


def resolve_config(cfg=None):
    """把用户配置补全成完整配置：缺项填默认，非法值回退默认并记录。"""
    out = dict(DEFAULT_CONFIG)
    notes = []
    for k, v in (cfg or {}).items():
        if k not in DEFAULT_CONFIG:
            continue
        if k in ("box_padding_nm", "salt_conc_M"):
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                notes.append("参数 %s 值 %r 无法解析，用默认 %s" % (k, v, DEFAULT_CONFIG[k]))
            continue
        if k == "membrane":
            out[k] = bool(v)
            continue
        if not v:
            continue
        val = str(v).strip()
        table = _ENUM_TABLES.get(k)
        if table is not None and val not in table:
            notes.append('参数 %s 的值 %r 不在可选范围内，已回退默认 %s' % (k, v, DEFAULT_CONFIG[k]))
            continue
        out[k] = val
    return out, notes



def compat_check(cfg):
    """检查力场/水模型/配体路线/膜是否互相自洽，返回告警字符串列表。

    这是本模块最有价值的部分：选错组合不会报错，只会让模拟结果悄悄偏掉，
    所以这里宁可多出告警，也不静默放行。
    """
    out = []
    ff_key = cfg.get("forcefield")
    ff = get_forcefield(ff_key)
    if not ff:
        return ["力场 %r 不在注册表中，将无法解析 .ff 路径" % ff_key]

    water_key = cfg.get("water")
    w = get_water(water_key)
    if not w:
        out.append("水模型 %r 不在注册表中" % water_key)
    else:
        if water_key not in ff.get("waters", []):
            out.append(
                "水模型 %s 不在 %s 的推荐列表内（推荐：%s）"
                % (w["label"], ff["label"], ", ".join(ff.get("waters", [])))
            )
        elif w.get("families") and ff.get("family") not in w["families"]:
            out.append(
                "水模型 %s 通常不与该力场家族（%s）搭配" % (w["label"], ff.get("family"))
            )

    route_key = cfg.get("ligand_route")
    route = get_ligand_route(route_key)
    if not route:
        out.append("配体参数路线 %r 不在注册表中" % route_key)
    else:
        if route_key not in ff.get("ligand_routes", []):
            out.append(
                "【力场不自洽】配体路线 %s 属 %s 家族，而蛋白力场 %s 属 %s 家族。"
                "混用力场家族需要在报告中显式说明，结论解释时要谨慎。"
                % (route["label"], route.get("family"), ff["label"], ff.get("family"))
            )
        if not route.get("automatic"):
            out.append(
                "配体路线 %s 不是全自动：需要在 app.cgenff.com 上传配体、"
                "取回 .str 后放回工作区，流程会在此等待。" % route["label"]
            )

    if cfg.get("membrane"):
        lipids = ff.get("lipids")
        if not lipids:
            out.append("力场 %s 不带脂质参数，无法建膜" % ff["label"])
        else:
            lip = cfg.get("lipid")
            if lip and lip.split(":")[0] not in lipids:
                out.append("脂质 %s 不在该力场可用的脂质列表内（%s）" % (lip, ", ".join(lipids)))
        out.append(
            "本地膜插入的平衡程度远低于 CHARMM-GUI Membrane Builder："
            "本流程只做几何摆放与最小化，不含多轮平衡、取向优化与脂质对称化。"
            "膜未平衡是硬结论，不得当作已平衡体系使用。"
        )

    pad = cfg.get("box_padding_nm")
    try:
        if float(pad) < 1.0:
            out.append("盒子边距 %.2f nm 偏小（建议 >= 1.0），可能违反最小镜像约定" % float(pad))
    except (TypeError, ValueError):
        pass

    return out


def ff_search_path(forcefields_root):
    """返回应该设给 GMXLIB 的目录（.ff 所在目录）。

    charmm36-jul2022 是我们自己装的，必须靠 GMXLIB 才能被 pdb2gmx 找到；
    GROMACS 自带力场则不需要。返回 None 表示无需设置 GMXLIB。
    """
    if not forcefields_root:
        return None
    return os.path.abspath(forcefields_root)


def list_options():
    """给 UI 用的选项列表：[(显示名, 值), ...]。"""
    def _f(d):
        return [(("%s — %s" % (v["label"], v.get("note", "")) if v.get("note") else v["label"]), k)
                for k, v in d.items()]
    return {
        "forcefield": _f(FORCEFIELDS),
        "water": _f(WATER_MODELS),
        "ligand_route": _f(LIGAND_ROUTES),
        "box": _f(BOX_SHAPES),
    }


def describe(cfg):
    """给审查报告用的一行式配置摘要。"""
    ff = get_forcefield(cfg.get("forcefield")) or {}
    w = get_water(cfg.get("water")) or {}
    r = get_ligand_route(cfg.get("ligand_route")) or {}
    b = get_box(cfg.get("box")) or {}
    membrane = "是（%s）" % cfg.get("lipid") if cfg.get("membrane") else "否"
    return "\n".join([
        "• 蛋白力场：%s" % ff.get("label", cfg.get("forcefield")),
        "• 水模型：%s" % w.get("label", cfg.get("water")),
        "• 配体路线：%s" % r.get("label", cfg.get("ligand_route")),
        "• 盒子：%s，边距 %s nm" % (b.get("label", cfg.get("box")), cfg.get("box_padding_nm")),
        "• 盐：%s %s M" % (cfg.get("salt"), cfg.get("salt_conc_M")),
        "• 建膜：%s" % membrane,
    ])


# 残基命名差异：同一残基在不同力场家族里叫不同名字。
# 实测踩到：CHARMM-GUI 产出的结构用 CHARMM 命名（HSD/HSE/HSP），
# 喂给 Amber 会直接 Fatal error: Residue 'HSD' not found in residue
# topology database —— 报错完全看不出原因。
# 服务器上核对过的实际名字：
#   CHARMM36: HIP HSD HSE HSP   （无 CYX，二硫键靠 specbond 自动识别）
#   Amber   : HID HIE HIP       （CYX = 二硫键半胱氨酸）
CHARMM_TO_AMBER_RESNAME = {"HSD": "HID", "HSE": "HIE", "HSP": "HIP"}
AMBER_TO_CHARMM_RESNAME = {"HID": "HSD", "HIE": "HSE", "HIP": "HIP", "CYX": "CYS"}

def rename_map_for(family, resnames):
    table = AMBER_TO_CHARMM_RESNAME if family == "charmm" else CHARMM_TO_AMBER_RESNAME
    seen = set(str(r).strip().upper() for r in resnames)
    return dict((r, table[r]) for r in sorted(seen) if r in table)

def scan_resnames(path):
    out, seen = [], set()
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return out
    with fh:
        for line in fh:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                r = line[17:20].strip().upper()
                if r and r not in seen:
                    seen.add(r)
                    out.append(r)
    return out
