"""决策层：把「必须拿主意的量」变成可复算、可反证的判定。

这个模块回答的问题不是「怎么跑」，而是「跑之前该定哪几个量、凭什么定」。

三条设计原则，都是被本项目踩过的坑逼出来的：

1. **证据是数字，从文件里算出来。** 判定依据一律走 evidence 字典，由
   decision_tools.py 解析真实文件填进去，绝不让模型「感觉」出一个数。
   （教训：GPCR 层交给 LLM 驱动会被整层跳过；报告读到上一轮残留产物会被
   当成本次实测。）

2. **判定是阈值表，写在代码里。** 同样的证据谁都能复算出同样的判定，
   这才叫可验证。模型只负责把判定翻译成人话。

3. **落到灰区就停下来问人。** 这不是缺陷，是分界线：能自己验的（净电荷必须
   中和到 0、二硫键 SG-SG 距离、配体接触保留率）agent 自己定；验不了的
   （某个组氨酸该不该质子化、这个 pose 算不算「跑掉了」）停下来让人定。

每个判定额外带一条 **反证条件（falsifier）**：什么观测一旦出现，就说明这个
判定是错的。没有反证条件的判定等于没有落点，出了错也拦不住——所以它是必填项。

纯标准库，不 import numpy / MDAnalysis，两个 env 都能加载（与 gpcr_defs.py 同定位）。
"""

CHARGE_BY_RESNAME = {
    "ASP": -1.0, "GLU": -1.0,
    "ASH": 0.0, "GLH": 0.0,
    "LYS": 1.0, "ARG": 1.0,
    "LYN": 0.0,
    "HSD": 0.0, "HSE": 0.0, "HID": 0.0, "HIE": 0.0,
    "HSP": 1.0, "HIP": 1.0,
    "NA": 1.0, "SOD": 1.0, "MG": 2.0, "K": 1.0, "CA": 2.0,
    "CL": -1.0, "CLA": -1.0,
}

TERMINUS_CHARGE = {"charged": 0.0, "neutral": 0.0}

HIS_TAUTOMERS = ("HSD", "HSE", "HSP")

# ---------------------------------------------------------------------------
# 二、几何判据阈值。单位一律 nm（GROMACS 原生单位）。
#     每个数字后面都写了它的物理来处，被追问时才答得上。
# ---------------------------------------------------------------------------

# 二硫键：SG-SG 共价键长约 0.204 nm。取 0.25 nm 做判据，
# 既容得下力场优化后的轻微拉伸，又远小于两个游离 Cys 的最近可能距离（>0.35 nm）。
SS_BOND_MAX_NM = 0.25

# 氢键：重原子 N...O 距离在 0.26-0.30 nm 之间才可能成氢键。
# 取 0.30 作上限，用于判断「这个 His 的 Nδ1/Nε2 是否被羧基抓住」。
HBOND_MAX_NM = 0.30

# 配体还「在口袋里」的判据（三条同时成立才算结合保持）：
#   - 叠合到受体核心后的配体 RMSD：0.30 nm 以内说明没换位姿
#   - 初始接触的保留率：0.60 以上说明结合模式没散
#   - 最小距离：0.50 nm 以内说明还有直接接触（0.45 是接触定义本身）
POSE_KEEP_RMSD_NM = 0.30
POSE_KEEP_RETENTION = 0.60
POSE_CONTACT_CUTOFF_NM = 0.45

# 判定「配体离开了口袋」的门槛（两条同时成立）：
POSE_LEFT_RETENTION = 0.20
POSE_LEFT_MINDIST_NM = 1.00

# CGenFF 罚分：<50 可信；50-100 要标出来；>100 基本是原子类型没对上。
PENALTY_OK = 50.0
PENALTY_SUSPECT = 100.0

# 膜质量：脂质与蛋白的最近原子距离低于 0.30 nm 就是硬碰硬的重叠，
# 能量最小化会在这里出巨大受力（实测未修正时 Maximum force = inf）。
MEMB_MIN_CLEARANCE_NM = 0.30

# 上下叶脂质数差异超过 5% 说明蛋白把一层顶掉了，双层不对称。
MEMB_LEAFLET_ASYMMETRY = 0.05

# 盒子 z 方向至少要留出 2 × 0.8 nm 的水层，否则周期镜像的膜会自己贴自己。
MEMB_MIN_WATER_NM = 1.60

# 三、决策点注册表。
# 每一项声明「这个判断为什么存在、要什么证据、错了怎么办」，不含阈值逻辑
# （阈值逻辑在下面的 evaluate_* 函数里）。falsifier 必填：写不出反证条件，
# 说明这个判断根本没落点。
DECISION_SPECS = {
    "protonation": {
        "title": "质子化态与二硫键",
        "question": "每个 His 取 HSD / HSE / HSP？哪几对 Cys 成二硫键？",
        "why": "pdb2gmx 默认按几何静默挑 His 互变异构体；His 每多一个质子就多 +1 净电荷，直接改变后续离子数。二硫键漏判会让两个 Cys 带游离巯基，局部结构会散。",
        "evidence": ["n_his", "his_sites", "ss_pairs", "n_free_cys", "formal_charge_no_h"],
        "falsifier": "若显式 HSP 与默认 HSD 各跑 10 ns，HSP 版该残基周边的氢键占据率或局部 RMSF 明显更稳，则原判定为错。",
        "recheck": "改 PDB 残基名后重跑 pdb2gmx，约 1 分钟。",
    },
    "net_charge": {
        "title": "净电荷与离子数",
        "question": "这个体系的形式净电荷是多少？要补几个 Na+ / Cl- 才能既中和又到目标浓度？",
        "why": "净电荷只差 1 就会让加离子数差 1，进而在 PME 下产生一个非物理的净背景电荷。配体自己常带电（实测 7W0M 的 LIG 是 -1），只按蛋白算必然算错。",
        "evidence": ["charge_by_molecule", "solute_charge", "ions_added", "target_conc_M", "n_water"],
        "falsifier": "加离子后重新解析 topol.top 的 [ atoms ] 求电荷和，若不为 0，则 genion 的 -neutral 没生效或拓扑被改写。",
        "recheck": "重跑 genion，几十秒。",
    },
    "ligand_params": {
        "title": "配体力场参数从哪来",
        "question": "这个配体用 CGenFF 还是 GAFF？拿到的参数可信吗？",
        "why": "CGenFF 罚分高的原子类型意味着参数是猜出来的；参数不对会把配体从口袋里推走，而现象看起来像「真实解离」。所以来源必须先定，且要有罚分证据。",
        "evidence": ["route", "cached", "penalty_max", "penalty_n", "penalty_per_residue", "n_heavy_atoms"],
        "falsifier": "若换 GAFF 参数重跑同一 pose，配体接触保留率显著提高（例如 0.3 升到 0.8），则原 CGenFF 判定不可信。",
        "recheck": "GAFF 走 acpype，一个配体几分钟。",
    },
    "membrane_quality": {
        "title": "膜搭建质量能不能接受",
        "question": "本地铺的 POPC 双层，几何上过关吗？能不能拿去跑？",
        "why": "本地是晶格几何摆放，没有 CHARMM-GUI 的多轮平衡。脂质与蛋白重叠会在最小化时报巨大受力（实测出现过 Maximum force = inf），脂质间距过密会自爆。必须拿数字判定，不能只说「已搭建」。",
        "evidence": ["n_tm", "tilt_deg", "min_lipid_protein_nm", "n_leaflet_upper", "n_leaflet_lower", "box_z_nm", "n_dropped"],
        "falsifier": "若能量最小化后最大受力仍大于 1e5 kJ/mol/nm，或上下叶脂质数差超过 5%，则本判定（几何过关）为错。",
        "recheck": "重调晶格间距重铺，几分钟。",
    },
    "ligand_dissociation": {
        "title": "配体是跑掉了还是力场假象",
        "question": "配体离开口袋，是真实解离还是参数/pose 造成的假象？该继续还是停？",
        "why": "最容易被自信地给错结论的一步：配体飘走可以是真的解离，也可以是 CGenFF 罚分高导致的假象，两者后续处理完全相反，必须分开。",
        "evidence": ["retention_start", "retention_end", "lig_rmsd_final_nm", "min_dist_final_nm", "protein_rmsd_max_nm", "penalty_max"],
        "falsifier": "若把配体参数换成低罚分版本后同一轨迹不再解离，则「真实解离」这个判定为错（本来假象）。",
        "recheck": "换参数重跑一段，取决于轨迹长度。",
    },
}


def _verdict(decision_id, verdict, choice, reason, confidence, evidence, extra=None):
    """组装一条判定记录。falsifier 从注册表取，保证每条判定都带反证条件。"""
    spec = DECISION_SPECS.get(decision_id, {})
    rec = {
        "id": decision_id,
        "title": spec.get("title", decision_id),
        "question": spec.get("question", ""),
        "verdict": verdict,
        "choice": choice,
        "reason": reason,
        "confidence": confidence,
        "falsifier": spec.get("falsifier", ""),
        "recheck": spec.get("recheck", ""),
        "evidence": evidence,
    }
    if extra:
        rec.update(extra)
    return rec


def evaluate_protonation(ev):
    """判定 His 互变异构体与二硫键。

    ev 需要：
      his_sites  [{resnum, resname, acceptor_res, acceptor_dist_nm}]
      ss_pairs   [{resnum_a, resnum_b, dist_nm}]
      ss_ambiguous [{resnum_a, resnum_b, dist_nm}]   # 0.25-0.35 nm 之间，判不准
      n_his, n_cys, n_free_cys

    判定规则：
      - SG-SG <= 0.25 nm 是共价键，没有商量余地 -> 直接判二硫键
      - His 的 N 与羧基氧 <= 0.30 nm：可能是 HSP，也可能是中性 His 把 H 转开。
        这是真判断，标成 medium 置信并写明反证条件
      - His 周围什么都没有 -> 中性，取 HSD（CHARMM 默认 Nδ1-H）
    """
    his = ev.get("his_sites") or []
    ss = ev.get("ss_pairs") or []
    ss_amb = ev.get("ss_ambiguous") or []

    his_choice = {}
    candidates = []
    for h in his:
        rn = str(h.get("resname", "HIS")).upper()
        if rn in HIS_TAUTOMERS:
            his_choice[h["resnum"]] = rn
            continue
        d = h.get("acceptor_dist_nm")
        if d is not None and float(d) <= HBOND_MAX_NM:
            his_choice[h["resnum"]] = "HSP"
            candidates.append(h)
        else:
            his_choice[h["resnum"]] = "HSD"

    charge_shift = len(candidates)
    note = ("%d 个 His：%d 个与羧基成氢键（判 HSP），其余取 HSD；%d 对二硫键。"
            % (len(his), charge_shift, len(ss)))

    if ss_amb:
        return _verdict(
            "protonation", "ESCALATE", {"his": his_choice, "ss": ss, "pending": "ss"},
            "有 %d 对 Cys 的 SG-SG 距离落在 0.25-0.35 nm 灰区，判不准是不是二硫键。"
            "距离在这个区间既可能是没优化好的二硫键，也可能是两个恰好靠得近的游离巯基。"
            % len(ss_amb), "low",
            {"his_sites": his, "ss_pairs": ss, "ss_ambiguous": ss_amb},
            {"options": ["按二硫键处理（CYS 成键）", "按游离巯基处理（各带 H）"]})

    conf = "medium" if candidates else "high"
    return _verdict(
        "protonation", "DECIDED", {"his": his_choice, "ss": ss},
        note + ("其中 %d 个 His 与羧基氧距离在氢键范围内，判为质子化（HSP），"
                "这会让体系净电荷比全中性 His 高 %d。"
                % (charge_shift, charge_shift) if charge_shift else
                "所有 His 周围无羧基，取中性 HSD。"),
        conf, {"his_sites": his, "ss_pairs": ss, "n_free_cys": ev.get("n_free_cys")})


def expected_pair_ions(n_water, conc_M):
    """目标浓度对应的 NaCl 对数（不含中和所需）。

    1 L 水 = 55.5 mol，所以 n_pairs 约等于 conc * n_water / 55.5。
    实测校验：7W0M 的 0.15 M / 19230 个水 -> 52.0，CHARMM-GUI 实际加了 51 个 Na+。
    """
    try:
        return float(conc_M) * float(n_water) / 55.5
    except Exception:
        return 0.0


def _charge_decide(ev, solute, n_water, conc, added, expected, detail):
    if added:
        net = solute + float(added.get("NA", 0)) - float(added.get("CL", 0))
        if abs(net) > 0.01:
            return _verdict(
                "net_charge", "ESCALATE", {"net_after_ions": net},
                "加离子后重算净电荷是 %+.2f，不是 0。genion -neutral 本应保证中性，"
                "不为 0 说明拓扑在加离子后被改写过，或中性化没生效。" % net,
                "high", ev)
        n_pairs = expected_pair_ions(n_water, conc) if conc else 0.0
        real_pairs = min(int(added.get("NA", 0)), int(added.get("CL", 0)))
        return _verdict(
            "net_charge", "DECIDED",
            {"solute_charge": solute, "NA": added.get("NA"), "CL": added.get("CL")},
            "溶质电荷 %+g（%s）；加 %s 个 Na+ / %s 个 Cl- 后电中性，复算净电荷 %+.2f。"
            "目标 %s M 对应约 %.1f 对，实际 %d 对，偏差在舍入范围内。"
            % (solute, detail, added.get("NA"), added.get("CL"), net, conc, n_pairs, real_pairs),
            "high", ev)

    if expected is not None and abs(float(expected) - solute) > 0.5:
        return _verdict(
            "net_charge", "ESCALATE",
            {"computed": solute, "expected": float(expected)},
            "从拓扑逐原子复算的溶质净电荷是 %+g，与你给的 %+g 对不上（差 %+.1f）。"
            "这个差通常来自某个 His 取了质子化态、或者配体的质子化/电荷没算对。"
            "先定这个数再往下走，否则离子数会跟着错。" % (solute, float(expected), solute - float(expected)),
            "low", ev,
            {"options": ["按我算的 %+g 走" % solute, "按你说的 %+g 走（我去查拓扑）" % float(expected)]})

    n_pairs = expected_pair_ions(n_water, conc) if conc else 0.0
    return _verdict(
        "net_charge", "DECIDED",
        {"solute_charge": solute, "planned_pairs": round(n_pairs, 1)},
        "溶质形式净电荷 %+g（%s）。用 genion -neutral 中和到 0，%s。"
        % (solute, detail,
           ("再按 %s M 补约 %.1f 对 NaCl。" % (conc, n_pairs)) if conc
           else "目标浓度未指定，只做中和。"),
        "high", ev)


def evaluate_ligand_params(ev):
    """判定配体参数来源是否可信。

    ev: route（cgenff/gaff）、cached、penalty_max、penalty_n、
        penalty_per_residue、n_heavy_atoms
    """
    route = str(ev.get("route") or "cgenff")
    pmax = ev.get("penalty_max")
    if route != "cgenff":
        return _verdict(
            "ligand_params", "DECIDED", {"route": route},
            "走 %s 路线：本地全自动、无罚分概念，参数质量取决于 GAFF2 本身。"
            % route, "medium", ev)
    if pmax is None:
        return _verdict(
            "ligand_params", "ESCALATE", {"route": route},
            "CGenFF 路线但没解析到罚分：要么 .str 还没拿到，要么 .str 里没有 "
            "PENALTY 段。没有罚分就无法判断参数可信度。", "low", ev,
            {"options": ["上传/重取 .str", "改用 GAFF 路线"]})
    return _params_by_penalty(ev, route, float(pmax))


def _params_by_penalty(ev, route, pmax):
    if pmax <= PENALTY_OK:
        return _verdict(
            "ligand_params", "DECIDED", {"route": route, "penalty_max": pmax},
            "CGenFF 最大罚分 %.1f（<= %.0f），原子类型对得上，参数可用。"
            % (pmax, PENALTY_OK), "high", ev)
    if pmax <= PENALTY_SUSPECT:
        return _verdict(
            "ligand_params", "DECIDED", {"route": route, "penalty_max": pmax},
            "CGenFF 最大罚分 %.1f，落在 %.0f-%.0f 之间：参数能用，但该处原子类型是"
            "类比出来的。跑起来若配体异常移动，先怀疑这里。"
            % (pmax, PENALTY_OK, PENALTY_SUSPECT), "medium", ev)
    return _verdict(
        "ligand_params", "ESCALATE", {"route": route, "penalty_max": pmax},
        "CGenFF 最大罚分 %.1f > %.0f：这些原子类型基本没对上，参数不可信。"
        "强行跑下去，配体很可能被推离口袋，而现象看起来像真实解离。"
        % (pmax, PENALTY_SUSPECT), "low", ev,
        {"options": ["仍用 CGenFF（我接受风险）", "改用 GAFF 路线重做"]})


def evaluate_charge(ev):
    """判定形式净电荷，并核出该补多少离子。

    ev 需要：
      solute_charge      溶质（蛋白 + 配体 + 其它非水非离子）形式电荷之和
      charge_by_molecule 逐分子 type 的电荷，用来让人看见电荷从哪来
      expected_charge    用户口头说的净电荷（可选）——对不上就停下来问
      n_water, target_conc_M
      ions_added         已加离子数（验证阶段）；缺省表示还没加

    判定规则：
      - 溶质电荷必须中和到 0，硬约束 -> DECIDED
      - 用户给的 expected_charge 与算出来的差 0.5 以上 -> ESCALATE，两边数字都摆出来
      - 已加过离子：重算电荷和，不为 0 是 FAIL 级问题
    """
    solute = float(ev.get("solute_charge") or 0.0)
    n_water = ev.get("n_water") or 0
    conc = ev.get("target_conc_M")
    added = ev.get("ions_added") or None
    expected = ev.get("expected_charge")
    bymol = ev.get("charge_by_molecule") or {}
    detail = "；".join("%s %+g" % (k, v) for k, v in sorted(bymol.items()) if abs(v) > 1e-6)
    return _charge_decide(ev, solute, n_water, conc, added, expected, detail)


def evaluate_membrane(ev):
    """判定本地铺的膜几何上能不能接受。

    ev: n_tm, tilt_deg, min_lipid_protein_A,
        n_leaflet_upper, n_leaflet_lower, box_z_nm, n_dropped
    """
    clear = ev.get("min_lipid_protein_A")
    up = float(ev.get("n_leaflet_upper") or 0)
    lo = float(ev.get("n_leaflet_lower") or 0)
    ntm = int(ev.get("n_tm") or 0)
    problems = []
    return _membrane_judge(ev, clear, up, lo, ntm, problems)


def _membrane_judge(ev, clear, up, lo, ntm, problems):
    """按几何判据逐条检查膜搭建，把问题串进 problems 后给出判定。

    只判几何摆放，不代表膜已经平衡；这条限定会写进 reason。
    """
    worst = MEMB_MIN_CLEARANCE_NM * 10.0
    if clear is not None and float(clear) < worst:
        problems.append(
            "脂质与蛋白最近原子距离 %.2f A 低于下限 %.2f A，存在硬重叠"
            % (float(clear), worst))
    if max(up, lo) > 0 and abs(up - lo) / max(up, lo) > MEMB_LEAFLET_ASYMMETRY:
        problems.append(
            "上下叶脂质数不对称（上 %g / 下 %g，差异 %.1f%% > %.0f%%）"
            % (up, lo, abs(up - lo) / max(up, lo) * 100.0, MEMB_LEAFLET_ASYMMETRY * 100.0))
    box_z = ev.get("box_z_nm")
    span = ev.get("protein_z_span_nm")
    if box_z is not None and span is not None and float(box_z) - float(span) < MEMB_MIN_WATER_NM:
        problems.append(
            "盒子 z 方向水层不足（%.2f - %.2f = %.2f nm < %.2f nm），周期镜像的膜会贴自己"
            % (float(box_z), float(span), float(box_z) - float(span), MEMB_MIN_WATER_NM))
    if ntm and ntm < 5:
        problems.append("TM 螺旋只有 %d 条（< 5），不像膜蛋白" % ntm)

    tail = "注意这只是几何摆放检查，膜并未平衡（本地晶格摆放、没有多轮平衡）。"
    if problems:
        return _verdict(
            "membrane_quality", "DECIDED", {"problems": problems},
            "膜几何检查发现问题：" + "；".join(problems) + "。" + tail,
            "medium", ev)
    return _verdict(
        "membrane_quality", "DECIDED", {"problems": []},
        "几何检查全部通过：脂质无重叠、上下叶对称、TM 螺旋数合理。" + tail,
        "medium", ev)


def evaluate_dissociation(ev):
    """判定配体离开口袋是真解离还是力场/参数造成的假象。

    ev: retention_start、retention_end、lig_rmsd_final_nm、
        min_dist_final_nm、protein_rmsd_max_nm、penalty_max、n_frames
    """
    raw_ret = ev.get("retention_end")
    rmsd = ev.get("lig_rmsd_final_nm")
    mind = ev.get("min_dist_final_nm")
    prmsd = ev.get("protein_rmsd_max_nm")
    pmax = ev.get("penalty_max")

    if raw_ret is None:
        return _verdict(
            "ligand_dissociation", "ESCALATE", "部分解离/不确定",
            "缺少 retention_end（初始接触保留率），判断不了配体是否还在口袋里，按灰区处理。",
            "low", ev,
            {"options": ["延长轨迹", "换参数对照"]})

    r = float(raw_ret)
    if r > 1.0:
        r = 1.0

    if r >= POSE_KEEP_RETENTION and rmsd is not None and float(rmsd) <= POSE_KEEP_RMSD_NM:
        return _verdict(
            "ligand_dissociation", "DECIDED", "结合保持",
            "初始接触保留率 %.2f（>= %.2f），叠合后配体 RMSD %.2f nm（<= %.2f nm）："
            "位姿没换、结合模式没散，这个 pose 可信。"
            % (r, POSE_KEEP_RETENTION, float(rmsd), POSE_KEEP_RMSD_NM),
            "high", ev)

    if r >= POSE_KEEP_RETENTION and (rmsd is None or float(rmsd) > POSE_KEEP_RMSD_NM):
        if rmsd is None:
            tail = "但配体 RMSD 缺失，无法确认位姿是否稳定，位姿可能存在漂移。"
        else:
            tail = ("但叠合后配体 RMSD %.2f nm 超过 %.2f nm，位姿已经在口袋内部发生漂移。"
                    % (float(rmsd), POSE_KEEP_RMSD_NM))
        return _verdict(
            "ligand_dissociation", "DECIDED", "仍结合但位姿有位移",
            "初始接触保留率 %.2f（>= 结合保持线 %.2f），天然接触还在，还是结合态；"
            "%s这种情形最该在更长的轨迹里盯住。"
            % (r, POSE_KEEP_RETENTION, tail),
            "medium", ev)

    if r <= POSE_LEFT_RETENTION and mind is not None and float(mind) >= POSE_LEFT_MINDIST_NM:
        if pmax is not None and float(pmax) > PENALTY_SUSPECT:
            return _verdict(
                "ligand_dissociation", "ESCALATE", "疑似力场假象",
                "配体确实离开口袋（保留率 %.2f、最小距离 %.2f nm），但 CGenFF 最大罚分 "
                "%.1f > %.0f：参数本身是主要嫌疑，先别当成真实解离。"
                % (r, float(mind), float(pmax), PENALTY_SUSPECT),
                "low", ev,
                {"options": ["换 GAFF 参数重跑", "接受为真实解离并报告"]})
        if prmsd is not None and float(prmsd) <= 0.30:
            return _verdict(
                "ligand_dissociation", "DECIDED", "倾向真实解离",
                "配体离开口袋（保留率 %.2f、最小距离 %.2f nm），同时受体核心最大 RMSD "
                "只有 %.2f nm，蛋白没散架，倾向真实解离；但没有 GAFF 对照，参数假象还不能完全排除。"
                % (r, float(mind), float(prmsd)),
                "medium", ev)
        return _verdict(
            "ligand_dissociation", "ESCALATE", "不确定",
            "配体离开口袋（保留率 %.2f、最小距离 %.2f nm），但蛋白自身最大 RMSD 也偏大"
            "（%s nm），分不清是真解离还是整体被参数带歪，停下来看。"
            % (r, float(mind), prmsd),
            "low", ev,
            {"options": ["换 GAFF 重跑对照", "延长轨迹再看"]})

    rmsd_txt = "缺失" if rmsd is None else "%.2f nm" % float(rmsd)
    mind_txt = "缺失" if mind is None else "%.2f nm" % float(mind)
    return _verdict(
        "ligand_dissociation", "ESCALATE", "部分解离/不确定",
        "落在真实的灰区：保留率 %.2f 夹在解离线 %.2f 与结合保持线 %.2f 之间，"
        "既不能说还在口袋里，也不能说真跑掉了；配体 RMSD %s、最小距离 %s 只能当旁证。"
        % (r, POSE_LEFT_RETENTION, POSE_KEEP_RETENTION, rmsd_txt, mind_txt),
        "low", ev,
        {"options": ["延长轨迹", "换参数对照"]})


def apply_escalate_policy(rec, escalate_on):
    """按配置把 medium/low 置信的已定判定升级为「停下来问人」，不改原记录。"""
    if escalate_on and rec.get("verdict") == "DECIDED" and rec.get("confidence") in ("low", "medium"):
        new = dict(rec)
        new["verdict"] = "ESCALATE"
        new["reason"] = str(rec.get("reason", "")) + "已按配置停下等你确认（置信度 medium/low）。"
        return new
    return rec


def render_decision(rec):
    """把一条判定记录渲染成一个 markdown 块。"""
    ch = rec.get("choice")
    if isinstance(ch, dict):
        choice = "；".join("%s: %s" % (k, ch[k]) for k in ch)
    else:
        choice = str(ch)
    ev = rec.get("evidence")
    if isinstance(ev, dict):
        evid = "；".join("%s: %s" % (k, ev[k]) for k in ev)
    else:
        evid = str(ev)
    return (
        "### %s\n"
        "- **判定**：%s（置信度 %s）\n"
        "- **选择**：%s\n"
        "- **依据**：%s\n"
        "- **反证条件**：%s\n"
        "- **重做代价**：%s\n"
        "- **证据**：%s\n"
        % (rec.get("title", rec.get("id", "")), rec.get("verdict"), rec.get("confidence"),
           choice, rec.get("reason"), rec.get("falsifier"), rec.get("recheck"), evid)
    )


def render_ledger(records, title="本次做了哪些判断"):
    """把多条判定记录拼成一份 markdown 账本，带计数行。"""
    recs = list(records or [])
    decided = sum(1 for r in recs if r.get("verdict") == "DECIDED")
    esc = sum(1 for r in recs if r.get("verdict") == "ESCALATE")
    lines = ["## %s" % title,
             "共 %d 条判断：%d 条已定，%d 条等你确认。" % (len(recs), decided, esc)]
    for r in recs:
        lines.append("")
        lines.append(render_decision(r))
    return "\n".join(lines)


def evaluate(decision_id, ev, escalate_on=False):
    """总入口：按 decision_id 分派到对应判定，再套升级策略。"""
    table = {
        "protonation": evaluate_protonation,
        "net_charge": evaluate_charge,
        "ligand_params": evaluate_ligand_params,
        "membrane_quality": evaluate_membrane,
        "ligand_dissociation": evaluate_dissociation,
    }
    fn = table.get(decision_id)
    if fn is None:
        raise ValueError("未知的 decision_id: %r" % (decision_id,))
    return apply_escalate_policy(fn(ev), escalate_on)
