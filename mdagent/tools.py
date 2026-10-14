"""MD Agent 领域工具层：GROMACS 模拟 + MD 轨迹后处理分析。

工具全部为 LangChain @tool，被 agent.py（LangGraph）与 mcp_server.py（MCP）共用。
工作区用 contextvars 隔离，与 rosetta/boltz agent 同一套机制。
"""
import os
import glob
import json
import re
import time
import shutil
import subprocess
import contextvars
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# GROMACS 2026.2 与 gmx_MMPBSA（服务器 conda 环境）
GMX = os.getenv("GMX_BIN", "/opt/gromacs/bin/gmx")
GMX_MMPBSA = os.getenv("GMX_MMPBSA_BIN", "/opt/gromacs/bin/gmx_MMPBSA")

# 线程安全的上下文变量，多任务并发时目录物理隔离
current_workspace = contextvars.ContextVar('current_workspace', default='.')


def get_workspace_path(sub_dir: str = "") -> str:
    """获取当前任务的独立目录（绝对路径），并自动创建"""
    base_dir = os.path.abspath(current_workspace.get())
    full_path = os.path.join(base_dir, sub_dir) if sub_dir else base_dir
    os.makedirs(full_path, exist_ok=True)
    return full_path


def _abs(path: str) -> str:
    """相对路径解析为绝对路径（必须以绝对路径传给 gmx，否则 cwd 会二次拼接）。

    agent 有时会传已含工作区前缀的相对路径（如 output/task_x/a.log），
    若该路径本身存在则直接采用。
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(current_workspace.get(), path))


def _resolve_traj(traj_path: str = "") -> str:
    """解析分析用轨迹：显式存在的路径优先，否则回退工作区最新 .xtc。

    单段生产轨迹（<run>.xtc）无需 trjcat 合并，直接分析即可。
    """
    if traj_path:
        p = _abs(traj_path)
        if os.path.exists(p):
            return p
    ws = os.path.abspath(current_workspace.get())
    merged = os.path.join(ws, "merged.xtc")
    if os.path.exists(merged):
        return merged
    cands = glob.glob(os.path.join(ws, "*.xtc"))
    if cands:
        return max(cands, key=os.path.getmtime)
    return ""


def _run(cmd: str, stdin: str = "", timeout: int = 600, cwd: str = None) -> tuple:
    """执行 shell 命令，可选 stdin 喂给 gmx 的分组选择。返回 (rc, stdout, stderr)"""
    try:
        p = subprocess.run(
            cmd, shell=True, cwd=os.path.abspath(cwd or current_workspace.get()),
            input=stdin.encode() if stdin else None,
            capture_output=True, timeout=timeout,
        )
        return p.returncode, p.stdout.decode('utf-8', 'replace'), p.stderr.decode('utf-8', 'replace')
    except subprocess.TimeoutExpired:
        return 124, "", f"命令超时 ({timeout}s): {cmd}"
    except Exception as e:
        return 1, "", f"【系统异常】{e}"


# ==========================================
# 1. 基础读取与诊断工具
# ==========================================
@tool
def read_file(file_path: str, max_lines: int = 60) -> str:
    """读取指定文件内容。用于查看 .xvg 结果、日志、目录列表等。"""
    try:
        p = _abs(file_path)
        if not os.path.exists(p):
            return f"【执行报错】文件 {p} 不存在。"
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        if len(lines) > max_lines:
            return "".join(lines[:max_lines]) + f"\n... (截断，共 {len(lines)} 行)"
        return "".join(lines)
    except Exception as e:
        return f"【系统异常】读取文件失败: {e}"


@tool
def check_logs(log_path: str, lines: int = 30) -> str:
    """错误排查工具：仅读取日志最后 N 行，极大节省 Token。"""
    p = _abs(log_path)
    if not os.path.exists(p):
        return f"【日志排查失败】未找到文件 {p}。"
    try:
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            tail = f.readlines()[-lines:]
        return f"【日志截取成功】{p} 最后 {lines} 行：\n" + "".join(tail)
    except Exception as e:
        return f"【系统异常】{e}"


def _detect_groups(tpr_path: str) -> dict:
    """从 tpr 探测默认分组名 -> 索引（Protein / Backbone / LIG / System ...）"""
    rc, out, err = _run(f'echo q | "{GMX}" make_ndx -f "{tpr_path}" 2>&1', timeout=120)
    groups = {}
    for line in out.splitlines():
        m = re.match(r'\s*(\d+)\s+(\S+)\s+:\s+\d+\s+atoms', line)
        if m:
            groups[m.group(2)] = int(m.group(1))
    return groups


@tool
def prepare_analysis_index(tpr_path: str = "step5_1.tpr") -> str:
    """【分析准备】从 .tpr 生成分析用 index.ndx（保留默认组 + 新增 Complex = Protein | LIG）。

    写入 <workspace>/analysis.ndx 与 <workspace>/groups.json（记录各分组索引，
    供后续 RMSD/RMSF/gyrate 等工具引用）。"""
    tpr = _abs(tpr_path)
    if not os.path.exists(tpr):
        return f"【执行报错】未找到 {tpr}"
    groups = _detect_groups(tpr)
    if "Protein" not in groups:
        return "【执行报错】tpr 中未找到 Protein 组，无法建立分析 index。"

    lig_idx = groups.get("LIG")
    ngrp = len(groups)
    sel = []
    if lig_idx is not None:
        sel.append(f"{groups['Protein']} | {lig_idx}")  # -> 新组 Complex
        sel.append(f"name {ngrp} Complex")
        sel.append("q")
        stdin = "\n".join(sel) + "\n"
        rc, out, err = _run(
            f'"{GMX}" make_ndx -f "{tpr}" -o analysis.ndx 2>&1',
            stdin=stdin, timeout=180,
        )
        if rc != 0 or not os.path.exists(_abs("analysis.ndx")):
            return f"【执行报错】make_ndx 失败：\n{err[-800:]}"
        groups["Complex"] = ngrp

    with open(_abs("groups.json"), "w") as f:
        json.dump(groups, f, indent=2)

    want = {k: groups[k] for k in ("System", "Protein", "Backbone", "LIG", "Complex") if k in groups}
    return (
        "【分析 index 已建立】analysis.ndx\n"
        f"关键分组：{want}\n"
        "下一步：compute_rmsd / compute_rmsf / compute_gyrate / compute_hbond"
    )


def _load_groups() -> dict:
    p = _abs("groups.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def _grp(name: str, default: int) -> int:
    return _load_groups().get(name, default)


# ==========================================
# 2. MD 模拟：后台提交 + 状态检查 + 轨迹合并
# ==========================================
@tool
def run_gromacs_md(tpr_path: str = "step5_1.tpr", nsteps: int = 0,
                   run_name: str = "md", ntomp: int = 8) -> str:
    """【MD 模拟】后台提交 gmx mdrun，立即返回（产物 <run_name>.xtc/.gro/.edr/.log）。

    tpr 是自包含的（不需要 topol.top / toppar）。
    nsteps=0 表示用 tpr 原始步数；短测试时可给小值（如 200000 约 400 ps）。

    Args:
        tpr_path: 起始 .tpr（CHARMM-GUI 通常是 step5_1.tpr）
        nsteps: 覆盖步数；0=用 tpr 原值
        run_name: 输出前缀
        ntomp: OpenMP 线程数
    """
    tpr = _abs(tpr_path)
    if not os.path.exists(tpr):
        return f"【执行报错】未找到 {tpr}"
    ws = os.path.abspath(current_workspace.get())
    try:
        n = int(nsteps or 0)
    except (TypeError, ValueError):
        n = 0
    nsteps_flag = f" -nsteps {n}" if n > 0 else ""
    cmd = f'"{GMX}" mdrun -s "{tpr}" -deffnm {run_name}{nsteps_flag} -ntmpi 1 -ntomp {int(ntomp)}'
    log = os.path.join(ws, f"{run_name}.out")
    pidf = os.path.join(ws, f"{run_name}.pid")
    try:
        subprocess.Popen(f'cd {ws} && nohup {cmd} > {run_name}.out 2>&1 & echo $! > {pidf}', shell=True)
        time.sleep(2)
        pid = open(pidf).read().strip() if os.path.exists(pidf) else ""
        steps_txt = f"{n} 步" if n > 0 else "tpr 原始步数"
        return (f"【MD 已挂起】mdrun 后台运行中 (PID={pid}, {steps_txt})\n"
                f"日志: {log}\n完成标志: {run_name}.gro 出现；可用 check_md_status 查看进度")
    except Exception as e:
        return f"【系统异常】提交 mdrun 失败: {e}"


@tool
def check_md_status(run_name: str = "md") -> str:
    """【状态】检查 mdrun 进度：解析日志最后的 Step/Time 与性能(ns/day)。"""
    ws = os.path.abspath(current_workspace.get())
    log = os.path.join(ws, f"{run_name}.log")
    if not os.path.exists(log):
        return f"【MD 状态】未找到 {run_name}.log，可能尚未启动或立即退出（查 {run_name}.out）。"
    try:
        with open(log, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    except Exception as e:
        return f"【系统异常】{e}"
    steps = re.findall(r'^\s*(\d{4,})\s+([\d.]+)\s*$', text, re.M)
    perf = re.findall(r'Performance:\s+([\d.]+)\s+([\d.]+)', text)
    done = os.path.exists(os.path.join(ws, f"{run_name}.gro"))
    lines = []
    if steps:
        lines.append(f"当前步数 {steps[-1][0]} 步，模拟时间 {steps[-1][1]} ps")
    if perf:
        lines.append(f"性能 {perf[-1][0]} ns/day")
    if not lines:
        lines.append("日志中暂无 Step/Time 记录（刚启动或已写完 .gro）")
    lines.append(f"完成标记 ({run_name}.gro): {'是' if done else '否'}")
    return "【MD 状态】\n" + "\n".join(lines)


# ==========================================
# 3. 轨迹合并：把分段生产轨迹拼成连续时间轴
# ==========================================
@tool
def merge_trajectories(tpr_path: str = "step5_1.tpr", traj_list="", out: str = "merged.xtc", start_times: str = "") -> str:
    """【轨迹合并】gmx trjcat 把多段 .xtc 拼成一条连续轨迹。

    CHARMM-GUI 分段生产(step5_1.xtc...)常从 0 ps 重启，须用 start_times
    手动给每段起点(如 "0,20000")，否则时间轴会重叠。

    Args:
        tpr_path: 起始 .tpr（用于校验）
        traj_list: 轨迹列表，逗号分隔字符串或列表
        out: 输出轨迹(默认 merged.xtc)
        start_times: 各段起始时间(ps)，逗号分隔；空则自动续接
    """
    if isinstance(traj_list, (list, tuple)):
        trajs = [str(t).strip() for t in traj_list if str(t).strip()]
    else:
        trajs = [t.strip() for t in str(traj_list).split(",") if t.strip()]
    if not trajs:
        return "【执行报错】traj_list 为空，请给出要合并的 .xtc 列表。"
    for t in trajs:
        if not os.path.exists(_abs(t)):
            return f"【执行报错】轨迹文件不存在: {t}"
    files = ",".join(trajs)
    times = [x.strip() for x in str(start_times).split(",") if x.strip()]
    if times:
        if len(times) != len(trajs):
            return f"【执行报错】start_times 个数({len(times)})与轨迹数({len(trajs)})不一致。"
        setflag, stdin = " -settime", "".join(f"{x}\n" for x in times)
    else:
        setflag, stdin = "", "c\n"
    cmd = f'"{GMX}" trjcat -f {files} -o "{out}"{setflag}'
    rc, o, e = _run(cmd, stdin=stdin, timeout=3600)
    if rc != 0 or not os.path.exists(_abs(out)):
        return f"【执行报错】trjcat 合并失败：\n{(e or o)[-800:]}"
    return f"【轨迹已合并】{out}（{len(trajs)} 段，起点时间={start_times or '自动续接'}）"


def _require_index():
    """确保 analysis.ndx 存在，返回其路径或 None"""
    ndx = _abs("analysis.ndx")
    return ndx if os.path.exists(ndx) else None


# ==========================================
# 4. 结构稳定性：RMSD / RMSF / 回转半径 / 氢键
# ==========================================
@tool
def compute_rmsd(tpr_path: str = "step5_1.tpr", traj_path: str = "merged.xtc",
                 fit_group: str = "Backbone", calc_group: str = "Backbone",
                 out: str = "rmsd.xvg") -> str:
    """【RMSD】gmx rms：先按 fit_group 叠合，再算 calc_group 的 RMSD。

    默认 Backbone↔Backbone(骨架稳定性)。配体用 calc_group="LIG"，
    复合物用 "Complex"。分组名经 analysis.ndx/groups.json 解析。
    """
    tpr, traj = _abs(tpr_path), _resolve_traj(traj_path)
    if not os.path.exists(tpr):
        return f"【执行报错】未找到 {tpr}"
    if not traj or not os.path.exists(traj):
        return "【执行报错】工作区内未找到可用轨迹(.xtc)，请先完成 MD 或显式给出 traj_path。"
    ndx = _require_index()
    if not ndx:
        return "【执行报错】缺少 analysis.ndx，请先 prepare_analysis_index。"
    fit, calc = _grp(fit_group, 4), _grp(calc_group, 4)
    cmd = f'"{GMX}" rms -s "{tpr}" -f "{traj}" -n "{ndx}" -o "{out}"'
    rc, o, e = _run(cmd, stdin=f"{fit}\n{calc}\n", timeout=3600)
    if rc != 0 or not os.path.exists(_abs(out)):
        return f"【执行报错】gmx rms 失败：\n{(e or o)[-800:]}"
    return f"【RMSD 完成】{out}（fit={fit_group}#{fit}, calc={calc_group}#{calc}）"


@tool
def compute_rmsf(tpr_path: str = "step5_1.tpr", traj_path: str = "merged.xtc",
                 group: str = "Backbone", out: str = "rmsf.xvg") -> str:
    """【RMSF】gmx rmsf -res：逐残基涨落，x 轴为残基号。"""
    tpr, traj = _abs(tpr_path), _resolve_traj(traj_path)
    if not os.path.exists(tpr):
        return f"【执行报错】未找到 {tpr}"
    if not traj or not os.path.exists(traj):
        return "【执行报错】工作区内未找到可用轨迹(.xtc)，请先完成 MD 或显式给出 traj_path。"
    ndx = _require_index()
    if not ndx:
        return "【执行报错】缺少 analysis.ndx，请先 prepare_analysis_index。"
    grp = _grp(group, 4)
    cmd = f'"{GMX}" rmsf -s "{tpr}" -f "{traj}" -n "{ndx}" -o "{out}" -res'
    rc, o, e = _run(cmd, stdin=f"{grp}\n", timeout=3600)
    if rc != 0 or not os.path.exists(_abs(out)):
        return f"【执行报错】gmx rmsf 失败：\n{(e or o)[-800:]}"
    return f"【RMSF 完成】{out}（group={group}#{grp}）"


@tool
def compute_gyrate(tpr_path: str = "step5_1.tpr", traj_path: str = "merged.xtc",
                   group: str = "Protein", out: str = "gyrate.xvg") -> str:
    """【回转半径】gmx gyrate：Rg 随时间，衡量蛋白紧致度。"""
    tpr, traj = _abs(tpr_path), _resolve_traj(traj_path)
    if not os.path.exists(tpr):
        return f"【执行报错】未找到 {tpr}"
    if not traj or not os.path.exists(traj):
        return "【执行报错】工作区内未找到可用轨迹(.xtc)，请先完成 MD 或显式给出 traj_path。"
    ndx = _require_index()
    if not ndx:
        return "【执行报错】缺少 analysis.ndx，请先 prepare_analysis_index。"
    grp = _grp(group, 1)
    cmd = f'"{GMX}" gyrate -s "{tpr}" -f "{traj}" -n "{ndx}" -o "{out}"'
    rc, o, e = _run(cmd, stdin=f"{grp}\n", timeout=3600)
    if rc != 0 or not os.path.exists(_abs(out)):
        return f"【执行报错】gmx gyrate 失败：\n{(e or o)[-800:]}"
    return f"【回转半径完成】{out}（group={group}#{grp}）"


@tool
def compute_hbond(tpr_path: str = "step5_1.tpr", traj_path: str = "merged.xtc",
                  out: str = "hbond.xvg") -> str:
    """【氢键】gmx hbond：Protein-LIG 氢键数随时间。

    GROMACS 2026 起 -o 只写 index(.ndx)，氢键数曲线走 -num：
    计数曲线 <out>_count.xvg（出图用这个），配对索引 <out>_bonds.ndx。
    """
    tpr, traj = _abs(tpr_path), _resolve_traj(traj_path)
    if not os.path.exists(tpr):
        return f"【执行报错】未找到 {tpr}"
    if not traj or not os.path.exists(traj):
        return "【执行报错】工作区内未找到可用轨迹(.xtc)，请先完成 MD 或显式给出 traj_path。"
    ndx = _require_index()
    if not ndx:
        return "【执行报错】缺少 analysis.ndx，请先 prepare_analysis_index。"
    prot, lig = _grp("Protein", 1), _grp("LIG", 13)
    stem = os.path.splitext(out)[0]
    count_out = stem + "_count.xvg"
    bonds_ndx = stem + "_bonds.ndx"
    cmd = (f'"{GMX}" hbond -s "{tpr}" -f "{traj}" -n "{ndx}" '
           f'-num "{count_out}" -o "{bonds_ndx}"')
    rc, o, e = _run(cmd, stdin=f"{prot}\n{lig}\n", timeout=3600)
    if rc != 0 or not os.path.exists(_abs(count_out)):
        return f"【执行报错】gmx hbond 失败：\n{(e or o)[-800:]}"
    return (f"【氢键完成】{count_out}（Protein#{prot} - LIG#{lig}）\n"
            f"配对索引: {bonds_ndx}")


# ==========================================
# 5. MM/GBSA 结合自由能（gmx_MMPBSA，后台提交）
# ==========================================
_MMPBSA_IN = """&general
   startframe = 1,
   endframe = 0,
   interval = 1,
   verbose = 2,
/
&gb
   igb = 5,
   saltcon = 0.150,
/
&decomp
   idecomp = 1,
   dec_verbose = 3,
/
"""


@tool
def run_mmpbsa(tpr_path: str = "step5_1.tpr", traj_path: str = "merged.xtc",
               index_path: str = "analysis.ndx", out: str = "FINAL_RESULTS_MMPBSA.dat") -> str:
    """【MM/GBSA】后台提交 gmx_MMPBSA，算蛋白-配体 GB 结合能 + 逐残基分解。

    生成 mmpbsa.in（igb=5, saltcon=0.15, idecomp=1 逐残基分解），
    -cg Protein LIG 指定两组。若工作区有 topol.top 则 -cp topol.top（标准模式）；
    否则走 tpr/CHARMM-GUI 模式。产物 <out> 完成后出现在工作区。
    """
    ws = os.path.abspath(current_workspace.get())
    tpr, traj, ndx = _abs(tpr_path), _abs(traj_path), _abs(index_path)
    if not os.path.exists(tpr):
        return f"【执行报错】未找到 {tpr}"
    if not os.path.exists(traj):
        return f"【执行报错】未找到 {traj}"
    if not os.path.exists(ndx):
        return f"【执行报错】未找到 {ndx}，请先 prepare_analysis_index。"
    with open(os.path.join(ws, "mmpbsa.in"), "w", encoding="utf-8") as f:
        f.write(_MMPBSA_IN)
    topol = os.path.join(ws, "topol.top")
    mode = "标准(topol.top)" if os.path.exists(topol) else "CHARMM-GUI/tpr"
    cmd = (f'"{GMX_MMPBSA}" -O -i mmpbsa.in -cs "{tpr}" -ct "{traj}" '
           f'-ci "{ndx}" -cg Protein LIG -nogui')
    if os.path.exists(topol):
        cmd += ' -cp "topol.top"'
    log = os.path.join(ws, "mmpbsa.log")
    pidf = os.path.join(ws, "mmpbsa.pid")
    try:
        subprocess.Popen(f'cd {ws} && nohup {cmd} > mmpbsa.log 2>&1 & echo $! > {pidf}', shell=True)
        time.sleep(2)
        pid = open(pidf).read().strip() if os.path.exists(pidf) else ""
        return (f"【MM/GBSA 已挂起】gmx_MMPBSA 后台运行中 (PID={pid}, 模式={mode})\n"
                f"日志: {log}\n完成标志: {out} 出现")
    except Exception as e:
        return f"【系统异常】提交 gmx_MMPBSA 失败: {e}"


# ==========================================
# 6. GPCR Nature 风格出图（风格原语见 plot_style.py，与 GPCR 分层出图共用）
# ==========================================
from plot_style import (  # noqa: E402
    INK, GREY, COLORS, setup_style, clean, panel, save,
)
import gpcr_plots as _GP  # noqa: E402


def _read_xvg(path):
    """读 .xvg，跳过 #/@ 注释行，返回 (xs, ys)——取第 1、2 列。"""
    xs, ys = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in ("#", "@"):
                continue
            parts = s.split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) < 2:
                continue
            xs.append(vals[0])
            ys.append(vals[1])
    return xs, ys


_PANEL_SPEC = {
    "rmsd":   ("RMSD", "Time (ps)", "RMSD (nm)"),
    "rmsf":   ("RMSF", "Residue", "RMSF (nm)"),
    "gyrate": ("Radius of gyration", "Time (ps)", "Rg (nm)"),
    "hbond":  ("Hydrogen bonds", "Time (ps)", "Count"),
}


@tool
def plot_md_figures(xvg_map="", out_prefix: str = "figs/md") -> str:
    """【出图】读多个 .xvg（rmsd/rmsf/gyrate/hbond），GPCR Nature 风格出图，保存 PDF + PNG(600dpi)。

    Args:
        xvg_map: dict 或 "key=path,key=path"（key ∈ rmsd/rmsf/gyrate/hbond），也接受 JSON
        out_prefix: 输出前缀，生成 <out_prefix>.pdf 与 <out_prefix>.png
    """
    if isinstance(xvg_map, dict):
        m = dict(xvg_map)
    else:
        text = str(xvg_map).strip()
        m = {}
        if text.startswith("{"):
            try:
                m = json.loads(text)
            except Exception:
                m = {}
        if not m and text:
            for tok in text.split(","):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    m[k.strip()] = v.strip()
    order = ["rmsd", "rmsf", "gyrate", "hbond"]
    panels = [(k, m[k]) for k in order if k in m and os.path.exists(_abs(m[k]))]
    if not panels:
        return "【执行报错】xvg_map 中未找到可读 .xvg（键应为 rmsd/rmsf/gyrate/hbond）。"

    setup_style()
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(3.6, 1.45 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]
    letters = "ABCDEFGH"
    for i, (key, path) in enumerate(panels):
        ax = axes[i]
        xs, ys = _read_xvg(_abs(path))
        title, xlabel, ylabel = _PANEL_SPEC.get(key, (key, "x", "y"))
        ax.plot(xs, ys, color=COLORS["A"], lw=0.9)
        clean(ax, grid=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        panel(ax, letters[i])
    out_abs = _abs(out_prefix)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    save(fig, out_abs)
    return (f"【出图完成】{len(panels)} 个面板\n"
            f"PDF: {out_prefix}.pdf\nPNG: {out_prefix}.png (600dpi)")


# ==========================================
# 7. 诊断与修复工具（repair 分支）
# ==========================================
# (命中子串元组, 诊断建议)
_GROMACS_HINTS = [
    (("lincs warning", "lincs error"),
     "LINCS 约束警告：多为时间步过大或体系 clash。建议减小 dt 或放宽约束；持续警告会导致 NaN。"),
    ((" nan ",),
     "出现 NaN：数值发散。检查 PBC/参数冲突或时间步过大，考虑用更小 dt 重跑。"),
    (("lost atoms", "number of atoms", "atom count"),
     "原子丢失/原子数不一致：拓扑与轨迹不匹配。trjcat 合并务必给对 start_times；"
     "MMPBSA 的 -cs/-ct 原子数必须一致。"),
    (("out of memory", "cuda", "memory allocation", "cudaerror"),
     "CUDA 显存不足：调小 -ntomp，或降低输出频率/换卡重跑。"),
    (("overlap", "not monoton"),
     "trjcat 时间重叠：分段轨迹从 0 重启，请用 start_times 手动指定每段起点。"),
]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


@tool
def diagnose_and_fix(tool_name: str) -> str:
    """GROMACS 诊断：读工作区 .log/.out 判定常见失败（LINCS/NaN/丢失原子/CUDA 显存/trjcat 重叠）。"""
    ws = os.path.abspath(current_workspace.get())
    logs = glob.glob(os.path.join(ws, "*.log")) + glob.glob(os.path.join(ws, "*.out"))
    for p in logs:
        try:
            text = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        low = text.lower()
        for keys, hint in _GROMACS_HINTS:
            if any(k in low for k in keys):
                tail = "".join(text.splitlines(True)[-8:])
                return f"【诊断·{os.path.basename(p)}】工具 {tool_name} 命中「{keys[0]}」，建议：{hint}\n日志末尾：\n{tail}"
    if not logs:
        return f"【诊断】工作区 {ws} 下无 .log/.out，任务可能尚未启动。"
    return "【诊断】已检查工作区日志，未命中已知 GROMACS 错误模式，请用 check_logs 人工查看。"


@tool
def auto_repair() -> str:
    """产物缺失时自动重跑：.gro 缺失→重提交 mdrun；.gro 有但无 merged.xtc→提示合并轨迹。"""
    ws = os.path.abspath(current_workspace.get())
    tprs = glob.glob(os.path.join(ws, "*.tpr"))
    gro = glob.glob(os.path.join(ws, "*.gro"))
    running = False
    for f in glob.glob(os.path.join(ws, "*.pid")):
        try:
            if _pid_alive(int(open(f).read().strip())):
                running = True
        except Exception:
            pass
    if tprs and not gro and not running:
        tpr = os.path.abspath(tprs[0])
        subprocess.Popen(
            f'cd {ws} && nohup "{GMX}" mdrun -s "{tpr}" -deffnm md -ntmpi 1 -ntomp 8 > md.out 2>&1 &',
            shell=True,
        )
        return f"【自动修复】未发现 .gro，已重新提交 gmx mdrun（-deffnm md, tpr={os.path.basename(tpr)}）。"
    if gro and not os.path.exists(os.path.join(ws, "merged.xtc")):
        segs = [os.path.basename(s) for s in sorted(glob.glob(os.path.join(ws, "*.xtc")))]
        return f"【自动修复建议】有 .gro 但缺 merged.xtc，分段轨迹={segs}，请调用 merge_trajectories。"
    if gro:
        return "【无需修复】已存在 .gro 产物，MD 正常。"
    return "【修复失败】工作区缺少 .tpr，无法重跑。"


# ==========================================
# 8. RAG 文档检索工具（GROMACS 命令/分析说明）
# ==========================================
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manuals", "md_docs")


def _load_doc_chunks():
    """加载 manuals/md_docs 下全部 .txt/.md 语料并切块。"""
    docs = []
    for root, _dirs, files in os.walk(DOCS_DIR):
        for fn in sorted(files):
            if not fn.lower().endswith((".txt", ".md")):
                continue
            path = os.path.join(root, fn)
            try:
                docs.extend(TextLoader(path, encoding="utf-8").load())
            except Exception:
                continue
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    return splitter.split_documents(docs)


def init_bm25_retriever():
    """BM25 本地关键词检索器（零 API 依赖，主检索方案）。"""
    try:
        from langchain_community.retrievers import BM25Retriever
        splits = _load_doc_chunks()
        if not splits:
            print("⚠️ RAG：manuals/md_docs 下未找到语料，检索不可用")
            return None
        return BM25Retriever.from_documents(splits, k=4)
    except Exception as e:
        print(f"⚠️ RAG：BM25 检索器构建失败: {e}")
        return None


def init_vector_db():
    """向量库（可选；embedding API 不可用时优雅降级为 None）。"""
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma

        embeddings_kwargs = dict(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", ""),
        )
        emb_model = os.getenv("EMBEDDING_MODEL", "").strip()
        if emb_model:
            embeddings_kwargs["model"] = emb_model
        embeddings = OpenAIEmbeddings(**embeddings_kwargs)

        if os.path.exists("chroma_db"):
            return Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        splits = _load_doc_chunks()
        if splits:
            return Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="chroma_db")
    except Exception as e:
        print(f"⚠️ 向量库构建失败（回退 BM25）: {e}")
        try:
            if os.path.isdir("chroma_db"):
                shutil.rmtree("chroma_db")
        except Exception:
            pass
    return None


bm25_retriever = init_bm25_retriever()
vector_db = init_vector_db()


@tool
def search_md_docs(query: str) -> str:
    """RAG 检索：检索 GROMACS 常用命令/RMSD/RMSF/gyrate/hbond/MMPBSA 说明，遇到未知参数或用法时调用。"""
    docs = []
    source = None
    if vector_db is not None:
        try:
            docs = vector_db.similarity_search(query, k=4)
            source = "向量检索"
        except Exception:
            docs = []
    if not docs and bm25_retriever is not None:
        try:
            docs = bm25_retriever.invoke(query)
            source = "BM25 关键词检索"
        except Exception:
            docs = []
    if not docs:
        return "【检索结果】MD 知识库中未找到相关内容（请检查 manuals/md_docs 语料）。"
    context = "\n---\n".join(d.page_content[:1500] for d in docs)
    return f"【检索成功·{source}】以下是 GROMACS/MD 知识库的参考资料：\n{context}"


# ==========================================
# 9. GPCR / 元动力学 重分析工具（走 worker 子进程）
# ==========================================
# agent venv 里没有 MDAnalysis/scipy/sklearn，重分析一律交给分析 env 的子进程。
_DEF_ANALYSIS_PY = '/opt/gromacs/bin/python'
ANALYSIS_PY = os.getenv('MD_ANALYSIS_PY', _DEF_ANALYSIS_PY)
_HERE = os.path.dirname(os.path.abspath(__file__))
WORKER_PY = os.getenv('MD_WORKER_PY', os.path.join(_HERE, 'worker.py'))

# analyze_uncertainty 不传 series 时的默认指标集（跨受体通用：两个保守盐桥/开关 + TM6 间隙 + 配体）
_DEF_UNC_SERIES = ('microswitch_DRY.csv:3.49-3.50,microswitch_CWxP.csv:6.48 chi1,'
                   'microswitch_NPxxY.csv:7.50-7.53,microswitch_PIF.csv:3.40-6.44,'
                   'tm6_opening.csv:TM6ic_TM3ic_nm,ligand_retention.csv:ligand_rmsd_nm')


def _py_analysis(task, payload=None, timeout=3600):
    """在分析 env 里跑 worker 的某个任务，返回解析后的 dict。

    失败一律返回 {'ok': False, 'error': ...}，不抛异常：工具层不该因分析失败而崩，
    让 agent 拿到错误文本自己决定重试或换参数。
    """
    ws = os.path.abspath(current_workspace.get())
    cmd = [ANALYSIS_PY, WORKER_PY, str(task), '--ws', ws]
    cmd += ['--json', json.dumps(payload or {}, ensure_ascii=False)]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=ws)
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': '分析任务 %s 超时（%ds）' % (task, timeout)}
    except Exception as e:
        return {'ok': False, 'error': '调用分析 worker 失败: %s' % e}
    out = p.stdout.decode('utf-8', 'replace').strip()
    line = out.splitlines()[-1] if out else ''
    try:
        return json.loads(line)
    except Exception:
        err = p.stderr.decode('utf-8', 'replace') or out
        return {'ok': False, 'error': 'worker 未返回 JSON：%s' % err[-1000:]}


def _py_tool(task, summary_fn, timeout=3600, **payload):
    """跑 worker 任务 -> 全量结果落盘 analysis_<task>.json -> 只把精简小结返回给 LLM。

    大 JSON 直接进上下文很浪费；落盘 + 返回小结，需要细节时 agent 用 read_file 取。
    """
    ws = os.path.abspath(current_workspace.get())
    r = _py_analysis(task, payload, timeout=timeout)
    fp = os.path.join(ws, 'analysis_%s.json' % task)
    try:
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(r, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        fp = ''
    if not r.get('ok'):
        return '【%s 失败】%s' % (task, str(r.get('error'))[:900])
    try:
        body = summary_fn(r)
    except Exception as e:
        body = '（小结生成失败: %s）' % e
    tail = ''
    if fp:
        tail = '\n完整结果: %s（需要细节用 read_file 查看）' % os.path.basename(fp)
    return '【%s 完成】\n%s%s' % (task, body, tail)


@tool
def detect_receptor_profile(top_path: str = "", traj_path: str = "") -> str:
    """【受体编号识别】提取受体序列，识别 GPCR motif（DRY/CWxP/NPxxY）与跨膜段，
    写出 gpcr.json 与 BW 通用编号（如 DRY=D3.49/R3.50/Y3.51）。

    **所有 GPCR 分析之前必须先跑这一步**：微开关位点、TM/ICL/ECL 段定义都依赖它。
    命中内置档案会标注「已人工核对」，否则为自动识别（建议人工确认残基号）。
    top_path 给 .gro/.pdb（分析库读不了 GROMACS 2026 的 .tpr）；留空则自动在工作区里找。
    """
    def sm(r):
        out = [str(r.get('summary', ''))]
        out.append('受体原子 %s / 配体原子 %s；%s 帧，帧间隔 %s ns' % (
            r.get('n_receptor_atoms'), r.get('n_ligand_atoms'),
            r.get('n_frames'), r.get('dt_ns')))
        if r.get('ligand_resnames'):
            out.append('配体残基名：%s' % ', '.join(r['ligand_resnames']))
        return '\n'.join(out)
    return _py_tool('detect_numbering', sm, timeout=2400, top=top_path, traj=traj_path)


@tool
def analyze_segments(top_path='', traj_path='', segments='', fit_segments='', stride=0):
    """【分段稳定性】以核心段（默认 TM1-TM4）叠合后，逐段算 Cα RMSD 时间序列与逐残基 RMSF。

    出 segments_rmsd.csv / segments_rmsf.csv。这是判断"哪条螺旋在动、哪个 loop 在飘"
    的第一手证据（比整条链一个 RMSD 有用得多）。跨膜段相对稳定、ICL/ECL 明显更飘是正常预期。
    """
    def sm(r):
        seg = r.get('segments', {})
        out = ['叠合段 %s；用了 %s 帧（跨度 %s ns）' % (
            r.get('fit_segments'), r.get('n_frames_used'), r.get('time_span_ns'))]
        for k in seg:
            v = seg[k]
            out.append('  %-6s %-12s RMSD=%s nm  RMSF=%s nm' % (
                k, v.get('range'), _r3(v.get('rmsd_mean_nm')), _r3(v.get('rmsf_mean_nm'))))
        tf = r.get('top_flexible_residues', [])[:5]
        if tf:
            out.append('  最飘的残基：' + ', '.join(
                '%s(%s)=%s nm' % (x['resnum'], x.get('bw') or '-', _r3(x['rmsf_nm'])) for x in tf))
        out.append('  源档案：%s' % r.get('profile_source'))
        return '\n'.join(out)
    pl = {'top': top_path, 'traj': traj_path, 'segments': segments,
          'fit_segments': fit_segments, 'stride': stride}
    return _py_tool('segments', sm, **pl)


def _r3(v):
    """数值格式化成 3 位小数；None / 非数值原样返回。"""
    try:
        return '%.3f' % float(v)
    except (TypeError, ValueError):
        return str(v)


import sys as _sys
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
import gpcr_defs as GT


def _r1(v):
    return _fmtn(v, 1)


def _r2(v):
    return _fmtn(v, 2)


def _fmtn(v, n):
    try:
        return ('%.' + str(int(n)) + 'f') % float(v)
    except (TypeError, ValueError):
        return str(v)


@tool
def analyze_microswitches(top_path='', traj_path='', switches='', stride=0):
    """【微开关】GPCR 保守微开关几何：PIF / CWxP / NPxxY / DRY / 离子锁。

    每个开关算：关键原子间距离（均值/最小/占据率）、chi1/chi2 圆统计与 rotamer 占据率、
    NPxxY 基序 Cα RMSD，出 microswitch_<名字>.csv。这是"受体处于什么状态"的核心证据：
    DRY 盐桥是否完整、W6.48 是哪个 rotamer、Y7.53 酚羟基朝向、PIF 三元组是否崩解。
    受体缺少某位点（如 APLNR 无 E6.30）时会列进 missing_measurements，而不是编造一个残基。
    """
    def sm(r):
        sw = r.get('switches', {})
        out = ['用了 %s 帧（跨度 %s ns）' % (r.get('n_frames_used'), r.get('time_span_ns'))]
        for name in GT.MICROSWITCH_ORDER:
            e = sw.get(name)
            if not e:
                continue
            out.append('%s：%s' % (name, e.get('summary', '')))
            for lbl, v in (e.get('pairs') or {}).items():
                out.append('    距离 %-12s 均值=%s nm 最小=%s 占据(<0.4nm)=%s' % (
                    lbl, _r3(v['mean_nm']), _r3(v['min_nm']), _r2(v['occ_0.4nm'])))
            for lbl, v in (e.get('dihedrals') or {}).items():
                out.append('    二面角 %-12s 均值=%s 度 R=%s rotamer=%s' % (
                    lbl, _r1(v['mean_deg']), _r2(v['R']), v.get('rotamer_occ')))
        mc = r.get('motif_ca_rmsd') or {}
        for k, v in mc.items():
            out.append('基序 Cα RMSD %s = %s nm' % (k, _r3(v['ca_rmsd_mean_nm'])))
        miss = r.get('missing_measurements') or []
        if miss:
            out.append('未能测量（位点缺失或原子名不匹配）：%s' % ', '.join(miss))
        return '\n'.join(out)
    return _py_tool('microswitches', sm, top=top_path, traj=traj_path,
                    switches=switches, stride=stride)


@tool
def analyze_tm6_opening(top_path='', traj_path='', ic_len=6, stride=0):
    """【TM6 激活位移】TM6 胞内端到 TM3/TM7 胞内端的 COM 距离随时间 + 相对首帧位移。

    出 tm6_opening.csv。class A GPCR 激活最标志性的几何变化就是 TM6 胞内端外移，
    所以这条曲线是"有没有往激活态走"的首选判据（结合微开关一起看）。
    """
    def sm(r):
        a = r.get('TM6ic_TM3ic_nm', {})
        b = r.get('TM6ic_TM7ic_nm', {})
        s = r.get('TM6ic_shift_nm', {})
        return ('TM6ic 残基 %s / TM3ic %s / TM7ic %s\n'
                '  TM6ic-TM3ic: 首=%s 末=%s 均值=%s nm Δ=%s nm\n'
                '  TM6ic-TM7ic: 首=%s 末=%s 均值=%s nm Δ=%s nm\n'
                '  TM6ic 相对首帧位移: 末=%s 最大=%s nm') % (
            r.get('residues', {}).get('TM6ic'), r.get('residues', {}).get('TM3ic'),
            r.get('residues', {}).get('TM7ic'),
            _r3(a.get('first_nm')), _r3(a.get('last_nm')), _r3(a.get('mean_nm')), _r3(a.get('delta_nm')),
            _r3(b.get('first_nm')), _r3(b.get('last_nm')), _r3(b.get('mean_nm')), _r3(b.get('delta_nm')),
            _r3(s.get('last_nm')), _r3(s.get('max_nm')))
    return _py_tool('tm6_opening', sm, top=top_path, traj=traj_path, ic_len=ic_len, stride=stride)


@tool
def analyze_ligand_retention(top_path='', traj_path='', contact_cutoff_nm=0.45, stride=0):
    """【配体保持】受体核心叠合后的配体 RMSD / COM 位移 / 最近距离 / 接触数 + 逐残基接触占据率。

    出 ligand_retention.csv 与 ligand_contacts.csv（按占据率排序的持续接触残基）。
    持续接触表是"配体结合在哪个口袋"的直接证据：正构口袋的典型模式是
    ECL2 + TM2/3/5/6/7；如果冒出别的区域，说明配体跑偏或体系有问题。
    """
    def sm(r):
        out = []
        rm = r.get('ligand_rmsd_nm', {})
        out.append('配体 RMSD 均值=%s 末=%s 最大=%s nm' % (
            _r3(rm.get('mean_nm')), _r3(rm.get('last_nm')), _r3(rm.get('max_nm'))))
        cs = r.get('ligand_com_shift_nm', {})
        md = r.get('min_dist_nm', {})
        nc = r.get('n_contact_atoms', {})
        out.append('COM 位移末=%s nm；最近距离均值=%s nm；平均接触原子=%s 个' % (
            _r3(cs.get('last_nm')), _r3(md.get('mean_nm')), _r1(nc.get('mean'))))
        out.append('持续接触残基（占据率降序）：')
        for x in (r.get('persistent_contacts') or [])[:10]:
            out.append('    %4s %-7s %-6s 占据=%s' % (
                x['resnum'], x.get('bw') or '-', x.get('segment'), _r2(x['occupancy'])))
        return '\n'.join(out)
    return _py_tool('ligand_retention', sm, top=top_path, traj=traj_path,
                    contact_cutoff_nm=contact_cutoff_nm, stride=stride)


@tool
def analyze_landscape(top_path='', traj_path='', n_states=4, cluster='kmeans', stride=0):
    """【构象景观】PCA + tICA（含 lag 扫描与 ITS）+ 聚类 + KDE 自由能面 + 状态占比/dwell/转移矩阵。

    出 landscape_projection.csv / landscape_states.csv / landscape_fes_pc.csv /
    landscape_fes_tica.csv / landscape_transition.csv。
    回答"体系在哪些构象态之间跳、每个态待多久、态间怎么转"。
    注意 FES 是轨迹投影的 KDE 估计，**不是重加权 FES**；要重加权用 analyze_fes。
    """
    def sm(r):
        out = []
        ev = r.get('explained_variance_ratio', [])[:4]
        out.append('PC 解释方差前 4 个：%s' % [_r3(v) for v in ev])
        out.append('tICA 特征值：%s（lag=%s 帧）' % (
            [_r3(v) for v in r.get('tica_eigenvalues', [])], r.get('config', {}).get('tica_lag_frames')))
        its = r.get('its_shortest_ns_by_lag', {})
        out.append('慢模式 ITS(ns) 随 lag：%s' % {k: (_r2(v) if isinstance(v, (int, float)) else v) for k, v in its.items()})
        st = r.get('states', {})
        out.append('状态占比：%s' % {k: _r3(v) for k, v in (st.get('occupancy') or {}).items()})
        for k, v in (st.get('dwell') or {}).items():
            out.append('    态 %s：出现 %s 次，平均停留 %s ns，最长 %s ns' % (
                k, v.get('n_dwell'), _r3(v.get('mean_ns')), _r3(v.get('max_ns'))))
        out.append('提示：%s' % r.get('caveat', ''))
        return '\n'.join(out)
    return _py_tool('landscape', sm, timeout=5400, top=top_path, traj=traj_path,
                    n_states=n_states, cluster=cluster, stride=stride)


@tool
def analyze_uncertainty(series='', n_block=6, n_boot=2000):
    """【不确定度】对指定时间序列做分块 bootstrap 95% CI + 自相关时间 tau + 有效样本量 ESS。

    series 形如 'microswitch_DRY.csv:3.49-3.50,tm6_opening.csv:TM6ic_TM3ic_nm'（逗号分隔）。
    **报任何均值之前都应该先跑这个**：MD 时间序列有强自相关，普通标准差会大幅低估误差；
    ESS 只有个位数时，说明这条曲线在当前时长里没有足够独立样本支撑结论。
    不给 series 时用一套默认（微开关/离子锁/TM6 间隙/配体 RMSD），
    默认序列里不存在的文件会单条报"找不到"，不影响其余序列。
    """
    if not str(series).strip():
        series = _DEF_UNC_SERIES

    def sm(r):
        out = ['分块 %s / bootstrap %s 次' % (r.get('n_block'), r.get('n_boot'))]
        for k, v in (r.get('series') or {}).items():
            if 'error' in v:
                out.append('  %s -> %s' % (k, v['error']))
                continue
            c = v.get('ci95', {})
            e = v.get('ess', {})
            out.append('  %-40s n=%s 均值=%s' % (k, v.get('n'), _fmtn(v.get('mean'), 4)))
            out.append('      95%%CI=[%s, %s]  tau=%s 帧  ESS=%s' % (
                _fmtn(c.get('lo'), 4), _fmtn(c.get('hi'), 4), _r1(e.get('tau')), _r1(e.get('ess'))))
        return '\n'.join(out)
    return _py_tool('uncertainty', sm, timeout=900, series=series,
                    n_block=n_block, n_boot=n_boot)


@tool
def analyze_correlation(top_path='', traj_path='', min_seq_sep=10, stride=0):
    """【相关运动】DCCM（Cα 位移矢量相关性，线性）+ 互信息 MI（高斯估计）+ 段间耦合矩阵。

    出 dccm.csv / mutual_information.csv / segment_coupling.csv。
    min_seq_sep 用来滤掉同螺旋 i→i+4 这类平凡近邻耦合（默认 10）。
    段间耦合矩阵（TM×TM）是读机制的地方——哪几条螺旋在协同运动。
    注意 MI 是高斯估计、DCCM 只看线性相关，两者都不是完整的非线性分析。
    """
    def sm(r):
        out = ['用了 %s 帧，%s 个 Cα' % (r.get('n_frames_used'), r.get('n_ca'))]
        out.append('平均 |DCCM|=%s；平均 MI=%s nats' % (
            _fmtn(r.get('mean_abs_dccm_offdiag'), 4), _fmtn(r.get('mean_mi_nats'), 4)))
        out.append('段间耦合（平均 |DCCM|）前几名：')
        for x in (r.get('top_segment_coupling') or [])[:6]:
            out.append('    %-6s - %-6s  %s' % (x['seg1'], x['seg2'], _fmtn(x['mean_abs_dccm'], 4)))
        out.append('最强远程序列耦合：')
        for x in (r.get('top_dccm_pairs') or [])[:5]:
            out.append('    DCCM %s(%s) - %s(%s) = %s' % (
                x['res1'], x.get('bw1') or '-', x['res2'], x.get('bw2') or '-', _r3(x['dccm'])))
        for x in (r.get('top_mi_pairs') or [])[:3]:
            out.append('    MI   %s(%s) - %s(%s) = %s nats' % (
                x['res1'], x.get('bw1') or '-', x['res2'], x.get('bw2') or '-', _fmtn(x['mi_nats'], 3)))
        return '\n'.join(out)
    return _py_tool('dccm_mi', sm, timeout=5400, top=top_path, traj=traj_path,
                    min_seq_sep=min_seq_sep, stride=stride)


@tool
def analyze_network(top_path='', traj_path='', contact_cutoff_nm=0.8, edge_freq=0.5, stride=0):
    """【残基网络 PSN】接触图 + 介数中心性 + 标签传播社区检测 + 前后半段社区重排。

    出 psn_residues.csv / psn_contacts.csv。高介数残基 = 网络枢纽（变构信号最可能
    经过的地方）；社区重排 = 哪些残基在轨迹前后半段改变了网络归属（常是状态转变信号）。
    contact_cutoff_nm 是 Cα-Cα 建边阈值，edge_freq 是边需在多少比例的帧里存在。
    """
    def sm(r):
        out = ['%s 帧；%s 条边；%s 个社区（规模 %s）' % (
            r.get('n_frames_used'), r.get('n_edges'),
            r.get('n_communities'), r.get('community_sizes'))]
        out.append('介数中心性最高的残基（网络枢纽）：')
        for x in (r.get('top_betweenness') or [])[:8]:
            out.append('    %4s %-7s %-6s bc=%s 度=%s' % (
                x['resnum'], x.get('bw') or '-', x.get('segment'),
                _r1(x['betweenness']), int(x['degree'])))
        out.append('前后半段社区重排残基 %s 个：' % r.get('n_rewired'))
        for x in (r.get('rewired_residues') or [])[:12]:
            out.append('    %4s %-7s %-6s  %s -> %s' % (
                x['resnum'], x.get('bw') or '-', x.get('segment'),
                x['from_community'], x['to_community']))
        out.append('提示：%s' % r.get('caveat', ''))
        return '\n'.join(out)
    return _py_tool('psn', sm, timeout=3600, top=top_path, traj=traj_path,
                    contact_cutoff_nm=contact_cutoff_nm, edge_freq=edge_freq, stride=stride)


@tool
def analyze_leadlag(series='', max_lag=20):
    """【先后顺序】激活坐标之间的 lead-lag：去趋势 + z 分数后的互相关取 argmax lag。

    正 lag 表示 a 领先 b（a 先动）。用来回答"谁先动"——TM6 先外移还是 DRY 先断？
    series 形如 'microswitch_PIF.csv:3.40-6.44,tm6_opening.csv:TM6ic_TM3ic_nm'。
    lag_at_boundary=true 表示最优 lag 顶在搜索窗边缘、先后顺序不可信，需加大 max_lag 复核。
    """
    def sm(r):
        out = ['%s 个点，帧间隔 %s ns，搜索窗 ±%s 帧' % (
            r.get('n_points'), r.get('dt_ns'), r.get('max_lag_frames'))]
        for x in (r.get('top_leadlag') or [])[:8]:
            flag = '  [lag 顶在边界，未辨识]' if x.get('lag_at_boundary') else ''
            out.append('    %s vs %s  corr=%s lag=%+d 帧（%s ns）领先=%s%s' % (
                str(x['a'])[:22], str(x['b'])[:22], _r3(x['corr']), x['best_lag_frames'],
                _r2(x['best_lag_ns']), x['leader'], flag))
        if r.get('n_boundary_lags'):
            out.append('注意：%s 条关系的最优 lag 顶在边界，需加大 max_lag 复核' % r['n_boundary_lags'])
        return '\n'.join(out)
    return _py_tool('leadlag', sm, timeout=1800, series=series, max_lag=max_lag)


@tool
def analyze_water_network(top_path='', traj_path='', water_cutoff_nm=0.35, stride=0):
    """【水网络】核心位点的水占据率 + 配体-水-残基 水桥。出 water_occupancy.csv / water_sites.csv。

    位点取 gpcr.json 里的 BW 锚点（即微开关位点）。占据率 = 该位点最近重原子在
    water_cutoff_nm 内有水的帧比例；水桥 = 同一个水同时在配体与位点的 cutoff 内。
    需要含水子轨迹（会自动用 gmx trjconv 抽一次，比较慢）。
    """
    def sm(r):
        out = ['%s 帧；水氧 %s 个；配体原子 %s 个；cutoff=%s nm' % (
            r.get('n_frames_used'), r.get('n_water_oxygen'),
            r.get('n_ligand_atoms'), r.get('water_cutoff_nm'))]
        out.append('位点水占据（降序）：')
        for x in (r.get('sites') or [])[:12]:
            out.append('    %-7s res=%4s %-6s 平均水数=%s 占据=%s 水桥帧比=%s' % (
                x['site'], x['resnum'], x.get('segment'), _r2(x['mean_water_count']),
                _r2(x['occupancy']), _r2(x['water_bridge_fraction'])))
        out.append('提示：%s' % r.get('caveat', ''))
        return '\n'.join(out)
    return _py_tool('water_network', sm, timeout=5400, top=top_path, traj=traj_path,
                    water_cutoff_nm=water_cutoff_nm, stride=stride)


@tool
def parse_plumed(colvar='', hills=''):
    """【PLUMED 解析】读 COLVAR / HILLS：字段、时间回跳分段、每列统计、CV 探索范围。

    元动力学数据先跑这个：确认 CV 叫什么、跑到了什么范围、偏置长到多少。
    重启拼接过的 COLVAR 时间会回跳，这里按单调段切分并报 n_time_segments；
    一个目录里常有 COLVAR 与 COLVAR_<tag> 属于不同 CV，candidates 会列出来避免拿错。
    """
    def sm(r):
        out = ['COLVAR=%s  HILLS=%s' % (r.get('colvar_file'), r.get('hills_file'))]
        cv = r.get('colvar') or {}
        if cv:
            out.append('COLVAR：字段 %s；%s 点；时间回跳段数 %s；时间范围 %s' % (
                cv.get('fields'), cv.get('n_points'),
                cv.get('n_time_segments'), cv.get('time_span')))
            for k, v in (cv.get('stats') or {}).items():
                out.append('    %-12s 均值=%s 最小=%s 最大=%s 末=%s' % (
                    k, _fmtn(v['mean'], 4), _fmtn(v['min'], 4),
                    _fmtn(v['max'], 4), _fmtn(v['last'], 4)))
        bg = r.get('bias_growth')
        if bg:
            out.append('偏置增长：首 %s -> 末 %s kJ/mol（最大 %s）' % (
                _r1(bg['first_kjmol']), _r1(bg['last_kjmol']), _r1(bg['max_kjmol'])))
        hl = r.get('hills') or {}
        if hl:
            out.append('HILLS：%s 个 hill；CV 数 %s；字段 %s' % (
                hl.get('n_hills'), hl.get('n_cv'), hl.get('fields')))
            out.append('    总高度 %s kJ/mol；首 %s -> 末 %s（比值 %s）' % (
                _r1(hl.get('total_height_kjmol')), _fmtn(hl.get('first_height_kjmol'), 4),
                _fmtn(hl.get('last_height_kjmol'), 4),
                _fmtn(hl.get('height_ratio_last_first'), 4)))
            for e in (hl.get('cvs') or []):
                out.append('    CV %s：范围 %s ~ %s（探索跨度 %s）' % (
                    e['name'], _fmtn(e['min'], 4), _fmtn(e['max'], 4),
                    _fmtn(e['explored_range'], 4)))
        cands = (r.get('colvar_candidates') or []) + (r.get('hills_candidates') or [])
        if len(cands) > 1:
            out.append('注意：目录里有多个候选文件，确认选对了：%s' % [c['name'] for c in cands])
        return '\n'.join(out)
    return _py_tool('plumed_parse', sm, timeout=1800, colvar=colvar, hills=hills)


@tool
def analyze_fes(hills='', bins=150, f_max_kJmol=200):
    """【重加权 FES】plumed sum_hills 从 HILLS 出自由能面 + 分块收敛 + 盆地/势垒。

    出 plumed_fes.csv。这是**重加权** FES（well-tempered 偏置按 (T+dT)/dT 缩放），
    与 analyze_landscape 里那个轨迹 KDE 估计不是一回事。三个必须一起看的量：
      1) convergence 的 rms_centered 是否随块长下降——不降就是没收敛；
      2) basins 的 dF / 势垒——只在收敛且 F<=f_max_kJmol 的已采样窗口内才有意义；
      3) fes_profile——FES 形状本身。
    局限：单 CV 的 1D PMF 不等于结合自由能。
    """
    def sm(r):
        out = ['HILLS=%s；%s 个 hill；%s 维' % (
            r.get('hills_file'), r.get('n_hills'), r.get('dims'))]
        out.append('分块收敛（只在已采样窗口内比较；rms_centered 才反映形状差异）：')
        for c in (r.get('convergence') or []):
            if 'error' in c:
                out.append('    截到 %3.0f%% 失败：%s' % (c['fraction'] * 100, str(c['error'])[:80]))
                continue
            out.append('    截到 %3.0f%% (%s hill)：rms=%s rms_centered=%s（%s 个网格点）' % (
                c['fraction'] * 100, c['n_hills'], _r2(c.get('rms_vs_full')),
                _r2(c.get('rms_centered')), c.get('n_grid_used')))
        for k, v in (r.get('basins') or {}).items():
            out.append('    %s = %s' % (k, _r2(v) if isinstance(v, float) else v))
        out.append('FES 形状（cv, F kJ/mol）：')
        for p in (r.get('fes_profile') or []):
            out.append('    cv=%s  F=%s' % (_fmtn(p['cv1'], 4), _r2(p['F_kJmol'])))
        out.append('提示：%s' % r.get('caveat', ''))
        return '\n'.join(out)
    return _py_tool('plumed_fes', sm, timeout=3600, hills=hills,
                    bins=bins, f_max_kJmol=f_max_kJmol)


@tool
def compare_systems(ws_b, metrics='', n_block=6):
    """【A/B 对比】当前工作区(A) 与 ws_b 工作区(B) 的同一套指标做差值 + 分块 bootstrap CI。

    metrics 形如 'microswitch_DRY.csv:3.49-3.50,tm6_opening.csv:TM6ic_TM3ic_nm'，
    不给则用一套默认指标。出 compare_metrics.csv。significant = 差值 95% CI 不含 0。
    **两个体系必须等长时间尺度才可比**，否则先把两段轨迹截到相同长度；序列短于 50 帧时
    CI 会明显偏乐观（会给出警告）。ws_b 必须是绝对路径。
    """
    def sm(r):
        out = ['A=%s' % r.get('workspace_a'), 'B=%s' % r.get('workspace_b')]
        out.append('%s 项指标，其中 %s 项显著（95%% CI 不含 0）' % (
            r.get('n_metrics'), r.get('n_significant')))
        for x in (r.get('metrics') or []):
            out.append('  %-22s %-18s A=%s B=%s Δ=%+s CI=[%+s, %+s] %s' % (
                str(x['file'])[:22], str(x['column'])[:18],
                _fmtn(x['mean_a'], 4), _fmtn(x['mean_b'], 4), _fmtn(x['delta_b_minus_a'], 4),
                _fmtn(x['ci95_lo'], 4), _fmtn(x['ci95_hi'], 4),
                '显著' if x['significant'] else ''))
        for w in (r.get('warnings') or []):
            out.append('警告：%s' % w)
        if r.get('missing'):
            out.append('缺失指标：%s' % r['missing'])
        out.append('提示：%s' % r.get('caveat', ''))
        return '\n'.join(out)
    return _py_tool('compare', sm, timeout=1800, ws_b=ws_b, metrics=metrics, n_block=n_block)


@tool
def plot_gpcr_figures(families='', out_dir='figs') -> str:
    """【GPCR 分层出图】读工作区里的分析 CSV/JSON，出 Nature 风格多面板图（PDF+PNG 600dpi）。

    可用 family：segments(分段稳定性) / switches(微开关) / tm6(胞内开放) / ligand(配体保留与口袋) /
    landscape(构象景观 FES+态) / coupling(DCCM+段耦合) / psn(网络介数与社区) /
    leadlag(时序先后) / water(水网络) / fes(PLUMED 元动力学 FES+收敛) / uncertainty(CI/ESS) /
    compare(A/B 对比，要求当前工作区里已有 compare_metrics.csv)。
    families 留空 = 全部尝试；缺数据的分组会被跳过并在返回里写明原因（不算失败）。
    这是纯读图步骤，不重算任何分析；先跑对应的 analyze_* 工具。
    """
    ws = os.path.abspath(current_workspace.get())
    try:
        res = _GP.plot_all(ws, out_dir=out_dir, families=families)
    except Exception as e:
        return '【plot_gpcr_figures 失败】%s' % e
    ok = [k for k, v in res.items() if v['ok']]
    skipped = [(k, v['msg']) for k, v in res.items() if not v['ok']]
    lines = ['【GPCR 出图完成】成功 %d / 尝试 %d 组' % (len(ok), len(res))]
    for k in ok:
        lines.append('  ✓ %-12s %s' % (k, res[k]['msg']))
    for k, why in skipped:
        lines.append('  ✗ %-12s 跳过：%s' % (k, why))
    lines.append('图片目录：%s（gpcr_<family>.pdf / .png）' % out_dir)
    return '\n'.join(lines)
