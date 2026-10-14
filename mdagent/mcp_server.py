"""MD Agent MCP Server: 把 tools.py 的工具通过 MCP 暴露给 Claude Desktop / Cursor 等。

典型编排: create_task -> run_gromacs_md -> (等 .gro) -> prepare_analysis_index
+ merge_trajectories -> compute_rmsd/rmsf/gyrate/hbond -> plot_md_figures -> run_mmpbsa
"""
import os
import sys
import glob
import argparse
import datetime
import functools

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

_gmx_bin = os.getenv("GMX_BIN", "")
_gmx_dir = os.path.dirname(_gmx_bin) if os.sep in _gmx_bin else ""
if _gmx_dir:
    os.environ["PATH"] = _gmx_dir + os.pathsep + os.environ.get("PATH", "")

from tools import (
    read_file,
    check_logs,
    prepare_analysis_index,
    run_gromacs_md,
    check_md_status,
    merge_trajectories,
    compute_rmsd,
    compute_rmsf,
    compute_gyrate,
    compute_hbond,
    run_mmpbsa,
    plot_md_figures,
    diagnose_and_fix,
    auto_repair,
    search_md_docs,
    detect_receptor_profile,
    analyze_segments,
    analyze_microswitches,
    analyze_tm6_opening,
    analyze_ligand_retention,
    analyze_landscape,
    analyze_uncertainty,
    analyze_correlation,
    analyze_network,
    analyze_leadlag,
    analyze_water_network,
    parse_plumed,
    analyze_fes,
    compare_systems,
    plot_gpcr_figures,
    current_workspace,
)

from prep_tools import (
    prep_set_config,
    prep_protein_topology,
    prep_ligand_cgenff,
    prep_ligand_gaff,
    prep_assemble,
    prep_build_box,
    prep_solvate,
    prep_ionize,
    prep_grompp,
    prep_review,
    prep_orient_protein,
    prep_build_membrane,
)

from decision_tools import (
    decide_protonation_scan,
    decide_apply_protonation,
    decide_charge_audit,
    decide_ligand_params,
    decide_membrane_check,
    decide_dissociation_check,
    decide_review,
)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "md-agent",
    instructions=(
        "GROMACS 分子动力学模拟 + MD/GPCR/元动力学 后处理分析 MCP 工具集。"
        "【体系准备】给蛋白结构（±配体）产出可直接跑的 .tpr："
        "prep_set_config 设力场/水模型/盒型/离子 -> prep_protein_topology -> "
        "prep_ligand_cgenff 或 prep_ligand_gaff -> prep_assemble -> prep_build_box -> "
        "prep_solvate -> prep_ionize -> prep_grompp -> prep_review 出审查报告。"
        "膜蛋白加 prep_orient_protein + prep_build_membrane（本地 POPC 双层，未平衡）。"
        "【判断层】准备前对输入有歧义处做判定并记账：decide_protonation_scan（质子化态与二硫键，"
        "会写 protein_decided.pdb 供 pdb2gmx 使用）+ decide_apply_protonation -> "
        "decide_ligand_params（CGenFF 罚分）-> decide_charge_audit（逐原子复算净电荷）-> "
        "decide_membrane_check（膜几何）；跑完后用 decide_dissociation_check 判配体是真实解离还是力场假象；"
        "最后 decide_review 出 summary/decisions.md。每条判定都带依据与反证条件；"
        "返回以「【需要你定夺】」开头表示 agent 判不了，必须问用户。"
        "【基础流程】create_task 建任务 -> run_gromacs_md 后台提交(异步) -> 等 .gro 出现 -> "
        "prepare_analysis_index + merge_trajectories -> compute_rmsd/rmsf/gyrate/hbond -> "
        "plot_md_figures 出图 -> run_mmpbsa 算 MM/GBSA 结合能。"
        "【GPCR 机制分析】必须先 detect_receptor_profile 建立 BW 通用编号(gpcr.json)，再按层调用："
        "第2层激活坐标 analyze_segments / analyze_microswitches / analyze_tm6_opening / analyze_ligand_retention；"
        "第3层构象景观 analyze_landscape / analyze_uncertainty；"
        "第4层耦合机制 analyze_correlation / analyze_network / analyze_leadlag / analyze_water_network；"
        "第5层元动力学 parse_plumed / analyze_fes；两体系对比 compare_systems；"
        "分析齐后用 plot_gpcr_figures 统一出 Nature 风格图（缺数据的分组会自动跳过）。"
        "【硬性要求】报任何均值前先跑 analyze_uncertainty 看 ESS；FES 结论必须先确认 convergence "
        "的 rms_centered 是否趋稳；位点缺失会进 missing_measurements，不要编造残基；"
        "跨体系对比前确认两侧 gpcr.json 的 BW 锚点一致。"
        "模拟与分析均为后台提交、立即返回：用 list_tasks 查产物状态、check_md_status 查进度、check_logs 查日志。"
    ),
)

_ACTIVE_TASK = {"task_id": None}


def _apply_workspace():
    task_id = _ACTIVE_TASK["task_id"]
    current_workspace.set(os.path.join("output", task_id) if task_id else ".")


def _wrap_tool(lc_tool):
    fn = lc_tool.func
    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        _apply_workspace()
        return fn(*args, **kwargs)
    return _wrapped


_ATOMIC_TOOLS = [
    read_file,
    check_logs,
    prepare_analysis_index,
    run_gromacs_md,
    check_md_status,
    merge_trajectories,
    compute_rmsd,
    compute_rmsf,
    compute_gyrate,
    compute_hbond,
    run_mmpbsa,
    plot_md_figures,
    diagnose_and_fix,
    auto_repair,
    search_md_docs,
    detect_receptor_profile,
    analyze_segments,
    analyze_microswitches,
    analyze_tm6_opening,
    analyze_ligand_retention,
    analyze_landscape,
    analyze_uncertainty,
    analyze_correlation,
    analyze_network,
    analyze_leadlag,
    analyze_water_network,
    parse_plumed,
    analyze_fes,
    compare_systems,
    plot_gpcr_figures,
    prep_set_config,
    prep_protein_topology,
    prep_ligand_cgenff,
    prep_ligand_gaff,
    prep_assemble,
    prep_build_box,
    prep_solvate,
    prep_ionize,
    prep_grompp,
    prep_review,
    prep_orient_protein,
    prep_build_membrane,
    decide_protonation_scan,
    decide_apply_protonation,
    decide_charge_audit,
    decide_ligand_params,
    decide_membrane_check,
    decide_dissociation_check,
    decide_review,
]

for _t in _ATOMIC_TOOLS:
    mcp.tool()(_wrap_tool(_t))


@mcp.tool()
def create_task(tpr_path, nsteps=0, run_name="md", traj_list="", start_times="", task_label=""):
    """创建新任务工作区并设为当前工作区。tpr_path 为起始 .tpr 的路径。
    
    Args:
    tpr_path: 起始 .tpr 路径(必填)
    nsteps: 覆盖步数(0=用 tpr 原值)
    run_name: 输出前缀
    traj_list: 分析/合并用的轨迹(.xtc, 逗号分隔)
    start_times: 分段轨迹起点时间(ps, 逗号分隔)
    task_label: 任务名后缀
    """
    label = (task_label or "").strip() or run_name or "md"
    task_id = f"task_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_{label}"
    workspace_dir = os.path.join(PROJECT_ROOT, "output", task_id)
    os.makedirs(workspace_dir, exist_ok=True)
    _ACTIVE_TASK["task_id"] = task_id
    _apply_workspace()
    return (
        f"任务已创建: {task_id}\n"
        f"工作区: output/{task_id}\n"
        f"tpr: {tpr_path}\n"
        "下一步: run_gromacs_md -> 等 .gro -> prepare_analysis_index + merge_trajectories -> 分析 -> 出图"
    )


@mcp.tool()
def list_tasks():
    """列出 output/ 下所有任务及各阶段产物状态(tpr/gro/xtc/ndx/xvg/figs/mmpbsa)。"""
    base = os.path.join(PROJECT_ROOT, "output")
    rows = []
    counter = lambda d, *pats: sum(len(glob.glob(os.path.join(d, p))) for p in pats)
    if os.path.isdir(base):
        for task_id in sorted(os.listdir(base), reverse=True):
            task_dir = os.path.join(base, task_id)
            if not os.path.isdir(task_dir):
                continue
            status = (
                f"tpr={counter(task_dir, '*.tpr')} | gro={counter(task_dir, '*.gro')} | "
                f"xtc={counter(task_dir, '*.xtc')} | ndx={counter(task_dir, 'analysis.ndx')} | "
                f"xvg={counter(task_dir, '*.xvg')} | figs={counter(task_dir, 'figs/*.png')} | "
                f"mmpbsa={counter(task_dir, 'FINAL_RESULTS_MMPBSA.dat')}"
            )
            active = "  <- 当前" if task_id == _ACTIVE_TASK["task_id"] else ""
            rows.append(f"{task_id}{active}\n    {status}")
    if not rows:
        return "output/ 下还没有任务。先调用 create_task。"
    return f"共 {len(rows)} 个任务(新->旧):\n" + "\n".join(rows)


@mcp.tool()
def set_workspace(task_id):
    """把后续工具工作区切换到 output/<task_id>；传空字符串则重置为项目根目录。"""
    task_id = (task_id or "").strip()
    if task_id and not os.path.isdir(os.path.join(PROJECT_ROOT, "output", task_id)):
        return f"任务目录 output/{task_id} 不存在，请用 list_tasks 查看。"
    _ACTIVE_TASK["task_id"] = task_id or None
    _apply_workspace()
    return f"工作区已切换到 output/{task_id}" if task_id else "工作区已重置为项目根目录。"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MD Agent MCP Server: 暴露 GROMACS/MD 工具 + 3 个管理工具")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        path = "/sse" if args.transport == "sse" else "/mcp"
        print(f"MD Agent MCP Server ({args.transport}) -> http://{args.host}:{args.port}{path}", file=sys.stderr)
        mcp.run(transport=args.transport)
