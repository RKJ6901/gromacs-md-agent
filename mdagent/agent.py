"""MD Agent：GROMACS 分子动力学模拟 + MD 轨迹后处理分析的 LangGraph 编排层。

骨架迁移自 rosetta-agent / boltz2-agent：
  router →(pipeline)→ 四步子图(建MD→等.gro→index/合并→RMSD/RMSF/Rg/Hbond→出图) → parse_results → summarize
         →(reply)→ 文档/对话分支
"""
import os
import glob
import json
import shutil
import zipfile
import datetime
import traceback
import time
from dotenv import load_dotenv

from tools import current_workspace, get_workspace_path

load_dotenv()

from typing import Annotated, Literal, TypedDict, Callable

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import gradio as gr

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
    decide_charge_audit,
    decide_ligand_params,
    decide_membrane_check,
    decide_dissociation_check,
    decide_apply_protonation,
    decide_review,
    DECISION_TOOLS,
)
import decision_defs as D

# ==========================================
# 1. 基础配置与模型初始化
# ==========================================
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL", "https://api.shubiaobiao.cn/v1")

if not api_key:
    raise ValueError("没有找到 LLM_API_KEY，请检查 .env 文件")

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "grok-4-1-fast-non-reasoning"),
    api_key=api_key,
    base_url=base_url,
    temperature=0,
)

# ==========================================
# 2. 工具分组：把"绝对禁止调用"从文本约束变成 schema 层的物理约束
# ==========================================
GENERAL_TOOLS = [search_md_docs, read_file]
REPAIR_TOOLS = [check_logs, diagnose_and_fix, auto_repair]

STEP1_TOOLS = [run_gromacs_md]
STEP2_TOOLS = [prepare_analysis_index, merge_trajectories]
STEP3_TOOLS = [compute_rmsd, compute_rmsf, compute_gyrate, compute_hbond]
STEP4_TOOLS = [plot_md_figures]

# GPCR 分层分析（必须先编号识别，再逐层调用；第2层依赖 gpcr.json 的 BW 锚点）
GPCR_SETUP_TOOLS = [detect_receptor_profile]
GPCR_L2_TOOLS = [analyze_segments, analyze_microswitches,
                 analyze_tm6_opening, analyze_ligand_retention]
GPCR_L3_TOOLS = [analyze_landscape, analyze_uncertainty]
GPCR_L4_TOOLS = [analyze_correlation, analyze_network,
                 analyze_leadlag, analyze_water_network]
GPCR_L5_TOOLS = [parse_plumed, analyze_fes]
GPCR_PLOT_TOOLS = [plot_gpcr_figures]

# 体系准备：从蛋白/配体结构造出可运行的 .tpr。
# 这一串全部是确定性节点（顺序固定、参数从工作区的 prep_config.json 读），
# 没有任何需要 LLM 判断的分支 —— 与 GPCR 分层同样的理由，交给 LLM 只会引入
# "静默什么都不做" 的失败模式。唯一的岔路是 CGenFF 缺 .str 时必须停下来等人工。
PREP_TOOLS = [prep_protein_topology, prep_ligand_cgenff, prep_ligand_gaff,
              prep_assemble, prep_build_box, prep_solvate, prep_ionize,
              prep_grompp, prep_review]
MAX_RETRIES = 2  # 单个 step 最多重试次数

# ==========================================
# 3. 任务感知工具节点（保持线程隔离行为）
# ==========================================
MAX_TOOL_RESULT_CHARS = 16000


def _truncate_tool_result(content: str, max_chars: int) -> str:
    """超过 max_chars 时保留首尾各一半，中间用省略号标记"""
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return f"{content[:half]}\n\n... [已截断 {len(content) - max_chars} 字符] ...\n\n{content[-half:]}"


class TaskAwareToolNode(ToolNode):
    """继承 ToolNode：拦截执行，读取 thread_id(即 task_id) 注入工作区上下文；
    并对超长 tool 结果做首尾截断，避免炸 context window"""

    def invoke(self, input, config=None, **kwargs):
        if config and "configurable" in config:
            task_id = config["configurable"].get("thread_id", ".")
            current_workspace.set(os.path.join("output", task_id))
        result = super().invoke(input, config=config, **kwargs)
        self._truncate_messages(result)
        return result

    async def ainvoke(self, input, config=None, **kwargs):
        if config and "configurable" in config:
            task_id = config["configurable"].get("thread_id", ".")
            current_workspace.set(os.path.join("output", task_id))
        result = await super().ainvoke(input, config=config, **kwargs)
        self._truncate_messages(result)
        return result

    @staticmethod
    def _truncate_messages(result):
        for m in result.get("messages", []):
            if isinstance(m, ToolMessage):
                content = str(m.content)
                if len(content) > MAX_TOOL_RESULT_CHARS:
                    m.content = _truncate_tool_result(content, MAX_TOOL_RESULT_CHARS)


# ==========================================
# 4. 主图状态定义
# ==========================================
class AgentState(TypedDict):
    """主图与子图共享的状态

    - messages       : 对话历史（add_messages 自动合并）
    - intent         : router 分类：pipeline / query / chat
    - pipeline_step  : 当前阶段（0=未启动, 1=建MD, 2=index/合并, 3=分析, 4=出图, 5=轮询）
    - failed_step    : 最近失败步骤编号，供 repair 回路决定回哪
    - retry_count    : 当前失败步骤已重试次数，达到 MAX_RETRIES 强制退出
    - progress_polls : progress_poll 自循环计数（达到 MAX_PROGRESS_POLLS 强制退出）
    - progress_done  : progress_poll 检测到 .gro 完成的标记
    - mode           : 流水线模式 md（模拟+基础分析）/ gpcr（只做 GPCR 分层分析）/ full（两者都做）
    - ws_b           : A/B 对比的第二个工作区（绝对路径），空则跳过对比步骤
    """
    messages: Annotated[list, add_messages]
    intent: str
    pipeline_step: int
    failed_step: int
    retry_count: int
    progress_polls: int
    progress_done: bool
    mode: str
    ws_b: str


# ==========================================
# 5. 公共辅助函数
# ==========================================
def _is_tool_error(msg) -> bool:
    """统一判定某条 ToolMessage 是否为报错"""
    if not isinstance(msg, ToolMessage):
        return False
    content = str(msg.content)
    return ("【执行报错】" in content) or ("【系统异常】" in content) or ("【执行失败】" in content)


def _llm_has_tool_call(state: AgentState) -> Literal["tool", "__next__"]:
    """通用 LLM 节点出口：必须产生 tool_call，否则视为异常直接退出"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool"
    return "__next__"


# ==========================================
# 6. 节点工厂：每个 step 的 LLM / Tool 节点由 schema 层硬约束权限
# ==========================================
def _make_step_llm_node(step_num: int, system_prompt: str, bound_tools):
    """工厂：生成 stepX_llm 节点。该 LLM 已 bind_tools(bound_tools)，只能调用本步允许的工具。"""
    step_llm = llm.bind_tools(bound_tools)

    def _node(state: AgentState):
        response = step_llm.invoke(
            [SystemMessage(content=system_prompt)] + list(state["messages"])
        )
        return {"messages": [response], "pipeline_step": step_num}

    _node.__name__ = f"step{step_num}_llm_node"
    return _node


def _make_step_tool_node(step_num: int, bound_tools):
    """工厂：生成 stepX_tool 节点。成功不改 pipeline_step；失败写 failed_step/retry_count。"""
    inner = TaskAwareToolNode(bound_tools)

    def _node(state: AgentState, config=None, **kwargs):
        result = inner.invoke(state, config=config, **kwargs)
        new_msgs = result.get("messages", [])
        update = {"messages": new_msgs}
        last = new_msgs[-1] if new_msgs else None
        if last and _is_tool_error(last):
            update["failed_step"] = step_num
            update["retry_count"] = state.get("retry_count", 0) + 1
        return update

    _node.__name__ = f"step{step_num}_tool_node"
    return _node


def _make_det_node(name: str, tools, headline: str):
    """工厂：确定性分析节点 —— 直接按固定顺序调用工具，不经过 LLM。

    GPCR 第2-5层与出图没有需要"判断"的地方：顺序固定（先编号再分层、最后出图），
    每个工具的参数都可省（路径由工作区自动解析）。交给 LLM 反而有两种坏结果：
    实测模型会返回空 tool_calls 让整条链被跳过（本次就是这么丢掉了第2-5层），
    或者它自作主张换参数。因此这些层改为确定性节点，LLM 只留在真正需要
    理解用户输入的地方（提交 MD、检索文档、写结论）。
    """

    def _node(state: AgentState, config=None, **kwargs):
        if config and "configurable" in config:
            task_id = config["configurable"].get("thread_id", ".")
            current_workspace.set(os.path.join("output", task_id))
        msgs = [HumanMessage(content="🔬 【%s】" % headline, name="GpcrLayer")]
        for t in tools:
            try:
                out = str(t.invoke({}))
            except Exception as e:
                out = "【%s 失败】%s" % (getattr(t, "name", "?"), e)
            msgs.append(HumanMessage(content=out, name="GpcrLayer"))
        return {"messages": msgs}

    _node.__name__ = name
    return _node


def _post_tool_route(next_node: str) -> Callable[[AgentState], str]:
    """工厂：生成 step_tool 节点出口条件边。成功 -> next_node，失败 -> repair"""

    def _route(state: AgentState):
        last = state["messages"][-1]
        if _is_tool_error(last):
            return "repair"
        return next_node

    return _route


# ==========================================
# 6.5 进度轮询：等 <run_name>.gro 出现（MD 完成标志）
# ==========================================
MAX_PROGRESS_POLLS = 120
PROGRESS_INTERVAL = 30  # 秒


def _count_md_products(workspace: str):
    """统计 workspace 下 MD 产物：返回 (.gro 数, .xtc 数)"""
    gro = glob.glob(os.path.join(workspace, "**", "*.gro"), recursive=True)
    xtc = glob.glob(os.path.join(workspace, "**", "*.xtc"), recursive=True)
    return len(gro), len(xtc)


def progress_poll_node(state: AgentState, config=None, **kwargs):
    """进度轮询节点：纯确定性，不调 LLM，读目录查 gmx mdrun 后台状态。

    - 出现 .gro -> MD 完成
    - 达到 MAX_PROGRESS_POLLS 上限强制退出
    """
    if config and "configurable" in config:
        workspace = os.path.join("output", config["configurable"].get("thread_id", "."))
    else:
        workspace = current_workspace.get()

    n_gro, n_xtc = _count_md_products(workspace)
    polls = state.get("progress_polls", 0) + 1

    if n_gro > 0:
        progress_text = f"✅ MD 完成！已生成 {n_gro} 个 .gro 结构 + {n_xtc} 个 .xtc 轨迹"
        done = True
    else:
        progress_text = f"⏳ gmx mdrun 后台计算中… 第 {polls} 次轮询（等待 .gro 出现）"
        done = False

    time.sleep(PROGRESS_INTERVAL)

    msg = HumanMessage(
        content=progress_text,
        name="ProgressReporter",
        id="progress_poll_status",  # 固定 id：add_messages reducer 自动覆盖，避免累积
    )
    return {
        "messages": [msg],
        "progress_polls": polls,
        "progress_done": done,
        "pipeline_step": 5,
    }


def progress_poll_route(state: AgentState) -> Literal["progress_poll", "__end__"]:
    if state.get("progress_done"):
        return "__end__"
    if state.get("progress_polls", 0) >= MAX_PROGRESS_POLLS:
        return "__end__"
    return "progress_poll"


# ==========================================
# 7. 流水线子图：建MD → 等.gro → index/合并 → 分析 → 出图，顺序硬编码
# ==========================================
STEP1_SYSTEM = (
    "你是【MD 模拟阶段】执行 Agent。"
    "任务：调用 run_gromacs_md 后台提交 gmx mdrun。"
    "从用户消息提取 tpr 路径、步数(nsteps，可空)、run_name、ntomp(线程数，可空)，原样传给工具；"
    "未给出的参数不要臆造，用工具默认值。"
    "权限：只能调用系统绑定的工具；调用完成后直接结束本轮。"
    "若工具返回【执行报错】，系统会自动进入修复分支，不要自行尝试别的工具。"
)
STEP2_SYSTEM = (
    "你是【分析准备阶段】执行 Agent。"
    "1) prepare_analysis_index(tpr_path) 生成 analysis.ndx + groups.json（必调）；"
    "2) merge_trajectories 仅在用户给出【多段轨迹列表】时才调用，把分段轨迹拼成 merged.xtc；"
    "用户没给轨迹列表（单段生产轨迹，如 <run_name>.xtc）时不要调用 merge_trajectories。"
    "tpr 路径、轨迹列表(逗号分隔)、起点时间(逗号分隔) 从用户消息提取并原样传入。"
    "权限：只调用绑定的工具，完成后直接结束本轮；报错由系统进入修复分支。"
)
STEP3_SYSTEM = (
    "你是【分析计算阶段】执行 Agent。"
    "任务：在同一轮内并行调用 compute_rmsd、compute_rmsf、compute_gyrate、compute_hbond 四个工具，"
    "tpr_path 从用户消息提取并原样传入；traj_path 传 merged.xtc（若已合并），"
    "单段生产轨迹则留空 —— 留空会自动使用工作区内最新 .xtc，不要臆造轨迹文件名。"
    "权限：只调用绑定的工具，完成后直接结束本轮；报错由系统进入修复分支。"
)
STEP4_SYSTEM = (
    "你是【出图阶段】执行 Agent。"
    "任务：调用 plot_md_figures 出 GPCR Nature 风格图。"
    "xvg_map 传 rmsd/rmsf/gyrate/hbond 到对应 xvg 文件的映射（hbond 用 hbond_count.xvg），out_prefix 用 figs/md。"
    "权限：只调用绑定的工具，完成后直接结束本轮。"
)

# ---------- GPCR 分层分析阶段 ----------
def validate_inputs() -> str:
    """确定性校验：工作区是否存在 .tpr（坏输入不进 GPU）"""
    ws = current_workspace.get()
    tprs = glob.glob(os.path.join(ws, "*.tpr"))
    if not tprs:
        return "【输入校验失败】工作区未找到 .tpr，无法继续。"
    return f"【输入校验通过】发现 tpr: {[os.path.basename(t) for t in tprs]}"


def build_pipeline_subgraph():
    """构建四步流水线子图。

    START → step1_llm →(tool) step1_tool →(ok) validate → progress_poll
          →(done) step2_llm →(tool) step2_tool →(ok) step3_llm →(tool) step3_tool
          →(ok) step4_llm →(tool) step4_tool →(ok) END
    任一 tool 报错 → repair；repair 后按 failed_step 回到对应 step。
    """
    sg = StateGraph(AgentState)

    sg.add_node("step1_llm", _make_step_llm_node(1, STEP1_SYSTEM, STEP1_TOOLS))
    sg.add_node("step1_tool", _make_step_tool_node(1, STEP1_TOOLS))
    sg.add_node("step2_llm", _make_step_llm_node(2, STEP2_SYSTEM, STEP2_TOOLS))
    sg.add_node("step2_tool", _make_step_tool_node(2, STEP2_TOOLS))
    sg.add_node("step3_llm", _make_step_llm_node(3, STEP3_SYSTEM, STEP3_TOOLS))
    sg.add_node("step3_tool", _make_step_tool_node(3, STEP3_TOOLS))
    sg.add_node("step4_llm", _make_step_llm_node(4, STEP4_SYSTEM, STEP4_TOOLS))
    sg.add_node("step4_tool", _make_step_tool_node(4, STEP4_TOOLS))
    sg.add_node("gpcr_setup", _make_det_node("gpcr_setup", GPCR_SETUP_TOOLS, "受体编号识别"))
    sg.add_node("gpcr_l2", _make_det_node("gpcr_l2", GPCR_L2_TOOLS, "第2层 激活坐标"))
    sg.add_node("gpcr_l3", _make_det_node("gpcr_l3", GPCR_L3_TOOLS, "第3层 构象景观"))
    sg.add_node("gpcr_l4", _make_det_node("gpcr_l4", GPCR_L4_TOOLS, "第4层 耦合机制"))
    sg.add_node("gpcr_l5", _make_det_node("gpcr_l5", GPCR_L5_TOOLS, "第5层 元动力学"))
    sg.add_node("gpcr_plot", _make_det_node("gpcr_plot", GPCR_PLOT_TOOLS, "GPCR 出图"))

    def gpcr_compare_node(state: AgentState, config=None, **kwargs):
        """A/B 对比：只有用户给了对比工作区才跑，否则明确记一句跳过。

        跳过时必须清掉旧对比产物：工作区常被 cp -r 复用，残留的
        analysis_compare.json / compare_metrics.csv 会被报告当成"本次实测"读出，
        与"本步跳过"自相矛盾（实测撞到过：一组旧 TM6 值被当成本次结果）。
        """
        if config and "configurable" in config:
            current_workspace.set(os.path.join("output", config["configurable"].get("thread_id", ".")))
        wsb = str(state.get("ws_b") or "").strip()
        if not wsb:
            note = "🔬 【A/B 对比】用户未提供对比工作区（ws_b），本步跳过。"
            note += _clear_compare_artifacts(current_workspace.get())
            return {"messages": [HumanMessage(content=note, name="GpcrLayer")]}
        try:
            out = str(compare_systems.invoke({"ws_b": wsb}))
        except Exception as e:
            out = "【compare_systems 失败】%s" % e
        return {"messages": [HumanMessage(content=out, name="GpcrLayer")]}

    gpcr_compare_node.__name__ = "gpcr_compare"
    sg.add_node("gpcr_compare", gpcr_compare_node)

    def validate_node(state: AgentState, config=None, **kwargs):
        try:
            report = validate_inputs()
        except Exception as e:
            report = f"【输入校验】执行出错: {e}"
        return {"messages": [HumanMessage(content=f"🧪 {report}", name="InputValidator")]}

    validate_node.__name__ = "validate_node"
    sg.add_node("validate", validate_node)

    # repair 分支
    repair_llm = llm.bind_tools(REPAIR_TOOLS)

    def repair_node(state: AgentState):
        sys_prompt = (
            "你是【自动修复 Agent】。当前流水线某一步刚报错，你的任务是："
            "调用合适的诊断/修复工具（check_logs / diagnose_and_fix / auto_repair）定位问题，然后直接给出修复建议。"
            "严禁重复触发同一报错工具；如确实无法自动恢复，请明确告知放弃。"
        )
        response = repair_llm.invoke([SystemMessage(content=sys_prompt)] + list(state["messages"]))
        return {"messages": [response]}

    repair_node.__name__ = "repair_node"

    def repair_route(state: AgentState) -> Literal["repair_tools", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "repair_tools"
        return "__end__"

    def repair_back_route(state: AgentState):
        if state.get("retry_count", 0) > MAX_RETRIES:
            return "__end__"
        return {
            1: "step1_llm",
            2: "step2_llm",
            3: "step3_llm",
            4: "step4_llm",
        }.get(state.get("failed_step", 1), "__end__")

    sg.add_node("repair", repair_node)
    sg.add_node("repair_tools", TaskAwareToolNode(REPAIR_TOOLS))
    sg.add_node("progress_poll", progress_poll_node)

    def _entry_route(state: AgentState):
        """gpcr 模式（在已有轨迹上做分析）跳过 MD 四步，直接从编号识别进入。"""
        if state.get("mode") == "gpcr":
            return "gpcr_setup"
        return "step1_llm"

    sg.add_conditional_edges(
        START,
        _entry_route,
        {"step1_llm": "step1_llm", "gpcr_setup": "gpcr_setup"},
    )
    sg.add_conditional_edges(
        "step1_llm",
        _llm_has_tool_call,
        {"tool": "step1_tool", "__next__": "__end__"},
    )
    sg.add_conditional_edges(
        "step1_tool",
        _post_tool_route("validate"),
        {"validate": "validate", "repair": "repair"},
    )
    sg.add_edge("validate", "progress_poll")
    sg.add_conditional_edges(
        "progress_poll",
        progress_poll_route,
        {"progress_poll": "progress_poll", "__end__": "step2_llm"},
    )

    sg.add_conditional_edges(
        "step2_llm",
        _llm_has_tool_call,
        {"tool": "step2_tool", "__next__": "__end__"},
    )
    sg.add_conditional_edges(
        "step2_tool",
        _post_tool_route("step3_llm"),
        {"step3_llm": "step3_llm", "repair": "repair"},
    )
    sg.add_conditional_edges(
        "step3_llm",
        _llm_has_tool_call,
        {"tool": "step3_tool", "__next__": "__end__"},
    )
    sg.add_conditional_edges(
        "step3_tool",
        _post_tool_route("step4_llm"),
        {"step4_llm": "step4_llm", "repair": "repair"},
    )
    sg.add_conditional_edges(
        "step4_llm",
        _llm_has_tool_call,
        {"tool": "step4_tool", "__next__": "__end__"},
    )
    def _route_after_md(state: AgentState):
        """MD 四步跑完后：报错进 repair；md 模式收工；full 模式继续走 GPCR 分层分析。"""
        if _is_tool_error(state["messages"][-1]):
            return "repair"
        if state.get("mode") == "full":
            return "gpcr_setup"
        return "__end__"

    sg.add_conditional_edges(
        "step4_tool",
        _route_after_md,
        {"repair": "repair", "gpcr_setup": "gpcr_setup", "__end__": "__end__"},
    )

    # GPCR 分层链：全部确定性，顺序固定。任一层内部工具失败只记在消息里，
    # 不会中断链条 —— 后面 hpcr_report 会把这些失败如实写进"边界"一节。
    _chain = ["gpcr_setup", "gpcr_l2", "gpcr_l3", "gpcr_l4", "gpcr_l5",
              "gpcr_plot", "gpcr_compare"]
    sg.add_node("decide_run", _make_det_node("decide_run", [decide_dissociation_check], "配体解离判定"))
    for _a, _b in zip(_chain, _chain[1:]):
        sg.add_edge(_a, _b)
    sg.add_edge("gpcr_compare", "decide_run")
    sg.add_edge("decide_run", "__end__")

    sg.add_conditional_edges(
        "repair",
        repair_route,
        {"repair_tools": "repair_tools", "__end__": "__end__"},
    )
    sg.add_conditional_edges(
        "repair_tools",
        repair_back_route,
        {
            "step1_llm": "step1_llm",
            "step2_llm": "step2_llm",
            "step3_llm": "step3_llm",
            "step4_llm": "step4_llm",
            "__end__": "__end__",
        },
    )

    return sg.compile()


# ==========================================
# 8. 主图节点：router / reply / parse_results / summarize
# ==========================================
INTENT_SYSTEM = (
    "你是任务意图路由器。根据用户最新输入，只回复以下三种之一，不要输出其他内容："
    "  pipeline —— 用户要求执行计算任务：准备蛋白/配体体系、跑 GROMACS MD 模拟、"
    "或对已有轨迹做后处理分析"
    "  query    —— 用户在询问 GROMACS/MD 文档、命令、参数或报错含义（知识检索）"
    "  chat     —— 问候、寒暄，或询问\"你能做什么\"这类与本系统能力有关的话题"
)

# router 节点：意图分类（pipeline / query / chat）+ 流水线模式（md / gpcr / full）
MD_MARKERS = ["tpr", "step5_1", ".gro", "mdrun", "md 模拟", "分子动力学", "轨迹",
              "rmsd", "run_name", "建md", "跑模拟", "模拟"]
GPCR_MARKERS = ["gpcr", "元动力学", "metadynamics", "plumed", "colvar", "hills",
                "微开关", "microswitch", "tm6", "fes", "dccm", "psn", "lead-lag",
                "leadlag", "自由能面", "景观", "landscape", "水网络", "水桥",
                "配体保留", "口袋", "ballesteros", "bw 编号", "激活坐标",
                "apelin", "aplnr", "7w0m", "8xzi", "分段稳定性", "构象景观", "耦合机制"]


PREP_MARKERS = ["准备体系", "建体系", "准备复合物", "pdb2gmx", "力场",
                "水模型", "溶剂化", "建膜", "生成tpr", "生成 tpr", "准备蛋白",
                "质子化", "二硫键", "净电荷", "判断", "审查"]


def _detect_mode(text) -> str:
    t = str(text).lower()
    if any(m in t for m in PREP_MARKERS):
        return "prepare"
    if ("只做 gpcr" in t) or ("只做gpcr" in t) or ("不跑 md" in t) or ("不跑md" in t):
        return "gpcr"
    if ("全流程" in t) or ("模拟+gpcr" in t) or ("模拟和 gpcr" in t):
        return "full"
    md = any(m in t for m in MD_MARKERS)
    gp = any(m in t for m in GPCR_MARKERS)
    if md and gp:
        return "full"
    if gp:
        return "gpcr"
    return "md" if md else ""


def _extract_ws_b(text) -> str:
    """从用户消息里取"对比工作区: /path"（跨体系 A/B 对比的目标工作区）。"""
    s = str(text)
    for marker in ("对比工作区:", "对比工作区：", "ws_b:", "ws_b："):
        i = s.find(marker)
        if i >= 0:
            rest = s[i + len(marker):].strip()
            for sep in ("|", ",", "\n"):
                if sep in rest:
                    rest = rest.split(sep)[0]
            return rest.strip()
    return ""


def router_node(state: AgentState) -> AgentState:
    if state.get("intent"):
        return {}

    msgs = state["messages"]
    user_msgs = [m.content for m in msgs if isinstance(m, HumanMessage)]
    last_user = user_msgs[-1] if user_msgs else ""

    mode = _detect_mode(last_user)
    if mode:
        return {"intent": "pipeline", "mode": mode, "ws_b": _extract_ws_b(last_user)}

    try:
        resp = llm.invoke([
            SystemMessage(content=INTENT_SYSTEM),
            HumanMessage(content=str(last_user)),
        ])
        intent = (resp.content or "").strip().lower()
    except Exception:
        intent = "chat"

    if intent not in ("pipeline", "query", "chat"):
        intent = "chat"
    return {"intent": intent, "mode": "md", "ws_b": ""}


# router 后的主图分支：pipeline 走子图，其余走 reply
def main_route_after_router(state: AgentState):
    if state.get("mode") == "prepare":
        return "prep"
    return "pipeline" if state.get("intent") == "pipeline" else "reply"


REPLY_SYSTEM = (
    "你是 GROMACS/MD 知识助手。该系统能做五件事，回答\"你能做什么\"时按这五条说："
    "① 体系准备：给蛋白结构（±配体）产出可直接跑的 .tpr，力场/水模型/盒型/离子可选，膜蛋白可本地插 POPC 双层；"
    "② MD 模拟与基础分析（RMSD/RMSF/Rg/氢键/出图）；"
    "③ 已有轨迹的 GPCR 分层分析（激活坐标/构象景观/耦合机制/元动力学 FES）；"
    "④ GROMACS 文档与报错检索。"
    "⑤ 判断层 —— 在体系准备前对质子化态与二硫键、净电荷、配体参数来源、膜质量做判断并记账，"
    "跑完后对配体是否真的离开口袋做判断；拿不准的会停下来问你。"
    "若用户询问文档或报错，可调用 search_md_docs / read_file；属于寒暄就直接回复。"
    "给出最终答复后不要再调用工具。"
)
reply_llm = llm.bind_tools(GENERAL_TOOLS)


def reply_node(state: AgentState):
    response = reply_llm.invoke([
        SystemMessage(content=REPLY_SYSTEM),
        *state["messages"],
    ])
    return {"messages": [response]}


def reply_route(state: AgentState) -> Literal["reply_tools", "summarize"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "reply_tools"
    return "summarize"


reply_tools_node = TaskAwareToolNode(GENERAL_TOOLS)


SUMMARIZE_SYSTEM = (
    "你是【统一收尾 Agent】。基于历史消息向用户输出最终答复，中文，禁止再调用任何工具，禁止追问。\n"
    "【若历史消息里有 '📐 【GPCR 结论骨架'】必须严格按它的五节组织答复，一节都不能少：\n"
    "  一、观察 —— 只陈述测到了什么；\n"
    "  二、统计 —— 给出 CI/ESS 与 FES 收敛判定，说明结论有多硬；\n"
    "  三、机制含义 —— 按 Class A GPCR 判据解读（这是可检验的假设，不要说成定论）；\n"
    "  四、边界 —— 明确写出哪些结论不能下（缺失位点、ESS 过低、FES 未收敛、无 PLUMED 数据等）；\n"
    "  五、产出清单 —— 照抄骨架里磁盘实测的图/表/JSON 清单。\n"
    "  骨架里的数字与产出清单必须原样引用，**不得改动、不得补充骨架里没有的数字**；"
    "骨架里写'无法估计/缺失/未收敛'的地方，答复里也必须照样说明，不能省略成好消息。\n"
    "  **产出清单是磁盘实测结果，禁止与之相反的说法**：清单里列出了图片就不能说'出图失败/未生成图片'。\n"
    "【若历史消息里有 '📐' 以外的决策台账内容（【已判定】或【需要你定夺】开头的消息，或 summary/decisions.md），"
    "必须原样转述每一条判断的结论、依据与反证条件；"
    "**若出现 【需要你定夺】，说明流程停下来等你拍板，要把它作为最重要的一句放在最前面说清楚**，"
    "不得含糊成\"已完成\"。\n"
    "【若跑的是体系准备】说明产出的 step5_1.tpr 与 summary/prep_review.md，"
    "并**如实转述**审查报告里的兼容性告警与「这次准备不能保证什么」，"
    "不得只说成功；若流程在某一步停下（例如等人工上传 .str），要明确说出来。\n"
    "【若只跑了 MD 流水线】依次说明 MD 模拟→分析准备→RMSD/RMSF/Rg/Hbond→出图 的结果，并提示后台计算状态。\n"
    "【若属于查询/对话】用一两句话总结答案。"
)


def summarize_node(state: AgentState):
    summarize_llm = llm.bind_tools([])
    response = summarize_llm.invoke([
        SystemMessage(content=SUMMARIZE_SYSTEM),
        *state["messages"],
    ])
    return {"messages": [response]}


# ==========================================
# 9. parse_results 节点：确定性汇总 xvg 关键数值
# ==========================================
def _xvg_stats(path: str):
    """读 2 列 .xvg，返回 {n, mean, min, max, last}；无可读数据返回 None。"""
    ys = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s or s[0] in ("#", "@"):
                    continue
                try:
                    v = [float(p) for p in s.split()]
                except ValueError:
                    continue
                if len(v) < 2:
                    continue
                ys.append(v[1])
    except Exception:
        return None
    if not ys:
        return None
    return {"n": len(ys), "mean": sum(ys) / len(ys), "min": min(ys), "max": max(ys), "last": ys[-1]}


def parse_md_results() -> str:
    """汇总 rmsd/rmsf/gyrate/hbond_count 的 xvg 关键数值，落盘 summary/md_eval_report.txt。"""
    ws = current_workspace.get()
    specs = [
        ("RMSD", "rmsd.xvg", "nm"),
        ("RMSF", "rmsf.xvg", "nm"),
        ("Rg", "gyrate.xvg", "nm"),
        ("Hbond", "hbond_count.xvg", "count"),
    ]
    rows, found = [], False
    for label, fn, unit in specs:
        p = os.path.join(ws, fn)
        if not os.path.exists(p):
            continue
        st = _xvg_stats(p)
        if not st:
            continue
        found = True
        rows.append(
            f"{label:6s} n={st['n']:5d} mean={st['mean']:.3f} min={st['min']:.3f} "
            f"max={st['max']:.3f} last={st['last']:.3f} ({unit})"
        )
    if not found:
        return "【结果汇总】未找到 rmsd/rmsf/gyrate/hbond xvg，分析可能尚未完成。"
    text = "\n".join(rows)
    try:
        d = get_workspace_path("summary")
        with open(os.path.join(d, "md_eval_report.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass
    return "【结果汇总】\n" + text


def parse_results_node(state: AgentState, config=None, **kwargs):
    """结果汇总节点（确定性，不调 LLM）。"""
    try:
        report = parse_md_results()
    except Exception as e:
        report = f"【结果汇总】执行出错: {e}"
    msg = HumanMessage(content=f"📊 {report}", name="MdEvalDirector")
    return {"messages": [msg]}


# ==========================================
# 9.5 GPCR 结论骨架（确定性）：把各层分析 JSON 收成"观察/统计/机制含义/边界"四节
# ==========================================
GPCR_TASKS = ["detect_numbering", "segments", "microswitches", "tm6_opening",
              "ligand_retention", "landscape", "uncertainty", "correlation",
              "network", "leadlag", "water_network", "plumed_parse",
              "plumed_fes", "compare"]


def _load_analysis(ws, task):
    p = os.path.join(ws, "analysis_%s.json" % task)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) and d.get("ok") else None
    except Exception:
        return None


def _clear_compare_artifacts(ws):
    """跳过 A/B 对比时清掉旧产物，避免报告把上一轮的数字当成本次结果。"""
    ws = os.path.abspath(ws)
    gone = []
    for fn in ("analysis_compare.json", "compare_metrics.csv"):
        fp = os.path.join(ws, fn)
        try:
            if os.path.exists(fp):
                os.remove(fp)
                gone.append(fn)
        except OSError:
            pass
    return (" 已清除上一轮残留的对比产物：%s。" % "、".join(gone)) if gone else ""


def _g(d, *keys, default=None):
    """安全取嵌套键：_g(d, 'a', 'b') == d['a']['b']，任一层缺失返回 default。"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def _f(v, n=3, suf=""):
    try:
        return ("%." + str(n) + "f") % float(v) + suf
    except (TypeError, ValueError):
        return "-"


def _ess_of(v):
    """autocorr_ess 返回 {'tau':..,'ess':..}；这里统一取标量 ESS（取不到给 NaN）。"""
    e = (v or {}).get("ess")
    if isinstance(e, dict):
        e = e.get("ess")
    try:
        return float(e)
    except (TypeError, ValueError):
        return float("nan")


def _sec_observe(A):
    """观察：只陈述测到什么，不下结论。"""
    out = []
    d = A.get("detect_numbering")
    if d:
        p = d.get("profile") or {}
        ver = "已人工核对" if p.get("verified") else "自动识别（编号需人工确认）"
        out.append("■ 受体编号：%s，%s；残基 %s-%s；%s 帧，帧间隔 %s ns" % (
            p.get("name", "?"), ver, p.get("seq_start"), p.get("seq_end"),
            d.get("n_frames"), d.get("dt_ns")))
        an = p.get("anchors") or {}
        if an:
            out.append("  BW 锚点：" + ", ".join("%s=%s" % (k, v) for k, v in sorted(an.items())))
        for n in (p.get("notes") or [])[:3]:
            out.append("  提示：" + str(n))
        if d.get("ligand_resnames"):
            out.append("  配体残基：%s（%s 个原子）" % (
                ", ".join(d["ligand_resnames"]), d.get("n_ligand_atoms")))

    s = A.get("segments")
    if s:
        segs = s.get("segments") or {}
        ranked = [(v.get("rmsd_mean_nm"), k) for k, v in segs.items() if v.get("rmsd_mean_nm") is not None]
        if ranked:
            ranked.sort()
            out.append("■ 分段稳定性（%s 帧，跨度 %s ns，以 %s 叠合）：" % (
                s.get("n_frames_used"), s.get("time_span_ns"), s.get("fit_segments")))
            out.append("  最稳 %s=%s nm；最松 %s=%s nm" % (
                ranked[0][1], _f(ranked[0][0]), ranked[-1][1], _f(ranked[-1][0])))
        tf = (s.get("top_flexible_residues") or [])[:3]
        if tf:
            out.append("  最飘残基：" + ", ".join(
                "%s(%s)=%s nm" % (x.get("resnum"), x.get("bw") or "-", _f(x.get("rmsf_nm"))) for x in tf))

    m = A.get("microswitches")
    if m:
        for sw, ent in (m.get("switches") or {}).items():
            for lab, v in (ent.get("pairs") or {}).items():
                out.append("■ 微开关 %-4s %-10s 均值 %s nm；<0.4nm 占据 %s；<0.5nm 占据 %s" % (
                    sw, lab, _f(v.get("mean_nm")), _f(v.get("occ_0.4nm"), 2), _f(v.get("occ_0.5nm"), 2)))
            for lab, v in (ent.get("dihedrals") or {}).items():
                out.append("  %-4s %-10s 均值 %s°；rotamer 占据 %s" % (
                    sw, lab, _f(v.get("mean_deg"), 1),
                    ", ".join("%s %.2f" % (k, x) for k, x in (v.get("rotamer_occ") or {}).items()) or "-"))

    t = A.get("tm6_opening")
    if t:
        e = t.get("TM6ic_TM3ic_nm") or {}
        out.append("■ TM6 胞内端：TM6ic-TM3ic 首帧 %s → 末帧 %s nm（均值 %s，极差 %s，Δ=%s）" % (
            _f(e.get("first_nm")), _f(e.get("last_nm")), _f(e.get("mean_nm")),
            _f(e.get("range_nm")), _f(e.get("delta_nm"))))
        out.append("  残基区间：%s" % (t.get("residues") or {}))
    return out


def _sec_observe2(A):
    """观察（续）：配体 / 景观 / 耦合 / 网络 / 时序 / 水 / 元动力学 / 对比。"""
    out = []
    lg = A.get("ligand_retention")
    if lg:
        e = lg.get("ligand_rmsd_nm") or {}
        out.append("■ 配体保留（接触判据 %s nm）：配体 RMSD 均值 %s nm，末帧 %s，最大 %s" % (
            _f(lg.get("contact_cutoff_nm"), 2), _f(e.get("mean_nm")),
            _f(e.get("last_nm")), _f(e.get("max_nm"))))
        out.append("  最近重原子距离均值 %s nm；平均接触原子数 %s" % (
            _f(_g(lg, "min_dist_nm", "mean_nm")),
            _f(_g(lg, "n_contact_atoms", "mean"), 1)))
        pc = (lg.get("persistent_contacts") or [])[:8]
        if pc:
            out.append("  持续接触位点：" + ", ".join(
                "%s(%s,%s)" % (x.get("resnum"), x.get("bw") or "", _f(x.get("occupancy"), 2))
                for x in pc))

    ls = A.get("landscape")
    if ls:
        cf = ls.get("config") or {}
        st = (ls.get("states") or {}).get("occupancy") or {}
        out.append("■ 构象景观：空间 %s，方法 %s，%d 个态；tICA lag=%s 帧" % (
            cf.get("space"), cf.get("method"), int(cf.get("n_states") or 0),
            cf.get("tica_lag_frames")))
        if st:
            top = sorted(st.items(), key=lambda kv: -float(kv[1]))[:3]
            out.append("  主要态占比：" + ", ".join("S%s=%.2f" % (k, float(v)) for k, v in top))
        dw = (ls.get("states") or {}).get("dwell") or {}
        if dw:
            out.append("  平均驻留：" + ", ".join(
                "S%s=%s ns" % (k, _f((v or {}).get("mean_ns"))) for k, v in dw.items()))

    co = A.get("correlation")
    if co:
        out.append("■ 残基耦合：Cα 数 %s；平均 |DCCM|=%s；平均 MI=%s nats" % (
            co.get("n_ca"), _f(co.get("mean_abs_dccm_offdiag")),
            _f(co.get("mean_mi_nats"), 4)))
        ts = (co.get("top_segment_coupling") or [])[:3]
        if ts:
            out.append("  最强段耦合：" + ", ".join(
                "%s-%s %s" % (x.get("seg1"), x.get("seg2"), _f(x.get("mean_abs_dccm"))) for x in ts))

    nw = A.get("network")
    if nw:
        out.append("■ 结构网络 PSN：%s 节点 / %s 边；%s 个社区；%s 个残基在前后半段换社区" % (
            nw.get("n_ca"), nw.get("n_edges"), nw.get("n_communities"), nw.get("n_rewired")))
        tb = (nw.get("top_betweenness") or [])[:5]
        if tb:
            out.append("  介数枢纽：" + ", ".join(
                "%s(%s) %.1f" % (x.get("resnum"), x.get("bw") or "-", float(x.get("betweenness") or 0))
                for x in tb))

    ll = A.get("leadlag")
    if ll:
        out.append("■ 时序先后（%s 点，帧间隔 %s ns，搜索窗 %s 帧）：" % (
            ll.get("n_points"), _f(ll.get("dt_ns")), ll.get("max_lag_frames")))
        for x in (ll.get("top_leadlag") or [])[:3]:
            out.append("  %s → %s 最优 lag=%s 帧（|corr|=%s）%s" % (
                x.get("a"), x.get("b"), x.get("best_lag_frames"), _f(x.get("corr")),
                "  [顶在搜索窗边缘，未辨识]" if x.get("lag_at_boundary") else ""))

    wt = A.get("water_network")
    if wt:
        out.append("■ 水网络：%s 个水位点，%s 个水氧，截断 %s nm" % (
            wt.get("n_sites"), wt.get("n_water_oxygen"), _f(wt.get("water_cutoff_nm"), 2)))
        for x in (wt.get("sites") or [])[:5]:
            out.append("  %s%s(%s) 占据 %s，水桥比例 %s" % (
                x.get("site"), x.get("resnum"), x.get("segment"),
                _f(x.get("occupancy"), 2), _f(x.get("water_bridge_fraction"), 2)))

    pp = A.get("plumed_parse")
    if pp:
        cv = pp.get("colvar") or {}
        out.append("■ PLUMED 解析：COLVAR 字段 %s；%s 个点；时间回跳段 %s 段" % (
            cv.get("fields") or _g(pp, "colvar", "sets"), cv.get("n_points"),
            cv.get("n_time_segments")))
        hs = pp.get("hills") or {}
        if hs:
            out.append("  HILLS：%s 个 hill，%s 个 CV" % (hs.get("n_hills"), hs.get("n_cv")))

    pf = A.get("plumed_fes")
    if pf:
        b = pf.get("basins") or {}
        out.append("■ 元动力学 FES（%dD，%s 个 hill）：全局最小在 CV=%s（截断 %s kJ/mol 内）" % (
            int(pf.get("dims") or 1), pf.get("n_hills"),
            _f(b.get("cv1_at_global_min"), 2), _f(b.get("f_cut_kJmol"), 0)))
        if b.get("barrier_between_minima_kJmol") is not None:
            out.append("  次极小 CV=%s（ΔF=%s kJ/mol），双井间势垒 %s kJ/mol" % (
                _f(b.get("cv1_at_second_min"), 2), _f(b.get("dF_min2_minus_min1_kJmol"), 1),
                _f(b.get("barrier_between_minima_kJmol"), 1)))

    cp = A.get("compare")
    if cp:
        out.append("■ A/B 对比：%s 项指标，%s 项显著" % (cp.get("n_metrics"), cp.get("n_significant")))
        for x in (cp.get("metrics") or []):
            out.append("  %s:%s  A=%s B=%s Δ=%s [%s, %s] %s" % (
                x.get("file"), x.get("column"), _f(x.get("mean_a"), 4), _f(x.get("mean_b"), 4),
                _f(x.get("delta_b_minus_a"), 4), _f(x.get("ci95_lo"), 4), _f(x.get("ci95_hi"), 4),
                "显著" if x.get("significant") else ""))
    return out


def _sec_stats(A):
    """统计：所有均值必须带不确定性；FES 必须带收敛判定。没有证据就明说没有。"""
    out = []
    un = A.get("uncertainty")
    if un:
        for k, v in (un.get("series") or {}).items():
            if not isinstance(v, dict) or "mean" not in v:
                why = (v or {}).get("error") if isinstance(v, dict) else "?"
                out.append("  %-30s 无法估计：%s" % (str(k)[:30], why))
                continue
            ci = v.get("ci95") or {}
            ess = _ess_of(v)
            flag = "  ⚠ ESS<50，该均值不可信" if (ess == ess and ess < 50) else ""
            out.append("  %-30s mean=%s  95%%CI=[%s, %s]  ESS=%s%s" % (
                str(k)[:30], _f(v.get("mean"), 4), _f(ci.get("lo"), 4),
                _f(ci.get("hi"), 4), _f(ess, 1), flag))
    else:
        out.append("  未跑 analyze_uncertainty：本节所有均值都没有 CI/ESS 支撑。")

    fm = A.get("plumed_fes")
    if fm:
        vals = [e.get("rms_centered") for e in (fm.get("convergence") or [])
                if e.get("rms_centered") is not None]
        if vals:
            okv = vals[-1] <= max(1.0, 0.6 * max(vals))
            out.append("  FES 收敛 rms_centered（25/50/75%%）：%s → 判定：%s" % (
                " / ".join(_f(x, 2) for x in vals),
                "趋稳，可做定性解读" if okv else "未趋稳，不得报自由能深度与势垒"))

    for tk, lab in (("tm6_opening", "TM6ic-TM3ic"), ):
        t = A.get(tk)
        if t:
            ci = _g(t, "TM6ic_TM3ic_nm", "ci95") or {}
            if ci:
                out.append("  %s 分块 bootstrap 95%%CI=[%s, %s]" % (
                    lab, _f(ci.get("lo")), _f(ci.get("hi"))))
    lg = A.get("ligand_retention")
    if lg:
        ci = _g(lg, "ligand_rmsd_nm", "ci95") or {}
        if ci:
            out.append("  配体 RMSD 95%%CI=[%s, %s]" % (_f(ci.get("lo")), _f(ci.get("hi"))))
    return out


def _sec_mech(A):
    """机制含义：把测到的量与 Class A GPCR 的公认判据对照（可检验的解释，非定论）。"""
    out = []
    t = A.get("tm6_opening")
    d63 = _g(t, "TM6ic_TM3ic_nm", "mean_nm")
    delta = _g(t, "TM6ic_TM3ic_nm", "delta_nm")
    if d63 is not None:
        if d63 >= 1.8:
            out.append("• TM6 胞内端间隙 %.2f nm 已到激活态量级（class A 非激活约 1.0-1.3 nm，激活侧常在 1.8 nm 以上）。" % d63)
        elif d63 >= 1.3:
            out.append("• TM6 胞内端间隙 %.2f nm 处于中间区间，未充分张开，更像非激活/部分激活之间的摆动。" % d63)
        else:
            out.append("• TM6 胞内端间隙 %.2f nm 偏小，未见明显外移，本段以非激活构象为主。" % d63)
    if delta is not None and abs(float(delta)) > 0.15:
        out.append("  首末帧变化 %+.2f nm，说明该段轨迹内有单向漂移，不只是热涨落。" % float(delta))

    sw = (A.get("microswitches") or {}).get("switches") or {}
    pif = _g(sw, "PIF", "pairs", "3.40-6.44") or {}
    if pif.get("occ_0.4nm") is not None:
        o = float(pif["occ_0.4nm"])
        out.append("• PIF：x3.40-F6.44 在 0.4 nm 内的占据 %.2f。" % o + (
            "三元组基本维持，未见崩解。" if o > 0.6 else
            "占据偏低，PIF 已部分/完全崩解 —— 这是 Class A 激活最常见的早期事件之一。"))
    cw = _g(sw, "CWxP", "dihedrals", "6.48 chi1") or {}
    if cw.get("rotamer_occ"):
        ro = cw["rotamer_occ"]
        top = max(ro.items(), key=lambda kv: kv[1])
        out.append("• CWxP toggle：W6.48 chi1 圆标准差 %s°，主 rotamer %s 占 %.2f。" % (
            _f(cw.get("circ_sd_deg"), 0), top[0], top[1]) + (
            "rotamer 基本锁定，toggle 未翻转。" if top[1] > 0.7 else
            "rotamer 分布分散，toggle 在采样中发生了翻转 —— 与 TM6 张开常协同出现。"))

    dry = _g(sw, "DRY", "pairs", "3.49-3.50") or {}
    if dry.get("occ_0.4nm") is not None:
        o = float(dry["occ_0.4nm"])
        out.append("• DRY：D3.49-R3.50 在 0.4 nm 内的占据 %.2f。" % o + (
            "盐桥稳定维持。" if o > 0.6 else
            "盐桥占据偏低，DRY 已松动，提示胞内侧构象重排。"))
    out.append("（以上为按公认判据的机制解读，属于可检验的假设；结论强度受统计一节的 CI/ESS 与收敛判定约束。）")
    return out


def _sec_limits(A):
    """边界：把"没测到/不可靠/不可比"的地方明确写出来，防止下游把缺失当成结论。"""
    out = []
    miss = (A.get("microswitches") or {}).get("missing_measurements") or []
    if miss:
        out.append("• 未能测到的位点（缺失≠无变化）：%s" % ", ".join(str(x) for x in miss[:12]))
    p = _g(A, "detect_numbering", "profile") or {}
    if p and not p.get("verified"):
        out.append("• BW 编号为自动识别、未经人工核对；残基号对外引用前必须确认。")
    if not A.get("plumed_fes"):
        out.append("• 工作区没有可用的 HILLS/COLVAR，本次不含重加权自由能面；"
                   "任何关于自由能深度/势垒的结论都缺乏依据。")
    un = A.get("uncertainty")
    if un:
        bad = [k for k, v in (un.get("series") or {}).items()
               if isinstance(v, dict) and _ess_of(v) == _ess_of(v) and _ess_of(v) < 50]
        if bad:
            out.append("• 这些序列 ESS<50，均值与差值不可作为定量结论：%s" % ", ".join(
                str(x) for x in bad[:8]))
    conv = (A.get("plumed_fes") or {}).get("convergence") or []
    vals = [e.get("rms_centered") for e in conv if e.get("rms_centered") is not None]
    if vals and vals[-1] > max(1.0, 0.6 * max(vals)):
        out.append("• FES 未收敛（rms_centered 仍随块长变化），"
                   "深井与势垒都可能是采样不足的假象，只能定性看形状。")
    nb = (A.get("leadlag") or {}).get("n_boundary_lags")
    if nb:
        out.append("• lead-lag 有 %s 对的最优 lag 顶在搜索窗边缘，先后顺序未辨识，需加大 max_lag。" % nb)
    cp = A.get("compare") or {}
    for w in (cp.get("warnings") or []):
        out.append("• 对比告警：%s" % w)
    if cp.get("missing"):
        out.append("• 对比中缺失的指标：%s" % cp["missing"][:8])
    if not cp:
        out.append("• 本次未做 A/B 跨体系对比（未提供对比工作区 ws_b），没有任何跨体系差异结论；"
                   "如需对比，用 --ws-b 指定第二个工作区。")
    return out


def _sec_artifacts(ws):
    """产出清单：直接读磁盘列文件，堵死 LLM 凭空说"没出图/工具报错"的口子。"""
    out = []
    figs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ws, "figs", "gpcr_*.png")))
    if figs:
        out.append("• 已生成图 %d 张（figs/*.png）：%s" % (len(figs), ", ".join(f[:-4] for f in figs)))
    else:
        out.append("• figs/ 下没有 gpcr_*.png —— 出图确实没有产物")
    csvs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ws, "*.csv")))
    if csvs:
        head = ", ".join(csvs[:14]) + (" …（共 %d 个）" % len(csvs) if len(csvs) > 14 else "")
        out.append("• 分析表 %d 个：%s" % (len(csvs), head))
    js = glob.glob(os.path.join(ws, "analysis_*.json"))
    out.append("• 分析结果 JSON %d 个" % len(js))
    return out


def build_gpcr_report():
    ws = current_workspace.get()
    A = {}
    for t in GPCR_TASKS:
        d = _load_analysis(ws, t)
        if d:
            A[t] = d
    if not A:
        return None
    lines = ["一、观察（测到了什么）"]
    lines += (_sec_observe(A) + _sec_observe2(A)) or ["  （本工作区没有可报告的量测）"]
    lines += ["", "二、统计（结论有多硬）"]
    lines += _sec_stats(A)
    lines += ["", "三、机制含义（按 Class A GPCR 公认判据的解读）"]
    lines += _sec_mech(A)
    lines += ["", "四、边界（哪些结论不能下）"]
    lines += _sec_limits(A) or ["  （未发现需要特别声明的边界）"]
    lines += ["", "五、产出清单（磁盘实测，以此为准）"]
    lines += _sec_artifacts(ws)
    try:
        _ledger = ""
        _lp = os.path.join(ws, "decision_log.json")
        if os.path.exists(_lp):
            with open(_lp, encoding="utf-8") as _lf:
                _ldata = json.load(_lf)
            _recs = _ldata.get("decisions") if isinstance(_ldata, dict) else None
            if isinstance(_recs, list) and _recs:
                _ledger = D.render_ledger(_recs)
        if _ledger:
            lines += ["", "六、本次做了哪些判断", _ledger]
    except Exception:
        pass
    text = "\n".join(lines)
    try:
        d = get_workspace_path("summary")
        with open(os.path.join(d, "gpcr_conclusion.md"), "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass
    return text


def gpcr_report_node(state: AgentState, config=None, **kwargs):
    try:
        text = build_gpcr_report()
    except Exception as e:
        return {"messages": [HumanMessage(
            content="📐 【GPCR 结论骨架】生成出错: %s" % e, name="GpcrReport")]}
    if not text:
        return {"messages": [HumanMessage(
            content="📐 【GPCR 结论骨架】工作区里没有 GPCR 分层分析结果，跳过。",
            name="GpcrReport")]}
    return {"messages": [HumanMessage(
        content="📐 【GPCR 结论骨架，已落盘 summary/gpcr_conclusion.md】\n" + text,
        name="GpcrReport")]}


# ==========================================
# 10. 构建与编译主图
# ==========================================
def _prep_after_protein(state, config=None, **kwargs):
    # 要不要建膜由配置决定；膜蛋白才需要定向 + 插膜两步。
    if config and "configurable" in config:
        current_workspace.set(os.path.join("output", config["configurable"].get("thread_id", ".")))
    cand = os.path.join(os.path.abspath(current_workspace.get()), "prep_config.json")
    try:
        fh = open(cand, encoding="utf-8")
        membrane = bool(json.load(fh).get("membrane"))
        fh.close()
    except Exception:
        membrane = False
    return "prep_orient" if membrane else "prep_ligand"


def _prep_ligand_node(state, config=None, **kwargs):
    # 配体那一步单独写：路线不同调不同工具，且 CGenFF 缺 .str 时必须停下等人工。
    if config and "configurable" in config:
        current_workspace.set(os.path.join("output", config["configurable"].get("thread_id", ".")))
    route = ""
    cand = os.path.join(os.path.abspath(current_workspace.get()), "prep_config.json")
    try:
        fh = open(cand, encoding="utf-8")
        route = json.load(fh).get("ligand_route", "")
        fh.close()
    except Exception:
        route = ""
    tool = prep_ligand_cgenff if route == "cgenff" else prep_ligand_gaff
    try:
        out = str(tool.invoke({}))
    except Exception as e:
        out = "【%s 失败】%s" % (getattr(tool, "name", "?"), e)
    return {"messages": [HumanMessage(content=out, name="PrepStage")]}



def _prep_ligand_route(state):
    # 只有 CGenFF 缺 .str（需要人工上传）时才停链；没配体走的是【跳过配体】，
    # 那属于纯蛋白体系，直接继续到下一步组装。
    body = str(getattr(state["messages"][-1], "content", ""))
    if "需要人工一步" in body:
        return "__end__"
    return "prep_assembly"


def _decide_route(state):
    # 判定层只要有一处需要人拍板，整条链就停下，不带着悬而未决的量继续往下跑。
    body = str(getattr(state["messages"][-1], "content", ""))
    if "【需要你定夺】" in body:
        return "__end__"
    return "continue"


def _prep_decide_node(state, config=None, **kwargs):
    # 质子化判定：先扫几何拿证据，判定通过才把结论写回 PDB。
    # 扫的那个工具一旦判成「需要你定夺」，就不再执行写回，直接停。
    if config and "configurable" in config:
        current_workspace.set(os.path.join("output", config["configurable"].get("thread_id", ".")))
    parts = []
    for t in (decide_protonation_scan, decide_apply_protonation):
        if parts and "【需要你定夺】" in parts[-1]:
            break
        try:
            out = str(t.invoke({}))
        except Exception as e:
            out = "【%s 失败】%s" % (getattr(t, "name", "?"), e)
        parts.append(out)
    return {"messages": [HumanMessage(content="\n\n".join(parts), name="PrepStage")]}


def build_prep_subgraph():
    # 体系准备子图：判定质子化/二硫键 → pdb2gmx → 配体 → 判定配体参数 → 组装 → 建盒
    # → 溶剂化 → 加离子 → 判定净电荷 → grompp → 判定膜质量 → 审查报告(含决策台账)。
    # 全部确定性节点：顺序固定、参数从工作区的 prep_config.json 读。理由同 GPCR 分层：
    # 交给 LLM 只会引入"静默什么都不做"的失败模式。
    sg = StateGraph(AgentState)
    sg.add_node("prep_protein", _make_det_node("prep_protein", [prep_protein_topology], "蛋白拓扑"))
    sg.add_node("prep_ligand", _prep_ligand_node)
    sg.add_node("prep_orient", _make_det_node("prep_orient", [prep_orient_protein], "蛋白定向"))
    sg.add_node("prep_membrane", _make_det_node("prep_membrane", [prep_build_membrane], "建膜"))
    sg.add_node("prep_decide", _prep_decide_node)
    sg.add_node("prep_decide_lig", _make_det_node("prep_decide_lig", [decide_ligand_params], "判定配体参数"))
    sg.add_node("prep_decide_charge", _make_det_node("prep_decide_charge", [decide_charge_audit], "判定净电荷"))
    sg.add_node("prep_decide_memb", _make_det_node("prep_decide_memb", [decide_membrane_check], "判定膜质量"))
    sg.add_node("prep_assembly", _make_det_node("prep_assembly", [prep_assemble], "组装拓扑"))
    sg.add_node("prep_box", _make_det_node("prep_box", [prep_build_box], "建盒"))
    sg.add_node("prep_solvation", _make_det_node("prep_solvation", [prep_solvate], "溶剂化"))
    sg.add_node("prep_ions", _make_det_node("prep_ions", [prep_ionize], "加离子"))
    sg.add_node("prep_tpr", _make_det_node("prep_tpr", [prep_grompp], "生成 tpr"))
    sg.add_node("prep_check", _make_det_node("prep_check", [prep_review, decide_review], "生成审查报告"))

    _e = [('prep_assembly', 'prep_box'), ('prep_box', 'prep_solvation')]

    _e += [('prep_solvation', 'prep_ions'), ('prep_ions', 'prep_decide_charge'),
           ('prep_tpr', 'prep_decide_memb'), ('prep_check', '__end__')]
    _e += [('prep_orient', 'prep_decide'), ('prep_decide', 'prep_protein'),
           ('prep_membrane', 'prep_ligand')]
    for _a, _b in _e:
        sg.add_edge(_a, _b)
    # 膜体系必须先定向再跑 pdb2gmx：pdb2gmx 不平移坐标，若先建拓扑，
    # 蛋白会停在原始位置而脂质围着原点铺，两者错开（实测差了 13 nm）。
    sg.add_conditional_edges(START, _prep_after_protein,
        {'prep_orient': 'prep_orient', 'prep_ligand': 'prep_decide'})
    sg.add_conditional_edges('prep_protein', _prep_after_protein,
        {'prep_orient': 'prep_membrane', 'prep_ligand': 'prep_ligand'})
    sg.add_conditional_edges('prep_ligand', _prep_ligand_route, {'prep_assembly': 'prep_decide_lig', '__end__': '__end__'})
    sg.add_conditional_edges('prep_decide_lig', _decide_route, {'continue': 'prep_assembly', '__end__': '__end__'})
    sg.add_conditional_edges('prep_decide_charge', _decide_route, {'continue': 'prep_tpr', '__end__': '__end__'})
    sg.add_conditional_edges('prep_decide_memb', _decide_route, {'continue': 'prep_check', '__end__': '__end__'})
    return sg.compile()

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("pipeline", build_pipeline_subgraph())
workflow.add_node("prep", build_prep_subgraph())
workflow.add_node("reply", reply_node)
workflow.add_node("reply_tools", reply_tools_node)
workflow.add_node("parse_results", parse_results_node)
workflow.add_node("gpcr_report", gpcr_report_node)
workflow.add_node("summarize", summarize_node)

workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    main_route_after_router,
    {"pipeline": "pipeline", "reply": "reply", "prep": "prep"},
)
workflow.add_edge("prep", "summarize")
workflow.add_edge("pipeline", "parse_results")
workflow.add_edge("parse_results", "gpcr_report")
workflow.add_edge("gpcr_report", "summarize")
workflow.add_conditional_edges(
    "reply",
    reply_route,
    {"reply_tools": "reply_tools", "summarize": "summarize"},
)
workflow.add_edge("reply_tools", "reply")
workflow.add_edge("summarize", END)


memory = MemorySaver()
agent_executor = workflow.compile(checkpointer=memory)


# 11. 打包任务产物为 zip
def package_results(task_id):
    if not task_id:
        return None, "找不到任务目标，请先提交执行一次任务。"
    task_dir = os.path.join("output", task_id)
    if not os.path.isdir(task_dir):
        return None, f"未找到任务目录 {task_dir}。"
    try:
        zip_path = shutil.make_archive(task_id + "_results", "zip", task_dir)
        return zip_path, f"打包完成：{zip_path}，请点击下载。"
    except Exception as e:
        return None, f"打包失败: {e}"


# ==========================================
# 12. UI 主题
# ==========================================
# 主题、CSS、布局都在 ui.py（build_ui），这里不再单独定义 Gradio 主题


# ==========================================
# 13. 任务提交逻辑：输入 = tpr + 步数 + run_name + 分析轨迹
# ==========================================
SYS_RULE = (
    "你是 GROMACS MD 体系准备、模拟与后处理分析助手。你的能力有四大块："
    "【体系准备】给一个蛋白结构（+可选配体），产出可直接跑的 .tpr："
    "pdb2gmx 生成蛋白拓扑 → 配体赋参（CGenFF 或 GAFF）→ 组装拓扑 → 建盒 → 溶剂化 → "
    "加离子中和 → grompp。可选力场（默认 CHARMM36，另有 Amber 系 / OPLS-AA / GROMOS）、"
    "水模型（默认 TIP3P，另有 TIP4P / TIP5P / SPC / SPC-E）、盒型与离子浓度；"
    "膜蛋白可选本地插 POPC 双层。准备完会出一份带判定的审查报告。"
    "【MD 模拟】建MD → 分析准备 → RMSD/RMSF/Rg/氢键 → 出图，可选 gmx_MMPBSA。"
    "【GPCR 分层分析】编号识别 → 激活坐标 → 构象景观 → 耦合机制 → 元动力学 FES → 出图 → 结论骨架。"
    "【文档检索】用 search_md_docs / read_file 查 GROMACS 命令与报错。"
    "系统按 md / gpcr / full / prepare 四种模式自动执行，不需要用户指定。"
    "【判断】对输入本身有歧义的地方（配体对接 pose 是否可信、质子化态、净电荷、"
    "配体参数来源、膜取向与质量、跑的过程中配体是否真的解离）会先做出判断并留下依据与反证条件；"
    "能自己验证的自己定，验证不了的会停下来问你。每条判断都写进 summary/decisions.md，"
    "连同它的依据与反证条件。"
    "【输入铁律】tpr 路径、轨迹列表、起点时间、对比工作区、蛋白与配体结构路径"
    "必须原样传递，禁止臆造。"
)

def _build_prompt(tpr_path, nsteps, run_name, traj_path, ws_b, analyze_only, user_msg):
    parts = []
    if analyze_only:
        parts.append("只做 GPCR 分层分析（不跑 MD 模拟）")
    if tpr_path:
        parts.append(f"tpr 路径: {tpr_path}")
    if nsteps:
        parts.append(f"步数: {nsteps}")
    if run_name:
        parts.append(f"run_name: {run_name}")
    if traj_path:
        parts.append(f"分析轨迹: {traj_path}")
    if ws_b:
        parts.append(f"对比工作区: {ws_b}")
    if user_msg:
        parts.append(f"指令: {user_msg}")
    return " | ".join(parts)


def submit_task(tpr_path, nsteps, run_name, traj_path, analyze_ws, ws_b, analyze_only,
                protein_pdb, ligand_file, prep_ff, prep_route, prep_water,
                prep_box, prep_pad, prep_salt, prep_conc, prep_membrane, prep_lipid,
                prep_escalate, user_msg, history, error_box, current_task_state):
    tpr_path = (tpr_path or "").strip()
    nsteps = (nsteps or "").strip()
    run_name = (run_name or "").strip() or "md"
    traj_path = (traj_path or "").strip()
    analyze_ws = (analyze_ws or "").strip()
    ws_b = (ws_b or "").strip()
    protein_pdb = (protein_pdb or "").strip()
    ligand_file = (ligand_file or "").strip()
    user_msg = (user_msg or "").strip()
    history = history or []
    error_box = error_box or ""

    task_id = current_task_state
    try:
        if analyze_ws and os.path.isdir(analyze_ws):
            task_id = os.path.basename(os.path.abspath(analyze_ws))
            workspace_dir = os.path.abspath(analyze_ws)
        else:
            task_id = f"task_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_{run_name}"
            workspace_dir = os.path.join("output", task_id)
        current_workspace.set(workspace_dir)
        os.makedirs(workspace_dir, exist_ok=True)

        # 准备体系：把界面上的选择落成 prep_config.json，工具从这里读参数。
        # 不靠 LLM 从自然语言里提取 —— 这些是下拉框选出来的确定值，
        # 经过模型只会多一层出错的可能。
        _is_prep = bool(protein_pdb or ligand_file)
        if _is_prep:
            _prep_cfg = {
                "forcefield": prep_ff, "water": prep_water,
                "ligand_route": prep_route, "box": prep_box,
                "box_padding_nm": prep_pad, "salt": prep_salt,
                "salt_conc_M": prep_conc, "membrane": bool(prep_membrane),
                "lipid": prep_lipid, "protein_pdb": protein_pdb,
                "ligand_file": ligand_file,
                "escalate": bool(prep_escalate),
            }
            with open(os.path.join(workspace_dir, "prep_config.json"), "w", encoding="utf-8") as _f:
                json.dump(_prep_cfg, _f, ensure_ascii=False, indent=2)

        final_prompt = _build_prompt(tpr_path, nsteps, run_name, traj_path, ws_b, analyze_only, user_msg)
        if _is_prep:
            final_prompt = ("准备体系 " + final_prompt).strip()
        if not final_prompt:
            yield history, "", error_box, task_id
            return

        config = {"configurable": {"thread_id": task_id}}
        history.append({"role": "user", "content": final_prompt})
        history.append({"role": "assistant", "content": "任务已提交，底层执行中..."})
        yield history, "", error_box, task_id

        for event in agent_executor.stream({
            "messages": [SystemMessage(content=SYS_RULE), HumanMessage(content=final_prompt)]
        }, config, stream_mode="updates"):
            for _node_name, node_state in event.items():
                if "messages" not in node_state:
                    continue
                last_msg = node_state["messages"][-1]
                if last_msg.type == "tool":
                    content = str(last_msg.content)
                    bad = _is_tool_error(last_msg)
                    yield history, "", error_box, task_id
                elif last_msg.type == "ai" and last_msg.content:
                    history[-1]["content"] += " " + str(last_msg.content)
                    yield history, "", error_box, task_id
                elif last_msg.type == "human":
                    nm = getattr(last_msg, "name", "")
                    if nm in ("ProgressReporter", "InputValidator", "MdEvalDirector", "GpcrReport"):
                        history[-1]["content"] += " " + str(last_msg.content)
                        yield history, "", error_box, task_id
    except Exception as e:
        error_box = "FATAL: " + str(e)
        yield history, "", error_box, task_id

    return history, "", error_box, task_id
def run_cli(tpr_path, nsteps, run_name, traj_path, analyze_ws, ws_b, analyze_only, user_msg):
    print("\n开始构建底层计算任务...")
    tpr_path = (tpr_path or "").strip()
    nsteps = (nsteps or "").strip()
    run_name = (run_name or "").strip() or "md"
    traj_path = (traj_path or "").strip()
    analyze_ws = (analyze_ws or "").strip()
    ws_b = (ws_b or "").strip()

    if analyze_ws and os.path.isdir(analyze_ws):
        task_id = os.path.basename(os.path.abspath(analyze_ws))
        workspace_dir = os.path.abspath(analyze_ws)
    else:
        task_id = f"task_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_{run_name}"
        workspace_dir = os.path.join("output", task_id)
    current_workspace.set(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)
    print(f"独立任务目录已创建: {workspace_dir}")

    final_prompt = _build_prompt(tpr_path, nsteps, run_name, traj_path, ws_b, analyze_only, user_msg)
    print(f"注入的系统指令: {final_prompt}\n")
    print("-" * 50)

    config = {"configurable": {"thread_id": task_id}}
    try:
        for event in agent_executor.stream(
            {"messages": [SystemMessage(content=SYS_RULE), HumanMessage(content=final_prompt)]},
            config,
            stream_mode="updates",
        ):
            for _node_name, node_state in event.items():
                if "messages" not in node_state:
                    continue
                last_msg = node_state["messages"][-1]
                if last_msg.type == "tool":
                    if _is_tool_error(last_msg):
                        print(f"执行报错 ({last_msg.name}): {last_msg.content}")
                    else:
                        print(f"执行成功: {last_msg.name}")
                elif last_msg.type == "ai" and last_msg.content:
                    print(f"\nAgent: {last_msg.content}\n")
                elif last_msg.type == "human":
                    nm = getattr(last_msg, "name", "")
                    if nm in ("ProgressReporter", "InputValidator", "MdEvalDirector", "GpcrReport"):
                        print(f"\n{last_msg.content}\n")
    except Exception as e:
        print(f"运行崩溃: {str(e)}")


# ==========================================
# 14. Gradio 页面构建
# ==========================================
from ui import build_ui, THEME, CSS

demo = build_ui(submit_task, package_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GROMACS MD 模拟与后处理 Agent 控制台")
    parser.add_argument("--mode", choices=["web", "cli"], default="web", help="运行模式: web(默认) 或 cli")
    parser.add_argument("--tpr", type=str, help="CLI模式必填: 起始 .tpr 路径")
    parser.add_argument("--nsteps", type=str, default="", help="CLI模式选填: 覆盖步数")
    parser.add_argument("--run-name", type=str, default="md", help="CLI模式选填: 输出前缀")
    parser.add_argument("--traj", type=str, default="", help="CLI模式选填: 分析轨迹(.xtc, 逗号分隔)")
    parser.add_argument("--ws", type=str, default="", help="CLI模式选填: 已有工作区(绝对路径)，填了就地分析不新建任务")
    parser.add_argument("--ws-b", type=str, default="", help="CLI模式选填: A/B 对比的第二个工作区(绝对路径)")
    parser.add_argument("--analyze-only", action="store_true", help="只对已有轨迹做 GPCR 分层分析，不跑 MD")
    parser.add_argument("--msg", type=str, default="", help="CLI模式选填: 附加自然语言指令")
    parser.add_argument("--port", type=int, default=7777, help="Web 模式选填: 监听端口（默认 7777，被占用时换一个）")

    args = parser.parse_args()

    if args.mode == "cli":
        if not args.tpr and not args.analyze_only:
            print("错误: CLI 模式下必须提供 --tpr 参数（或加 --analyze-only 只做分析）")
            print("示例: python mdagent/agent.py --mode cli --tpr step5_1.tpr --nsteps 5000000 --run-name md")
            print("示例: python mdagent/agent.py --mode cli --analyze-only --traj md.xtc --run-name apo")
            exit(1)
        run_cli(args.tpr, args.nsteps, args.run_name, args.traj, args.ws, args.ws_b, args.analyze_only, args.msg)
    else:
        print(f"正在启动网页控制台（端口 {args.port}）...")
        demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=False, theme=THEME, css=CSS)
