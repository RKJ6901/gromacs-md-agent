"""md-agent 的 Chainlit 前端：复用 md_agent 里的 LangGraph 图与工具，只换界面。

启动：chainlit run md_chainlit.py -h 0.0.0.0 -p 7779
设置面板填路径，聊天框写要做什么（等价于原来的"附加指令"）。
"""
import datetime
import os

import chainlit as cl
from chainlit.input_widget import Select, Switch, TextInput
from langchain_core.messages import HumanMessage, SystemMessage

import md_agent as A

WELCOME = (
    "### MD Agent 控制台\n\n"
    "这个系统能做五件事：\n\n"
    "1. **体系准备** —— 给蛋白结构（± 配体），产出可直接跑的 `.tpr`。"
    "力场、水模型、盒型、离子浓度可选；膜蛋白可本地插 POPC 双层。\n"
    "2. **MD 模拟** —— 建 MD、跑 GROMACS，出 RMSD/RMSF/Rg/氢键图。\n"
    "3. **GPCR 分层分析** —— 对已有轨迹做激活坐标 / 构象景观 / 耦合机制 / 元动力学 FES。\n"
    "4. **文档检索** —— 查 GROMACS 命令与报错。\n"
    "5. **判断层** —— 准备体系前会判断质子化态/二硫键、净电荷、配体参数来源、膜质量，"
    "跑完判断配体是否真的离开口袋；每条判断都写下依据与反证条件，拿不准会停下来问你。\n\n"
    "准备体系时，在左下角 **设置** 里填蛋白与配体路径；跑模拟或分析时填 tpr / 工作区。"
    "留空的项会被忽略。"
)


NODE_LABELS = {
    "router": "判定任务模式",
    "progress_poll": "等待模拟进度",
    "init": "启动",
    "validate": "输入校验",
    "parse_results": "解析分析结果",
    "gpcr_report": "汇总结论骨架",
    "gpcr_setup": "第1层 编号识别",
    "gpcr_l2": "第2层 激活坐标",
    "gpcr_l3": "第3层 构象景观",
    "gpcr_l4": "第4层 耦合机制",
    "gpcr_l5": "第5层 元动力学",
    "gpcr_plot": "出图",
    "gpcr_compare": "A/B 对比",
    "prep_decide": "判定质子化与二硫键",
    "prep_decide_lig": "判定配体参数来源",
    "prep_decide_charge": "判定净电荷",
    "prep_decide_memb": "判定膜质量",
    "decide_run": "判定配体解离",
    "reply": "检索与回复",
    "reply_tools": "检索文档",
}




def _prep_settings():
    # 选项直接从 md_prep_defs 取，避免两处维护力场/水模型列表。
    import md_prep_defs as _P
    out = [
        TextInput(id='protein_pdb', label='准备体系-蛋白pdb', initial=''),
        TextInput(id='ligand_file', label='准备体系-配体文件', initial=''),
    ]
    out.append(Select(id='prep_ff', label='力场', values=list(_P.FORCEFIELDS.keys()), initial_value='charmm36-jul2022'))
    out.append(Select(id='prep_route', label='配体路线', values=list(_P.LIGAND_ROUTES.keys()), initial_value='cgenff'))
    out.append(Select(id='prep_water', label='水模型', values=list(_P.WATER_MODELS.keys()), initial_value='tip3p'))
    out.append(Switch(id="prep_escalate", label="遇到歧义停下来问我", initial=False))
    return out


def _write_prep_cfg(workspace_dir, prep_ff, prep_water, protein_pdb, ligand_file, prep_escalate=False):
    # 把设置面板里选的准备参数落成 prep_config.json（工具从那里读）。
    import json as _json
    cfg = {
        'forcefield': prep_ff,
        'water': prep_water,
        'ligand_route': 'cgenff',
        'box': 'dodecahedron',
        'box_padding_nm': 1.0,
        'salt': 'NaCl',
        'salt_conc_M': 0.15,
        'membrane': False,
        'lipid': 'POPC',
        'protein_pdb': protein_pdb,
        'ligand_file': ligand_file,
        'escalate': bool(prep_escalate),
    }
    path = os.path.join(workspace_dir, 'prep_config.json')
    with open(path, 'w', encoding='utf-8') as f:
        _json.dump(cfg, f, ensure_ascii=False, indent=2)
    return path

def _settings_inputs():
    return [
        TextInput(id="tpr", label="起始 .tpr 路径（跑模拟时填）", initial=""),
        TextInput(id="nsteps", label="步数（可空，空=用 tpr 原值）", initial=""),
        TextInput(id="run_name", label="输出前缀 run_name", initial="md"),
        TextInput(id="traj", label="分析轨迹 .xtc（多段用逗号分隔）", initial=""),
        TextInput(id="ws", label="分析工作区（填了就就地分析，不新建任务目录）", initial=""),
        TextInput(id="ws_b", label="对比工作区（可空，A/B 对比用）", initial=""),
        Switch(id="analyze_only", label="只做 GPCR 分层分析，不跑模拟", initial=False),
    ]


@cl.on_chat_start
async def _start():
    cl.user_session.set("cfg", {})
    await cl.Message(content=WELCOME).send()
    await cl.ChatSettings(_settings_inputs() + _prep_settings()).send()


@cl.on_settings_update
async def _settings(settings):
    cl.user_session.set("cfg", settings or {})
@cl.on_message
async def _message(msg: cl.Message):
    cfg = cl.user_session.get("cfg") or {}
    text = (msg.content or "").strip()

    tpr = str(cfg.get("tpr") or "").strip()
    nsteps = str(cfg.get("nsteps") or "").strip()
    run_name = str(cfg.get("run_name") or "").strip() or "md"
    traj = str(cfg.get("traj") or "").strip()
    ws = str(cfg.get("ws") or "").strip()
    ws_b = str(cfg.get("ws_b") or "").strip()
    analyze_only = bool(cfg.get("analyze_only"))
    protein_pdb = str(cfg.get("protein_pdb") or "").strip()
    ligand_file = str(cfg.get("ligand_file") or "").strip()
    prep_ff = str(cfg.get("prep_ff") or "").strip() or "charmm36-jul2022"
    prep_route = str(cfg.get("prep_route") or "").strip() or "cgenff"
    prep_water = str(cfg.get("prep_water") or "").strip() or "tip3p"
    prep_escalate = bool(cfg.get("prep_escalate"))
    is_prep = bool(protein_pdb or ligand_file)

    # 只有真填了任务参数、或消息本身像任务，才当任务跑，否则当普通对话。
    # 不能无条件把 run_name 拼进 prompt：字符串 "run_name" 本身就是路由的 MD 关键词，
    # 一带上，路由就会把"你好"也判成 md 任务（这就是"发你好也在跑任务"的原因）。
    has_task = bool(tpr or nsteps or traj or ws or ws_b or analyze_only or is_prep)
    if has_task:
        prompt = A._build_prompt(tpr, nsteps, run_name, traj, ws_b, analyze_only, text)
    else:
        prompt = text

    if not prompt.strip():
        await cl.Message(content="先在左下角设置里填路径，或直接说明要做什么。").send()
        return

    if is_prep:
        prompt = ('准备体系 ' + prompt).strip()

    is_task = has_task or bool(A._detect_mode(prompt))

    if ws and os.path.isdir(ws):
        task_id = os.path.basename(os.path.abspath(ws))
        workspace_dir = os.path.abspath(ws)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        task_id = "task_%s_%s" % (stamp, run_name)
        workspace_dir = os.path.join("output", task_id)
    A.current_workspace.set(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)
    if is_prep:
        _write_prep_cfg(workspace_dir, prep_ff, prep_water, protein_pdb, ligand_file, prep_escalate)

    if is_task:
        await cl.Message(content="已提交，工作区 `%s`，模式按输入自动判定。" % task_id).send()

    config = {"configurable": {"thread_id": task_id}}
    init = {"messages": [SystemMessage(content=A.SYS_RULE), HumanMessage(content=prompt)]}
    answer = ""
    try:
        async for event in A.agent_executor.astream(init, config, stream_mode="updates"):
            for node, st in (event or {}).items():
                msgs = (st or {}).get("messages") or []
                if not msgs:
                    continue
                body = str(getattr(msgs[-1], "content", "") or "").strip()
                if not body:
                    continue
                if node == "summarize":
                    answer = body
                    continue
                if not is_task:
                    continue
                async with cl.Step(name=NODE_LABELS.get(node, node), type="tool") as step:
                    step.output = body[:2000]
    except Exception as e:
        await cl.Message(content="执行出错：%s" % e).send()
        return

    await cl.Message(content=answer or "执行结束，详见上方步骤。").send()

    if is_task:
        zip_path, note = A.package_results(task_id)
        if zip_path:
            await cl.Message(content=note, elements=[cl.File(name=os.path.basename(zip_path), path=zip_path)]).send()
