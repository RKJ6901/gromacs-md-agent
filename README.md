# MD Agent — GROMACS 分子动力学模拟 + GPCR / 元动力学后处理分析

把 rosetta-agent / boltz2-agent 的 LangGraph 骨架迁移到 MD 领域。一句话驱动两类任务：

1. **跑模拟**：提交 `gmx mdrun`（后台）→ 等 `.gro` → 建分析 index / 合并轨迹 →
   RMSD/RMSF/Rg/氢键 → 出图，可选 gmx_MMPBSA 结合能。
2. **做机制分析**：对已有轨迹做 GPCR 分层分析（Ballesteros-Weinstein 编号 → 激活坐标 →
   构象景观 → 耦合机制 → 元动力学 FES → 出图），最后给一份**五段式结论**
   （观察 / 统计 / 机制含义 / 边界 / 产出清单）。

## 架构 (LangGraph)

主图：

```
router ─(pipeline)─> pipeline 子图 ─> parse_results ─> gpcr_report ─> summarize
       └(reply)───> reply 分支(reply_tools 循环) ───────────────────> summarize
```

`pipeline` 子图（顺序硬编码；MD 四步任一步报错进 repair 分支）：

```
【LLM 驱动】 step1(run_gromacs_md 后台提交) -> progress_poll(等 *.gro) -> validate(检查 tpr)
             -> step2(prepare_analysis_index + merge_trajectories)
             -> step3(compute_rmsd/rmsf/gyrate/hbond)
             -> step4(plot_md_figures)
【确定性】   -> gpcr_setup(detect_receptor_profile)
             -> gpcr_l2(segments/microswitches/tm6_opening/ligand_retention)
             -> gpcr_l3(landscape/uncertainty)
             -> gpcr_l4(correlation/network/leadlag/water_network)
             -> gpcr_l5(parse_plumed/analyze_fes)
             -> gpcr_plot(plot_gpcr_figures)
             -> gpcr_compare(compare_systems，仅有对比工作区时)
```

三种模式（router 从用户输入自动判定）：

| 模式 | 触发 | 行为 |
|---|---|---|
| `md` | 提到 tpr / mdrun / 模拟 | 只跑 MD 四步 |
| `gpcr` | `--analyze-only`，或只提分析（元动力学/微开关/FES…） | 跳过 MD，直接做分层分析 |
| `full` | 两者都提到（"全流程"） | MD + 分层分析 |

### 核心机制

- **TaskAwareToolNode**：从 `thread_id` 注入独立工作区 `output/<task_id>/`，多任务物理隔离。
- **分层分析是确定性节点，不经过 LLM**（`_make_det_node`）：这几层没有需要"判断"的地方
  （顺序固定、参数全可省、路径自动解析），交给 LLM 反而实测会被整层跳过。
  LLM 只留在真正要理解用户输入的地方：提交 MD、检索文档、写结论。
- **五段式结论骨架**（`gpcr_report`，确定性、不调 LLM）：每个数字都直接取自分析 JSON；
  summarize 只负责按骨架组织语言。第五节"产出清单"直接 `glob` 磁盘，
  用来堵死"LLM 凭空说出图失败"。
- **诚实性硬约束**（写进骨架而非祈祷模型自觉）：报均值前给分块 bootstrap 95% CI 与有效样本量
  ESS（ESS<50 的均值标注不可信）；FES 结论先看收敛（`rms_centered` 是否随块长趋稳），
  未收敛不得报自由能深度与势垒；测不到的位点进 `missing_measurements`，不编造残基。
- progress_poll 纯确定性轮询 `.gro`，不消耗 LLM；MemorySaver 断点续跑；
  超长工具结果首尾截断。

## 文件

| 文件 | 作用 |
|---|---|
| `mdagent/tools.py` | 工具层：GROMACS 命令封装 + 29 个 `@tool`（LangGraph 与 MCP 共用）+ 出图工具 |
| `mdagent/agent.py` | LangGraph 编排 + Gradio UI + CLI（三种模式） |
| `mdagent/mcp_server.py` | MCP Server（29 个原子工具 + create_task/list_tasks/set_workspace） |
| `mdagent/gpcr_defs.py` | GPCR 领域层：BW 通用编号、TM/ICL/ECL 段、微开关位点；**纯标准库**，两个 env 都能 import |
| `mdagent/worker.py` | 重分析后端（跑在分析 env）：编号识别、分段、微开关、TM6、配体保留、PLUMED 解析 |
| `mdagent/gpcr_tasks.py` | 第 3/4/5 层：景观、不确定度、DCCM/MI、PSN、lead-lag、水网络、FES、A/B 对比 |
| `mdagent/plot_style.py` | 共享出图排版原语（两套出图共用，风格强制一致） |
| `mdagent/gpcr_plots.py` | GPCR 12 组分层图 |
| `mdagent/ui.py` / `mdagent/ui.css` | 控制台外观层：Gradio 6 主题、设计系统 CSS、布局；业务回调由 `mdagent/agent.py` 传入 |
| `mdagent/chainlit_app.py` | Chainlit 前端（另一套界面）：复用同一张图，把每个节点渲染成可展开的步骤；设置面板在左下角 |
| `mdagent/manuals/md_docs/` | RAG 语料 |

**为什么拆两个 env**：agent venv 只装 langchain/langgraph/gradio/mcp；MDAnalysis/scipy/sklearn
在分析 env 里。工具层通过子进程调 worker，agent venv 不必背这些重依赖。

## 工具一览（29 个）

| 分组 | 工具 |
|---|---|
| 模拟 | `run_gromacs_md` `check_md_status` |
| 分析准备 | `prepare_analysis_index` `merge_trajectories` |
| 常规分析 | `compute_rmsd` `compute_rmsf` `compute_gyrate` `compute_hbond` `run_mmpbsa` |
| 基础出图 | `plot_md_figures` |
| 编号 / 第2层 | `detect_receptor_profile` `analyze_segments` `analyze_microswitches` `analyze_tm6_opening` `analyze_ligand_retention` |
| 第3层 | `analyze_landscape` `analyze_uncertainty` |
| 第4层 | `analyze_correlation` `analyze_network` `analyze_leadlag` `analyze_water_network` |
| 第5层 | `parse_plumed` `analyze_fes` |
| 对比 / 出图 | `compare_systems` `plot_gpcr_figures` |
| 诊断/检索/IO | `diagnose_and_fix` `auto_repair` `read_file` `check_logs` `search_md_docs` |

## 安装

```bash
conda env create -f environment.yml && conda activate md-agent
cp .env.example .env
# 填 LLM_MODEL / LLM_API_KEY / LLM_BASE_URL、GMX_BIN / GMX_MMPBSA_BIN；
# 重分析还需 MD_ANALYSIS_PY 指向装了 MDAnalysis/scipy/sklearn 的 python
```

## 用法

```bash
# Web 控制台（默认，http://0.0.0.0:7777；被占用时用 --port 换端口）
python mdagent/agent.py

# 另一套前端：Chainlit（步骤式界面，端口自定）
chainlit run mdagent/chainlit_app.py --host 0.0.0.0 --port 7779

# 跑 MD 模拟 + 基础分析
python mdagent/agent.py --mode cli --tpr step5_1.tpr --nsteps 5000000 --run-name md

# 只对已有轨迹做 GPCR 分层分析（就地分析，不新建任务目录）
python mdagent/agent.py --mode cli --analyze-only \
    --ws /data2/JRK/md_workspaces/gpcr_test --traj md.xtc --run-name apo

# 可选：A/B 对比（第二个体系必须已经跑过同样的分层分析）
python mdagent/agent.py --mode cli --analyze-only --ws /path/A --ws-b /path/B
```

MCP（接入 Claude Desktop / Cursor）：

```json
{"mcpServers": {"md-agent": {
  "command": "/path/to/venv/bin/python",
  "args": ["/abs/path/to/mdagent/mcp_server.py"]}}}
```

## 关键约定与坑（踩过并已修）

- **MDAnalysis 读不了 GROMACS 2026 的 `.tpr`**（tpx 版本太新）→ 拓扑走 `.gro`/`.pdb`，
  坐标走 `.xtc`；但 `gmx trjconv -pbc mol` 反而**需要** `.tpr`，所以两者并存、用途不同。
- **MDAnalysis 读出的是 Å，GROMACS 是 nm** → 一律 ×0.1，否则距离全部大 10 倍。
- **`gmx hbond` 在 2026 改了语义**：`-o` 只写 index(.ndx)，氢键数曲线要用 `-num <out>_count.xvg`。
- **TM6 的胞内端是它的低残基号一侧**：class A 里 TM6 自胞内走向胞外，与 TM3/TM7 相反。
  取错端会把跨膜束对角距离（~4.5 nm）当成胞内开口。
- **传 gmx 的路径必须是绝对路径**：`_run()` 会把 cwd 切到工作区，相对路径会被二次拼接。
- 分段轨迹从 0 ps 重启时必须用 `start_times` 手动指定每段起点。
- `plumed sum_hills` 的 1D 输出第 3 列是**导数**不是自由能，只能按列名 `file.free` 取。
- 元动力学欠采样区的尾部 FES 会虚高到几百 kJ/mol（well-tempered 固有偏差），
  盆地/收敛分析必须限制在 `F <= f_max` 的已采样窗口内。
- **复用工作区别用 `cp -r` 带旧产物**：工作区被复制后，上一轮的 `analysis_*.json` /
  `*.csv` 仍在盘上，报告会把它们当成"本次实测"读出（实测撞到过：旧 TM6 值 4.47 nm
  与本次 1.47 nm 同时出现）。跳过 A/B 时节点会主动清掉旧的
  `analysis_compare.json` / `compare_metrics.csv`；其他层每次都会覆写，一般无残留。

## 体系准备（蛋白 + 配体 → 可运行的 .tpr）

除了跑模拟和分析已有轨迹，agent 还能把**蛋白结构和配体**准备成可直接跑的 `.tpr`。

界面上第三页「准备体系」：填蛋白 `.pdb` 和配体 `.mol2/.sdf`，选力场/水模型/盒型/离子，
提交即可。流程全部确定性（参数从工作区的 `prep_config.json` 读，不经过 LLM）：

```
prep_protein  pdb2gmx 生成蛋白拓扑（自动做残基名跨力场归一化）
prep_ligand   配体赋参：CGenFF（半自动）或 GAFF/acpype（全本地）
prep_assembly 拼 topol.top（atomtypes 拆到独立文件，满足 GROMACS 的指令顺序要求）
prep_box      editconf 建盒
prep_solvation solvate 加水
prep_ions     genion 加盐并中和
prep_tpr      grompp 产出 step5_1.tpr
prep_check    生成 summary/prep_review.md 审查报告
```

产物名特意用 `step5_1.tpr`，与 CHARMM-GUI 惯例一致，`run_gromacs_md` 默认就找这个名字。

### 力场与水模型

| 力场 | 家族 | 配体路线 | 备注 |
|---|---|---|---|
| `charmm36-jul2022` | CHARMM | CGenFF | **推荐**；本机唯一自带脂质与 CGenFF 的力场 |
| `amber99sb-ildn` / `amber14sb` / `amber19sb` | Amber | GAFF | 需残基名改名（流程自动做） |
| `oplsaa` / `gromos54a7` | OPLS / GROMOS | — | 本机无对应家族的配体参数路线 |

水模型：`tip3p`（CHARMM36 默认）/ `tip4p` / `tip4pew` / `tip5p` / `spc` / `spce`。

`mdagent/prep_defs.py` 里有兼容性检查：选了不自洽的组合（如 GAFF 配体配 CHARMM36 蛋白）
不会报错，但会出告警并写进审查报告。

### 审查报告

`summary/prep_review.md` 是流程产物之一，不是可选的装饰。准备阶段的错误大多是
「不崩但结果不可用」——力场不自洽、残基名不匹配、膜未平衡、盒子过小，都不会让程序停下来。
报告把配置、兼容性告警、逐步执行记录、体系组成、以及**这次准备不能保证什么**全列出来。

### 力场的额外依赖

CHARMM36 的 GROMACS 移植版在 `forcefields/charmm36-jul2022.ff/`（来自官方 jul2022 tgz）。
`GMXLIB` 会指向该目录——实测它是追加而非替换，GROMACS 自带的力场仍能找到。

配体赋参的额外依赖：

- CGenFF 转换器：`/data/opt-bk/opt/PyAutoFEP/Tools/cgenff_charmm2gmx_py3.py`
  必须用一个带 networkx 的 Python 跑（三个 conda env 都没有）
  赋参本身要在 `app.cgenff.com` 上传完成（无免费接口），流程会在缺 `.str` 时停下等人工
- acpype：**需在 AmberTools 环境中通过 `conda run -n gmx` 调用**
  （它内部要调 antechamber → sqm，直接调因缺 AmberTools 环境变量会秒退）

上面这些可执行文件的具体路径写进 `.env`
（`PREP_CONVERTER_PY` / `PREP_ACPYPE` / `PREP_CONDA` / `PREP_PACKMOL`），代码里不再硬编码绝对路径。

## 判断层（decision layer）

一个标准复合物不需要 agent：`pdb2gmx` + `solvate` + `genion` 谁都能跑。**需要判断的地方才是
agent 的意义**——输入本身有歧义时（配体对接 pose 可不可信、质子化态、净电荷、配体参数来源、
膜取向与质量、跑的过程中配体是否真的解离），agent 先做出判断并记账，而不是默认往下跑。

五个判断点（前四个在准备阶段，第五个在跑完之后）：

1. **质子化态与二硫键** `decide_protonation_scan` / `decide_apply_protonation` —— 扫几何找
   可成键的 Cys 对与组氨酸质子化取向，写 `protein_decided.pdb` 供 `pdb2gmx` 使用；
2. **净电荷** `decide_charge_audit` —— 逐原子复算 `topol.top`，不信 `grompp` 的汇总；
3. **配体参数来源** `decide_ligand_params` —— 看 CGenFF 罚分，定半自动赋参结果能不能用；
4. **膜质量** `decide_membrane_check` —— 查 POPC 双层几何、蛋白跨膜位置与倾角；
5. **配体是否真的离开口袋** `decide_dissociation_check` —— 区分真实解离与力场假象。

每条判断都带 **依据 / 反证条件 / 重做代价** 三样：依据说明结论是从哪个文件的哪个数读出来的，
反证条件说明出现什么就表明判断错了、要重做。证据一律**从真实文件解析**（PDB 坐标、`topol.top`
电荷、CGenFF 罚分、`.gro` 盒子），**绝不由模型生成**——模型只判断能不能用，不编数。

拿不准就不硬撑：工具返回以「需要你定夺」开头，整条链**停下来等用户拍板**，不带着悬而未决的量
往下跑。所有判断写进工作区 `decision_log.json`，并汇总成 `summary/decisions.md`。

判断层用 **7W0M** 的 CHARMM-GUI 真值端到端核对：净电荷 **+7**、二硫键 **Cys102–Cys181**、
组氨酸**全部 HSD**，三条都对得上才算通过。
