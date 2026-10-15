# GROMACS MD Agent

`gromacs-md-agent` 是一个 LangGraph 编排的 agent，把 **体系准备 → MD 模拟 → GPCR / 元动力学后处理分析** 三段串成一条流水线：给一句话，它判断这是要跑模拟、准备体系，还是对已有轨迹做机制分析，再按固定的图执行。准备阶段把蛋白（±配体）做成可运行的 `.tpr`；模拟阶段后台提交 `gmx mdrun` 并轮询；分析阶段做常规指标（RMSD / RMSF / Rg / 氢键）和 GPCR 分层分析（编号 → 激活坐标 → 构象景观 → 耦合 → 元动力学 FES），最后出一份带不确定性标注的中文结论。整套能力从四个入口暴露——CLI、Gradio 控制台、Chainlit 前端、MCP——它们共用同一套工具。

> 它既不是套在 `gmx` 命令外面的一层 LLM 壳，也不是把命令串起来的 bash 脚本。顺序、每个节点能调哪些工具、出错之后怎么恢复，都写死在图的拓扑里；对输入本身有歧义的地方，判断层在开跑之前先把量定下来并记账，每条判断都写明什么观测会证明它是错的。

---

## 1. 仓库结构

接原骨架的目录，各文件职责如下（注释即用途）：

```
.
├── .env.example          # LLM / GROMACS / gmx_MMPBSA 路径模板（复制成 .env 后填）
├── .gitattributes        # 换行符与二进制文件的处理规则
├── .gitignore            # 排除 output/、.env、forcefields/ 等
├── environment.yml       # conda 环境定义（agent venv：langchain / langgraph / gradio / mcp）
├── run_cli.sh            # CLI 批量提交脚本
├── README.md             # 本文件
├── AGENT_NOTES.md        # 12 个设计取舍的深挖：问题 / 优化点 / 做法 / 为什么不用别的 / 会被追问什么
├── BUGLOG.md             # 16 条真实 bug 的现象、定位过程、根因、修法
├── PROJECT_EXPLAINER.md  # 面向没有 agent 背景读者的项目说明
├── docs/
│   └── figs/*.png        # 文档示例图（agent_200ps.png / md_400ps.png）
├── mdagent/
│   ├── agent.py              # LangGraph 编排（主图 + pipeline / prep 两个子图）+ Gradio UI + CLI
│   ├── tools.py              # 工具层：GROMACS 命令封装 + 30 个 @tool（含 GPCR 出图，LangGraph 与 MCP 共用）
│   ├── worker.py             # 重分析后端（跑在分析 env）：编号识别、分段、微开关、TM6、配体保留、PLUMED 解析
│   ├── gpcr_defs.py          # GPCR 领域层：BW 通用编号、TM/ICL/ECL 段、微开关位点（纯标准库）
│   ├── gpcr_tasks.py         # 第 3/4/5 层：景观、不确定度、DCCM/MI、PSN、lead-lag、水网络、FES、A/B 对比
│   ├── gpcr_plots.py         # GPCR 12 组分层图
│   ├── plot_style.py         # 共享出图排版原语（两套出图共用，风格强制一致）
│   ├── prep_defs.py          # 体系准备领域层：力场 / 水模型 / 盒型 / 配体路线 / 脂质注册表 + compat_check()（纯标准库）
│   ├── prep_tools.py         # 体系准备工具层：12 个 @tool，蛋白拓扑 → 配体赋参 → 拼 topol.top → 建盒 → 溶剂化 → 加离子 → grompp → 审查报告（含本地膜插入）
│   ├── decision_defs.py      # 判断层·策略层：阈值表与判定逻辑（依据 / 反证条件 / 重做代价）（纯标准库）
│   ├── decision_tools.py     # 判断层·证据与执行层：7 个 @tool，把真实文件解析成证据、落进 decision_log.json 并执行判定
│   ├── ui.py                 # 控制台外观层：Gradio 6 主题、设计系统 CSS、布局；业务回调由 agent.py 传入
│   ├── ui.css                # 控制台设计系统样式
│   ├── chainlit_app.py       # Chainlit 前端：复用同一张图，把每个节点渲染成可展开的步骤；设置面板在左下角
│   ├── mcp_server.py         # MCP Server（49 个原子工具 + create_task/list_tasks/set_workspace）
│   └── manuals/md_docs/      # RAG 语料
└── tests/
    ├── e2e_test.py           # 端到端脚本
    ├── e2e_tools.py          # 工具层端到端脚本
    ├── test_agent_gpcr.py    # GPCR 分析节点测试
    └── test_gpcr_plots.py    # 出图测试
```

## 2. 框架（LangGraph）

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

**关键设计**

- schema 级权限：每个 LLM 节点在工厂里就 `bind_tools([本步工具])`，清单外的工具连 schema 都看不见。收尾节点 `bind_tools([])`，物理上禁止再调任何工具，防的是模型陷入"一直调工具但永远不给结论"的循环。
- 分层分析是确定性节点，不经过 LLM（`_make_det_node`）：这几层没有需要"判断"的地方（顺序固定、参数全可省、路径自动解析），交给 LLM 反而实测会被整层跳过——模型返回空的 `tool_calls`，被出口条件当成"任务做完了"，一层都没跑且没有任何缺失提示。LLM 只留在真正要理解用户输入的地方：提交 MD、检索文档、写结论。
- MD 四步任一步报错进 repair 分支：读日志判断原因，修好后回到出错那步重来，`retry_count` 超过 2 才真退出。
- 五段式结论骨架（`gpcr_report`，确定性、不调 LLM）：每个数字都直接取自分析 JSON，summarize 只负责按骨架组织语言；第五节"产出清单"直接 `glob` 磁盘，用来堵死"LLM 凭空说出图失败"。
- 诚实性硬约束（写进骨架而非祈祷模型自觉）：报均值前给分块 bootstrap 95% CI 与有效样本量 ESS（ESS<50 的均值标注不可信）；FES 结论先看收敛（`rms_centered` 是否随块长趋稳），未收敛不得报自由能深度与势垒；测不到的位点进 `missing_measurements`，不编造残基。
- progress_poll 纯确定性轮询 `.gro`，不消耗 LLM；MemorySaver 断点续跑；超长工具结果首尾截断。

## 3. 流水线三段

| 段 | 工具 | 作用 |
|---|---|---|
| 体系准备 | `prep_protein_topology` `prep_ligand_cgenff` / `prep_ligand_gaff` `prep_assemble` `prep_build_box` `prep_solvate` `prep_ionize` `prep_grompp` `prep_review`（膜蛋白再加 `prep_orient_protein` `prep_build_membrane`） | 蛋白 + 配体 → 可运行的 `step5_1.tpr` 与审查报告 |
| MD 模拟 | `run_gromacs_md` `check_md_status` `prepare_analysis_index` `merge_trajectories` `compute_rmsd` `compute_rmsf` `compute_gyrate` `compute_hbond` `plot_md_figures` `run_mmpbsa` | 后台提交 `gmx mdrun`、轮询、建索引合并轨迹、算常规指标（RMSD / RMSF / Rg / 氢键）、出图、MM/GBSA 结合能 |
| GPCR 分层分析 | `detect_receptor_profile` `analyze_segments` `analyze_microswitches` `analyze_tm6_opening` `analyze_ligand_retention` `analyze_landscape` `analyze_uncertainty` `analyze_correlation` `analyze_network` `analyze_leadlag` `analyze_water_network` `parse_plumed` `analyze_fes` `plot_gpcr_figures` `compare_systems` | 编号对齐 → 激活坐标 → 构象景观 → 耦合机制 → 元动力学 FES，最后出图与 A/B 对比 |

### 输入铁律

tpr 路径、轨迹列表、起点时间、对比工作区、蛋白与配体结构路径必须原样传递，模型不得臆造——它编一个不存在的路径，工具就报错，整条链停下来。

---

## 4. 判断层（decision layer）

判断层回答的不是"怎么跑"，而是"跑之前该定哪几个量、凭什么定"。一个标准复合物的准备是一条流水线（`pdb2gmx` + `solvate` + `genion`），谁都能跑；agent 的价值不在这类输入上，而在输入本身有歧义的时候——配体对接 pose 可不可信、质子化态、净电荷、配体参数来源、膜取向与质量、跑的过程中配体是否真的解离。这些地方先做出判断并记账，而不是默认往下跑。

### 4.1 五个判断点

| 判断点 | 工具 | 判据 | 落在哪里（产物） |
|---|---|---|---|
| 质子化态与二硫键 | `decide_protonation_scan` / `decide_apply_protonation` | 每对 Cys 的 SG–SG ≤ 0.25 nm 判二硫键；每个 His 的 Nδ1/Nε2 与最近羧基氧 ≤ 0.30 nm 判 HSP，否则取中性 HSD；SG–SG 落在 0.25–0.35 nm 灰区就停下 | `protein_decided.pdb`（供 `pdb2gmx`）、`decision_log.json` |
| 净电荷 | `decide_charge_audit` | 逐原子复算 `topol.top` 的电荷列、按 moleculetype 归并、乘分子数；溶质必须中和到 0 | `decision_log.json`（供 `genion` 定离子数） |
| 配体参数来源 | `decide_ligand_params` | CGenFF 最大罚分 ≤ 50 可信；50–100 标注"该处原子类型是类比出来的"；> 100 停下 | `decision_log.json` |
| 膜质量 | `decide_membrane_check` | 脂质–蛋白最近原子距离 ≥ 0.30 nm；上下叶脂质数差 ≤ 5%；盒 z 方向水层 ≥ 1.60 nm；TM 螺旋 ≥ 5 条 | `decision_log.json` |
| 配体是否真的离开口袋 | `decide_dissociation_check` | 接触保留率 ≥ 0.60 且叠合后配体 RMSD ≤ 0.30 nm 判结合保持；保留率 ≤ 0.20 且最小距离 ≥ 1.00 nm 判离开 | `decision_log.json` |

前四个判断在准备阶段，第五个在跑完之后。阈值都写在 `mdagent/decision_defs.py` 的常量里，每个数字后面写了它的物理来处（SG–SG 共价键长约 0.204 nm，取 0.25 作判据；CGenFF 罚分的 50 / 100 两档；接触保留率的 0.60 / 0.20）。

### 4.2 每条判断都带反证条件

每条判断都带三样：依据（结论是从哪个文件的哪个数读出来的）、反证条件（出现什么观测就说明这个判断错了、要重做）、重做代价。反证条件是必填项——`DECISION_SPECS` 里每一条都写了 `falsifier`，写不出反证条件的判断等于没有落点，出了错也拦不住。

以质子化判定为例：依据是每个 His 的 Nδ1/Nε2 到最近羧基氧的距离、每对 Cys 的 SG–SG 距离；反证条件是"若显式 HSP 与默认 HSD 各跑 10 ns，HSP 版该残基周边的氢键占据率或局部 RMSF 明显更稳，则原判定为错"；重做代价是"改 PDB 残基名后重跑 `pdb2gmx`，约 1 分钟"。净电荷那条同理，反证条件是"加离子后重新解析 `topol.top` 的 [ atoms ] 求电荷和，若不为 0，则 `genion` 的 `-neutral` 没生效或拓扑被改写"。

证据一律从真实文件解析（PDB 坐标、`topol.top` 电荷、CGenFF 罚分、`.gro` 盒子），绝不由模型生成——模型只判断能不能用，不编数。解析不到就宁可缺项、跳过落账，不凭感觉补一个数。

### 4.3 落在灰区就停下来

判定有三个结果：`DECIDED`（能自己验的，agent 自己定）、`ESCALATE`（验不了的，停下来问人）。灰区的判定在返回串最前面打上 `【需要你定夺】` 标记，图认出这个前缀就停下整条链，不带着悬而未决的量往下跑。

每次运行有一个 `escalate` 开关（Gradio 第三页的复选框「遇到歧义停下来问我」，Chainlit 设置面板里的同名开关）：打开后 `medium` / `low` 置信度的已定判定也升级成"停下来问人"；默认关闭，agent 自己决定并记账。

所有判断写进工作区 `decision_log.json`，并由 `decide_review` 汇总成 `summary/decisions.md`。账本里每条都是"判定 / 选择 / 依据 / 反证条件 / 重做代价 / 证据"。

### 4.4 真值校验

判断层用 7W0M 的 CHARMM-GUI 真值端到端核对：从原始的 `MD_run.pdb`（51 个未质子化的 HIS、97 个 CYS、带配体、无离子）出发，这一层独立复现了 CHARMM-GUI 自己的决定——净电荷 +7（PROA +8 + LIG −1）、二硫键 Cys102–Cys181（SG–SG 0.203 nm）、组氨酸全部取 HSD。这说明判断逻辑在一个真实体系上对得上实验者手工做的决定。这是单个体系，不是一次验证运动。

## 5. 体系准备（蛋白 + 配体 → 可运行的 .tpr）

除了跑模拟和分析已有轨迹，agent 还能把蛋白结构和配体准备成可直接跑的 `.tpr`。

### 5.1 流程

界面上第三页「准备体系」：填蛋白 `.pdb` 和配体 `.mol2 / .sdf`，选力场 / 水模型 / 盒型 / 离子，提交即可。流程全部确定性（参数从工作区的 `prep_config.json` 读，不经过 LLM）：

```
prep_decide    判定质子化态与二硫键，写 protein_decided.pdb（供 pdb2gmx 读）
prep_protein   pdb2gmx 生成蛋白拓扑（自动做残基名跨力场归一化）
prep_ligand    配体赋参：CGenFF（半自动）或 GAFF/acpype（全本地）；无配体则跳过
prep_decide_lig 判定配体参数来源能不能用
prep_assembly  拼 topol.top（atomtypes 拆到独立文件，满足 GROMACS 的指令顺序要求）
prep_box       editconf 建盒
prep_solvation solvate 加水
prep_ions      genion 加盐并中和
prep_decide_charge 逐原子复算净电荷，核加离子数
prep_tpr       grompp 产出 step5_1.tpr
prep_decide_memb   判定膜几何能不能接受
prep_check     生成 summary/prep_review.md，并汇总判断台账 summary/decisions.md
```

产物名特意用 `step5_1.tpr`，与 CHARMM-GUI 惯例一致，`run_gromacs_md` 默认就找这个名字。
膜蛋白会先走 `prep_orient` → `prep_decide` → `prep_protein` → `prep_membrane` 插入本地 POPC 双层，再进入上面的链（定向必须在 `pdb2gmx` 之前——`pdb2gmx` 不平移坐标，先建拓扑的话蛋白会停在原始位置而脂质围着原点铺，实测差 13 nm）。

### 5.2 力场与水模型

| 力场 | 家族 | 配体路线 | 备注 |
|---|---|---|---|
| `charmm36-jul2022` | CHARMM | CGenFF | 推荐；本机唯一自带脂质与 CGenFF 的力场 |
| `amber99sb-ildn` / `amber14sb` / `amber19sb` | Amber | GAFF | 需残基名改名（流程自动做） |
| `oplsaa` / `gromos54a7` | OPLS / GROMOS | — | 本机无对应家族的配体参数路线 |

水模型：`tip3p`（CHARMM36 默认）/ `tip4p` / `tip4pew` / `tip5p` / `spc` / `spce`。

`mdagent/prep_defs.py` 里有兼容性检查：选了不自洽的组合（如 GAFF 配体配 CHARMM36 蛋白）不会报错，但会出告警并写进审查报告。

CHARMM36 的 GROMACS 移植版放在 `<仓库根>/forcefields/charmm36-jul2022.ff/`，这个目录不在仓库里，需要自己取官方文件解压到该位置（见 7.3）。`GMXLIB` 会指向它——实测这是追加而非替换，GROMACS 自带的力场仍能找到。

### 5.3 配体赋参的两条路线

- CGenFF（半自动）：赋参本身要在 `app.cgenff.com` 上传完成，没有免费的全自动接口；流程会在缺 `.str` 时停下等人工。取回的 `.str` 用一个转换器（默认 `cgenff_charmm2gmx_py3.py`）转成 GROMACS 拓扑，该转换器必须用一个带 networkx 的 Python 跑（三个 conda env 都没有）。
- GAFF / acpype（全本地）：必须在 AmberTools 环境中通过 `conda run` 调用 acpype——它内部要调 antechamber → sqm，直接调会因缺 AmberTools 环境变量而秒退（`Cannot properly run sqm`）。

上面这些可执行文件的具体路径写进 `.env`（`PREP_*`，见 7.4），代码里不再硬编码绝对路径。

### 5.4 审查报告

`summary/prep_review.md` 是流程产物之一，不是可选的装饰。准备阶段的错误大多是"不崩但结果不可用"——力场不自洽、残基名不匹配、膜未平衡、盒子过小，都不会让程序停下来。报告把配置、兼容性告警、逐步执行记录、体系组成、以及这次准备不能保证什么，全列出来。

---

## 6. 任务目录结构

MD 任务（`output/task_YYYYMMDD_HHMM_<run_name>/`）：

```
md.gro                  # 模拟完成标志（也是分析拓扑来源之一）
md.xtc                  # 轨迹
md.edr                  # 能量
md.log                  # mdrun 日志（repair 读它判断报错）
md.cpt                  # 检查点
md.pid                  # 后台 mdrun 的 PID
md.out                  # nohup 的 stdout/stderr
analysis.ndx            # 分析索引
groups.json             # 索引分组定义
rmsd.xvg
rmsf.xvg
gyrate.xvg
hbond_count.xvg         # 氢键数曲线（-num <out>_count.xvg）
hbond_bonds.ndx         # 氢键配对索引（-o <out>_bonds.ndx）
figs/
  md.pdf
  md.png
```

分析任务（`--analyze-only` 或 `--ws <已有目录>`）：

```
gpcr.json               # 编号档案
analysis_<task>.json    # 各层分析结果
*.csv                   # dccm.csv / compare_metrics.csv / landscape_fes_*.csv 等
analysis_sub.xtc        # Protein_LIG 子轨迹
analysis_sub.gro        # 子轨迹拓扑
figs/
  gpcr_*.png | gpcr_*.pdf    # 12 组分层图
summary/
  gpcr_conclusion.md    # 五节中文结论
```

`analysis_<task>.json` 覆盖 segments、microswitches、tm6_opening、ligand_retention、landscape、uncertainty、dccm_mi、psn、leadlag、water_network、plumed_parse、plumed_fes、compare 各层结果。

子轨迹 `analysis_sub.*` 只含蛋白与配体：9 万原子的全体系切到 5 千原子，后面所有分析都读它，快一个量级。

准备任务：

```
prep_config.json        # 力场 / 水模型 / 盒型 / 离子 / escalate 等配置
prep_log.json           # 逐步执行记录与告警
decision_log.json       # 判断层账本
protein_decided.pdb     # 判断层改写残基名后的蛋白（供 pdb2gmx）
topol.top
complex.gro
box.gro
solv.gro
ionized.gro
step5_1.tpr             # 可直接喂给 run_gromacs_md
summary/
  prep_review.md        # 审查报告
  decisions.md          # 判断账本
```

## 7. 安装

### 7.1 系统依赖

- GROMACS（`gmx`）。只在 GROMACS 2026.2 上验证过。
- gmx_MMPBSA（可选，`run_mmpbsa` 算 MM/GBSA 结合能时用到）。
- 配体赋参另需一个带 networkx 的 Python 与 CGenFF 转换器；走 GAFF 路线还要 acpype / AmberTools。本地膜插入按晶格摆放，不需要额外可执行文件。

### 7.2 Python 环境

```
git clone https://github.com/RKJ6901/gromacs-md-agent.git
cd gromacs-md-agent
conda env create -f environment.yml
conda activate md-agent
cp .env.example .env
```

要两个 env，不是因为洁癖：agent venv 只装 langchain / langgraph / gradio / mcp，跑图和界面；MDAnalysis / scipy / sklearn 这些重依赖单独放在分析 env 里。工具层通过子进程调 `worker.py`（用 `MD_ANALYSIS_PY` 指定的解释器），agent venv 不必背这些库。

### 7.3 力场文件

CHARMM36 的 GROMACS 移植版来自官方 jul2022 tgz，需要自己取来解压到 `<仓库根>/forcefields/charmm36-jul2022.ff/`。这个目录不在仓库里。工具跑 gmx 时会把 `GMXLIB` 指向 `<仓库根>/forcefields/`——实测这是追加而非替换，GROMACS 自带的力场仍能找到。

### 7.4 配置

复制 `.env.example` 为 `.env`，把下面这些具体路径填进去（代码里不硬编码）：

```
# LLM（OpenAI 兼容接口）
LLM_MODEL=grok-4-1-fast-non-reasoning
LLM_API_KEY=...
LLM_BASE_URL=https://api.shubiaobiao.cn/v1

# GROMACS / gmx_MMPBSA 可执行文件
GMX_BIN=/opt/gromacs/bin/gmx
GMX_MMPBSA_BIN=/opt/gromacs/bin/gmx_MMPBSA

# 重分析用的 python（装了 MDAnalysis / scipy / sklearn）
MD_ANALYSIS_PY=/path/to/analysis-env/bin/python

# 配体赋参（只有体系准备需要）
PREP_CONVERTER_PY=/opt/conda/bin/python
PREP_CGENFF_CONVERTER=/data/opt-bk/opt/PyAutoFEP/Tools/cgenff_charmm2gmx_py3.py
PREP_ACPYPE=/opt/conda/envs/ambertools/bin/acpype
PREP_CONDA=/opt/conda/bin/conda
```

## 8. 使用

### 8.1 Web 控制台

```
python mdagent/agent.py
```

默认监听 http://0.0.0.0:7777（被占用时用 `--port` 换一个）。三个标签页：「运行模拟」「分析已有轨迹」「准备体系」。

### 8.2 Chainlit 前端

```
chainlit run mdagent/chainlit_app.py --host 0.0.0.0 --port 7779
```

另一套界面：复用同一张图，把每个图节点渲染成一个可展开的步骤；设置面板在左下角。

### 8.3 CLI

```
# 跑 MD 模拟 + 基础分析
python mdagent/agent.py --mode cli --tpr step5_1.tpr --nsteps 5000000 --run-name md

# 只对已有轨迹做 GPCR 分层分析（就地分析，不新建任务目录）
python mdagent/agent.py --mode cli --analyze-only \
    --ws /data2/JRK/md_workspaces/gpcr_test --traj md.xtc --run-name apo

# 可选：A/B 对比（第二个体系必须已经跑过同样的分层分析）
python mdagent/agent.py --mode cli --analyze-only --ws /path/A --ws-b /path/B
```

批量提交用 `run_cli.sh`（环境变量 `TPR` / `NSTEPS` / `RUN` / `TRAJ` 覆盖默认值）：

```
bash run_cli.sh
```

### 8.4 MCP

`mdagent/mcp_server.py` 把 49 个原子工具 + `create_task` / `list_tasks` / `set_workspace` 三个管理工具暴露给 Claude Desktop / Cursor 等 MCP 客户端：

```json
{"mcpServers": {"md-agent": {
  "command": "/path/to/venv/bin/python",
  "args": ["/abs/path/to/mdagent/mcp_server.py"]}}}
```

---

## 9. 排错

| 现象 | 排查方向 |
|---|---|
| `Residue 'HSD' not found in residue topology database` | 残基名与目标力场家族不匹配（CHARMM 的 HSD vs Amber 的 HID）。准备阶段会按力场自动改名；手工跑 `pdb2gmx` 时先确认残基名。 |
| `Invalid order for directive atomtypes` | 拼 `topol.top` 时 `atomtypes` 没拆到独立文件。GROMACS 要求该指令在最前，工具会把它拆出来；手工拼时注意顺序。 |
| `Maximum force = inf` | 能量最小化时原子硬重叠，多半是本地插入的膜与蛋白贴太近。判断层会报最小脂质–蛋白距离；低于 0.30 nm 的下限要重铺。 |
| `Cannot properly run sqm` | acpype 直接调、缺 AmberTools 的环境变量而秒退。改成在 AmberTools 环境里用 `conda run` 调 acpype。 |
| `no 'obabel' executable` | acpype 依赖的 obabel 不在 PATH。把 acpype 所在 bin 目录加进 PATH 再调。 |
| `Topology include file ... not found` | GROMACS 找不到被 include 的力场 / 拓扑文件。确认 CHARMM36 移植版已解压到 `<仓库根>/forcefields/`，且 `GMXLIB` 指向它。 |
| FES 数值离谱（到几百甚至上千 kJ/mol） | 把 `plumed sum_hills` 1D 输出第 3 列的导数当成了自由能。按列名 `file.free` 取、不按位置取；欠采样区的尾部本来就会虚高，分析要限制在已采样窗口内。 |
| MDAnalysis 打不开 `.tpr`（tpx version 太新） | GROMACS 2026 的 tpr 版本超出 MDAnalysis 识别范围。拓扑走 `.gro` / `.pdb`，坐标走 `.xtc`；`gmx trjconv -pbc mol` 仍需 tpr，两者并存。 |
| 报告里出现上一轮的 A/B 数字 | 工作区被 `cp -r` 复制、带进旧的 `analysis_compare.json` / `compare_metrics.csv`。跳过 A/B 时节点会主动清掉这两个旧产物；复用工作区别用 `cp -r`。 |

## 10. 已知限制

- CGenFF 配体赋参是半自动的：要在 `app.cgenff.com` 手工上传配体取回 `.str`，流程在缺 `.str` 时停下等人工，没有免费的全自动接口。这条路线目前尚未在真实配体上完整跑通。
- 本地插入的膜是晶格几何摆放的初始结构，未做平衡。实测出现过最小脂质–蛋白距离 2.08 Å、低于 3.0 Å 下限的情况；判断层会把这个报出来，但不会自动修好。
- 判断层的真值校验只做了 7W0M 一个体系（CHARMM-GUI 的净电荷 +7、二硫键 Cys102–Cys181、组氨酸全 HSD）。
- 只在 GROMACS 2026.2 上验证过。
- 等待模拟完成的轮询上限是 30 秒 × 120 次（约 1 小时），更长的模拟需要手动重试。
- 断点续跑依赖 MemorySaver，进程重启后不保留。
- 没有自动化测试套件；`tests/` 下是端到端脚本，需要真实数据才能跑。

## 11. 致谢

- [GROMACS](https://www.gromacs.org) —— 模拟引擎。
- [CHARMM36](http://mackerell.umaryland.edu/charmm_ff.shtml)（Mackerell lab 的 GROMACS 移植版）—— 蛋白 / 脂质力场。
- [gmx_MMPBSA](https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA) —— MM/GBSA 结合能。
- [PLUMED](https://www.plumed.org) —— 元动力学与 FES。
- [MDAnalysis](https://www.mdanalysis.org) —— 轨迹分析。
- [CGenFF / cgenff_charmm2gmx](https://cgenff.com) —— 配体 CGenFF 赋参。
- [ACPYPE](https://github.com/alanwilter/acpype) 与 [AmberTools](https://ambermd.org/AmberTools.php) —— GAFF 赋参。
- [MCP](https://modelcontextprotocol.io) —— 把工具暴露给外部 AI 应用。
- [LangChain](https://github.com/langchain-ai/langchain) / [LangGraph](https://github.com/langchain-ai/langgraph) —— 编排层。
- [Gradio](https://www.gradio.app) / [Chainlit](https://github.com/Chainlit/chainlit) —— 两套前端。
- [rosetta-agent-pipeline](https://github.com/clearlove-seven1/rosetta-agent-pipeline) —— 本项目迁移自它的骨架。
