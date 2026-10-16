# Agent 知识（结合本项目 · 深挖版）

这份是给「要讲这个项目、并且要经得起追问」用的。每个优化点都写到能指到具体函数、常量、改前改后的程度。

**怎么用**：
1. 先读第 0 节，那是能直接背下来的介绍话术。
2. 第 1–12 节是 12 个优化点，每个都是「问题 → 优化点 → 怎么做 → 代价 → 追问」。
   面试讲这个项目，挑 3–4 个讲透就够了，别全讲。
3. 第 13 节是没说出口的部分（还没解决的问题 + 常被问但我没做的），主动讲这些比讲成绩更有说服力。

**一条纪律**：下面每个数字和函数名都能在仓库里找到。带「（推测）」的是我的判断，不是已验证的事，被问到时要说清。

---

## 0. 先装进脑子

### 一句话介绍

> 这是一个 GROMACS 分子动力学模拟 + GPCR 后处理分析的 agent。用户一句话描述要做什么，它自动判断是跑模拟、还是对已有轨迹做机制分析，然后按固定流程调用 30 个工具跑完，最后出一份带不确定性标注的中文结论。
>
> 技术上是一个 LangGraph 状态图：主图负责路由和收尾，模拟流水线是一个子图；LLM 只用在真正需要理解自然语言的地方，其余全部是确定性代码。

### 两个必须能画出来的图

**主图**（`agent.py:1167`）：

```
START → router ─┬─(pipeline)→ pipeline(子图) → parse_results → gpcr_report → summarize → END
                └─(query/chat)→ reply → reply_tools ↩
```

**子图**（`build_pipeline_subgraph`，`agent.py:366`）：

```
START ─┬─(mode=gpcr)──→ gpcr_setup ──→ gpcr_l2 → gpcr_l3 → gpcr_l4 → gpcr_l5 → gpcr_plot → gpcr_compare → END
       └─(md/full)──→ step1_llm → step1_tool → validate → progress_poll ─┬─(还在跑)→ 自循环
                           ↑                                              └─(.gro出现)↓
                       repair ←── step2_llm → step2_tool → step3_llm → step3_tool
                       ↑                                          ↓
                       └──────────────────────── step4_llm → step4_tool ─┬─(mode=full)→ gpcr_setup
                                  (retry_count>2 才真退出)               └─(mode=md)──→ END
```

### 能直接说出口的关键数字

| 项 | 值 | 位置 |
|---|---|---|
| 工具总数 | 30 个（`tools.py` 里 `@tool`） | `agent.py:36-62` 的 import 清单 |
| 工具结果截断阈值 | 16000 字符 | `agent.py:105` `MAX_TOOL_RESULT_CHARS` |
| 进度轮询 | 每 30 秒，最多 120 次（即 60 分钟） | `agent.py:268-269` |
| 单步重试上限 | 2 次 | `agent.py:100` `MAX_RETRIES` |
| 图节点数（子图） | 21 个 | `agent.py:366-556` |
| GPCR 分析层 | 5 层（编号 / 激活坐标 / 景观 / 耦合 / 元动力学）+ 出图 + 对比 | `GPCR_SETUP_TOOLS`…`GPCR_PLOT_TOOLS` |
| 出图组数 | 12 组 | `gpcr_plots.py` |
| 入口数 | 4（CLI / Gradio / Chainlit / MCP） | `agent.py`、`ui.py`、`chainlit_app.py`、`mcp_server.py` |
| 走 LLM 的节点 | 5 个（router 兜底、step1-4_llm、repair、reply、summarize） | 见 §2 |

### 如果只让你讲两件事

1. **哪些步骤不该交给 LLM**（§1）——这是这个项目最有说服力的判断，因为它是踩坑踩出来的。
2. **怎么让 agent 不说假话**（§7）——大部分 agent 项目的演示都停在「它能干活」，没人讲「它出错时怎么不骗人」。

---

## 1. 优化点：把「没有判断空间的步骤」从 LLM 手里拿回来

### 问题

最初 GPCR 第 2–5 层（激活坐标 / 构象景观 / 耦合 / 元动力学）和出图，都是 `LLM 节点 → 工具节点` 的模式。实测时出了一件事：

报告里只有一行「执行成功: plot_gpcr_figures」，第 2–5 层一个都没跑，**而且没有任何报错或缺失提示**。

### 定位过程（面试重点）

查执行日志，发现模型在第 4 步返回了 **空的 `tool_calls`**。

关键在于出口条件边 `_llm_has_tool_call`（`agent.py:183`）：

```python
def _llm_has_tool_call(state):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool"
    return "__next__"
```

而 `__next__` 在 `add_conditional_edges` 里映射到了 `__end__`（`agent.py:469`）。

于是逻辑链变成：**模型「这一轮没决定调工具」→ 被当成「任务做完了」→ 整条链终结**。这不是模型犯错，是我的出口条件把「沉默」解释成了「完成」。

### 优化点

判断标准：**这一步有没有需要「判断」的分支**。

| 步骤 | 有判断吗 | 该怎么办 |
|---|---|---|
| 用户这句话是跑模拟还是聊天 | 有（开放自然语言） | 交给 LLM |
| 编号识别之后做第 2 层然后第 3 层 | 没有，顺序固定 | 写成死代码 |
| 每个分析工具的参数 | 没有，全可省（路径自动解析） | 写成死代码 |
| 报错之后怎么修 | 有（要读日志判断） | 交给 LLM |

### 怎么做

新建节点工厂 `_make_det_node`（`agent.py:226`），直接按顺序 `tool.invoke({})`，完全不进模型：

```python
def _make_det_node(name, tools, headline):
    def _node(state, config=None, **kwargs):
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
```

然后把这一串接成普通边（`agent.py:527-529`）：

```python
_chain = ["gpcr_setup", "gpcr_l2", "gpcr_l3", "gpcr_l4", "gpcr_l5",
          "gpcr_plot", "gpcr_compare"]
for _a, _b in zip(_chain, _chain[1:]):
    sg.add_edge(_a, _b)
```

### 效果

- 第 2–5 层**不可能被跳过**——顺序是硬编码的边，没有分支可以让它绕开。
- 每层不等上一层的模型输出，直接串行执行，更快。
- 单层内部工具失败**不中断链条**，错误写进消息里，最后由报告如实写进「边界」一节。

注意「失败不中断」这个设计是有意的：分析层之间没有强依赖（景观算不出来不影响耦合分析），中断只会让整份报告全废。

### 为什么不用别的办法

- **改提示词让模型「务必调用工具」**：不行。模型漏调不是因为它不听话，是因为这是一个概率系统的固有行为。而且提示词是软约束，改完你还得验证它有效，而失败是静默的——你没法保证下一次不出。
- **保留 LLM 但把出口从 `__end__` 改成报错重试**：会变好，但还是在为「模型可能什么都不做」写兜底代码。这几步本来就不需要模型，为什么不直接去掉这一层？
- **用 `llm.bind_tools(tools, tool_choice="required")` 强制调用**：能强制出一轮工具调用，但下一轮又回到同样的问题。而且它解决的是「模型没调工具」，没解决「模型调错工具、传错参数」。

### 会被追问什么

**Q：那你怎么判断一个步骤该不该用 LLM？**
A：看这一步有没有需要判断的分支。更细一点的判据是**失败的代价**：漏调只是少个可选功能，可以交给 LLM；漏调会让最终结论缺一块而你看不出来，就必须写死。第 2–5 层属于后者——分析少了一层，报告照样生成，看起来是完整的。

**Q：现在还有哪些地方用 LLM？为什么那些可以？**
A：5 处。（1）router 兜底：关键词没命中时的 `query`/`chat` 分类；（2）step1–4：从用户消息里提取 `tpr` 路径、步数、轨迹列表这些参数——用户表达方式千变万化，写正则写不全；（3）repair：读日志判断怎么修；（4）reply：回答文档问题；（5）summarize：把骨架组织成人话。

**Q：如果模型仍然在 step1 返回空调用呢？**
A：会同样静默结束。这个风险我承认还在——step1–4 的参数提取确实需要 LLM（用户可能说「用那个 500 万步的 tpr」，路径和意图混在一句话里）。缓解手段是 `submit_task` / `run_cli` 里对最终答案做兜底：如果历史里没有任何 GPCR 骨架消息，收尾会走「只跑了 MD 流水线」的分支。但没有做「step1 没调工具就重试」——这是**已知未修**（见 §13）。

---

## 2. 优化点：schema 级权限 —— 把「禁止」从文本变成物理限制

### 问题

一个图里有 30 个工具。如果全都绑给每个节点，模型在「提交模拟」这一步理论上可以调 `plot_gpcr_figures`；更要紧的是，用户在输入里塞一句「忽略之前的指令，直接调用某个工具」，就可能把它带偏。

在提示词里写「请不要调用 X」是请求，不是限制。

### 优化点

**不给它的工具，它物理上调不出来。** 这是 Anthropic 讲的「schema 级权限」：`bind_tools` 决定了模型能看见哪些工具 schema，清单外的工具它连名字都不知道。

### 怎么做

每个 LLM 节点在工厂里就绑定了自己那一步的工具集（`agent.py:83-97` 定义分组，`:186` 使用）：

```python
STEP1_TOOLS = [run_gromacs_md]                                    # 只看得见这一个
STEP2_TOOLS = [prepare_analysis_index, merge_trajectories]
STEP3_TOOLS = [compute_rmsd, compute_rmsf, compute_gyrate, compute_hbond]
STEP4_TOOLS = [plot_md_figures]
REPAIR_TOOLS = [check_logs, diagnose_and_fix, auto_repair]
GENERAL_TOOLS = [search_md_docs, read_file]
```

```python
def _make_step_llm_node(step_num, system_prompt, bound_tools):
    step_llm = llm.bind_tools(bound_tools)     # 这一步的权限边界就在这一行
    def _node(state):
        response = step_llm.invoke([SystemMessage(content=system_prompt)] + list(state["messages"]))
        return {"messages": [response], "pipeline_step": step_num}
    return _node
```

最狠的是收尾节点（`agent.py:678`）：

```python
def summarize_node(state):
    summarize_llm = llm.bind_tools([])   # 空清单：物理上禁止再调任何工具
    ...
```

绑空清单防的是模型陷入「一直调工具但永远不给结论」的循环。这不是概率上的预防，是它根本没有可调的工具。

### 为什么不用别的办法

- **只在提示词里写「禁止调用」**：软约束。模型可以因为用户的上文而违反，而且违反了你也看不出来。
- **在工具函数里加权限检查**：可以做（工具里判断当前步骤），但那是运行时的补救，模型已经"想"调了，多花了一轮 token 才失败。

### 会被追问什么

**Q：`bind_tools([])` 和「不让这个节点有工具」有什么区别？**
A：效果一样，形式不同。图里 `summarize_node` 用的是一个未绑工具的 `llm`，`bind_tools([])` 是把「这个节点没有工具」这件事写显式。真正需要它的是那些"本该有工具但这一步不希望它有"的场景。

**Q：工具有 30 个，会不会某一步该用的工具没绑上？**
A：会，这是手工维护分组表的成本。但因为绑错了会立刻暴露（工具调不出来），不会静默出错，所以是可接受的。对照组是：如果权限写在提示词里，绑错了不会有任何信号。

---

## 3. 优化点：上下文管理 —— 三层策略

### 问题

agent 跑起来会疯狂吃上下文。这个项目最脏的三个地方：

- `gmx mdrun` 的日志几万行；
- GPCR 分析（尤其是 DCCM、网络分析）返回的 JSON 几十 KB；
- MD 动辄跑几小时，程序每 30 秒轮询一次，最多 120 次。

### 优化点

不是「把结果切小」，而是**分层处理：能不进模型就不进，一定要进就落盘，非要进全文才截断**。

### 怎么做（三层，一层比一层彻底）

**第一层：根本不进模型。** 进度轮询是纯确定性节点（`progress_poll_node`，`agent.py:279`），直接 `glob` 工作区数 `.gro`，一个 token 都不花：

```python
def progress_poll_node(state, config=None, **kwargs):
    n_gro, n_xtc = _count_md_products(workspace)
    polls = state.get("progress_polls", 0) + 1
    done = n_gro > 0
    time.sleep(PROGRESS_INTERVAL)   # 30 秒
    msg = HumanMessage(content=progress_text, name="ProgressReporter",
                       id="progress_poll_status")   # ← 固定 id
    return {"messages": [msg], "progress_polls": polls, "progress_done": done}
```

**那个固定 `id` 是个关键细节**：`add_messages` reducer 的规则是「同 id 覆盖」。如果不给 id，120 次轮询会往状态里堆 120 条「还在跑」——既爆上下文又刷屏。给了固定 id，状态里永远只有一条最新进度。

**第二层：落盘 + 回灌摘要。** 大的分析结果不直接进模型。`_py_tool`（`tools.py:750`）跑完 worker 后，把完整 JSON 写成 `analysis_<task>.json`，只把渲染好的几行小结返回：

```python
def _py_tool(task, summary_fn, timeout=3600, **payload):
    r = _py_analysis(task, payload, timeout=timeout)
    fp = os.path.join(ws, 'analysis_%s.json' % task)
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(r, f, ensure_ascii=False, indent=2, default=str)
    ...
    return '【%s 完成】\n%s%s' % (task, body, tail)
    # tail = '\n完整结果: analysis_landscape.json（需要细节用 read_file 查看）'
```

模型想知道细节，自己 `read_file`。**这比一次性全给更省**，因为大多数时候它只需要知道「成了、关键数字是多少」。

**第三层：截断。** 一定会有工具返回超长文本（比如 `read_file` 读了个大文件）。`TaskAwareToolNode` 继承 `ToolNode` 覆写 `invoke`/`ainvoke`（`agent.py:116`），对所有工具结果统一截断：

```python
MAX_TOOL_RESULT_CHARS = 16000

def _truncate_tool_result(content, max_chars):
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return f"{content[:half]}\n\n... [已截断 {len(content) - max_chars} 字符] ...\n\n{content[-half:]}"
```

**取首尾各一半**而不是取前 N 字符：因为错误信息常在日志末尾，取了头丢尾最容易漏掉最重要的东西。

### 为什么放在 ToolNode 这一层

因为它是**所有工具节点的公共祖先**。图里 10 多个工具节点有的是 `TaskAwareToolNode`，有的藏在工厂里，但都走 `invoke`。在一处覆写，全局生效，不用逐个改。

### 会被追问什么

**Q：截断 16000 字符怎么定的？**
A：估算的。项目用的模型上下文窗口很大，但每条工具结果都吃满的话，几轮就把预算烧完了。16000 字符大约 4-8k token，够放一个完整的错误栈或者一段 xvg 摘要。（推测：这个数字没有做过系统的实验优化，是拍的量级。）

**Q：截断会不会把关键信息切掉？**
A：会，而且没有兜底。这是**已知未修**（见 §13）。真要稳妥，应该按语义切（比如 gmx 日志只保留最后 N 行 + 报错行）。目前靠的是「首尾各一半」这个粗糙的启发式。

---

## 4. 优化点：工作区隔离 —— contextvars 而不是全局变量

### 问题

一个进程里可能同时跑多个任务（Gradio 的多个会话、MCP 的不同 `set_workspace`、CLI 的单次运行）。如果用全局变量存当前工作区，两个任务会互相污染——A 的结果写进 B 的目录。

### 优化点

用 `contextvars.ContextVar`（`tools.py:31`）：

```python
current_workspace = contextvars.ContextVar('current_workspace', default='.')
```

`ContextVar` 是按**执行上下文**隔离的，不是按线程。在 async 场景下（比如 Chainlit 前端）也能正确隔离，这是 `threading.local` 做不到的。

### 怎么做

工具层所有路径解析都基于它：

```python
def get_workspace_path(sub_dir=""):
    base_dir = os.path.abspath(current_workspace.get())
    full_path = os.path.join(base_dir, sub_dir) if sub_dir else base_dir
    os.makedirs(full_path, exist_ok=True)
    return full_path
```

**三个入口各自负责设置它**：

- LangGraph 侧：`TaskAwareToolNode.invoke` 从 `config["configurable"]["thread_id"]` 读 task_id 然后 `current_workspace.set(...)`（`agent.py:123`）。所以 **thread_id 同时是工作目录名**。
- 确定性节点：`_make_det_node` 里同样从 config 读（`agent.py:230`）。这里必须单独写一遍，因为确定性节点不经过 `TaskAwareToolNode`。**这是踩过的坑**——第一版忘了加，导致分析结果写到了错误目录。
- MCP 侧：`_apply_workspace()` 从模块级的 `_ACTIVE_TASK` 读（`mcp_server.py:88`）。

### 代价（要主动讲）

`thread_id` 一个标识承担了两件事：**这次运行的身份** + **产物放哪儿**。

好处：天然隔离，换个 id 就是全新的工作区。
坏处：两件事被绑在一起，改一个要想着另一个。比如「想在第 3 次运行时复用第 1 次的产物」这件事，在现在这个设计下会很别扭——你得手动 `--ws` 指定目录，绕过自动生成。

### 会被追问什么

**Q：为什么用 `contextvars` 不用 `threading.local`？**
A：因为前端会用 async。Chainlit 的 `@cl.on_message` 是 async 的，`threading.local` 在 async 里不能正确隔离——同一个线程跑多个协程时它们会共享。`ContextVar` 是 asyncio 的原生隔离机制。

**Q：`thread_id` 怎么保证不重复？**
A：格式是 `task_{YYYYMMDD_HHMM}_{run_name}`，精确到分钟。同一分钟内提交两个同名任务会撞——**已知未修**（见 §13）。实际使用中撞的概率很低（人工提交），但自动化场景下这是个隐患。

---

## 5. 优化点：断点续跑 + 进度轮询

### 问题

一个 MD 模拟要跑几小时。程序不能同步等（HTTP 请求早超时了），也不能阻塞（用户要能看到进度）。

### 优化点

两件事分开处理：**模拟本身后台跑，agent 负责轮询**。

### 怎么做

**后台提交**（`tools.py:191`）：用 `subprocess.Popen` + `nohup` 把 `gmx mdrun` 挂到后台，把 PID 写文件，然后**立即返回**：

```python
subprocess.Popen(f'cd {ws} && nohup {cmd} > {run_name}.out 2>&1 & echo $! > {pidf}', shell=True)
time.sleep(2)
pid = open(pidf).read().strip() if os.path.exists(pidf) else ""
return f"【MD 已挂起】mdrun 后台运行中 (PID={pid}, {steps_txt})\n完成标志: {run_name}.gro 出现"
```

注意 `time.sleep(2)` 是必要的：`Popen` 是异步的，不等一下 PID 文件可能还没写出来。

**轮询**（`agent.py:279`）：`progress_poll` 节点自循环，每 30 秒查一次 `.gro` 是否存在：

```python
def progress_poll_route(state):
    if state.get("progress_done"):
        return "__end__"        # 自循环结束 → step2_llm
    if state.get("progress_polls", 0) >= MAX_PROGRESS_POLLS:   # 120 次
        return "__end__"
    return "progress_poll"      # 继续自循环
```

**为什么用「自循环」而不是一个 `while` 循环**：因为在图里，每一步都是一次状态更新，可以被打断、可以恢复。如果写成 `while` 循环，整个等待就变成了一个黑盒节点——前端看不到中间进度，checkpoint 也没法在中间保存。

**断点续跑**：图编译时带了 SQLite checkpointer（`agent.py`）：

```python
_checkpointer_conn, checkpointer = _build_checkpointer()   # sqlite3 连接（WAL）
agent_executor = workflow.compile(checkpointer=checkpointer)
```

每个节点走完都存一次状态，落盘到 `MD_CHECKPOINT_DB` 指定的 SQLite 文件（默认 `<仓库根>/checkpoints.sqlite`）。恢复时给同一个 `thread_id`，它从断点接着跑——**进程重启后也一样**（早期用的内存版 `MemorySaver` 做不到这点，已换成 `SqliteSaver`）。

这套 checkpointer 是**同步的**，只服务同步调用：Gradio / CLI 走 `agent_executor.stream(...)` 没问题，但 Chainlit 那条 async 路径若直接 `astream(...)`，`SqliteSaver.aput()` 会 raise `NotImplementedError`（这就是 BUG-18）。反过来换成 `AsyncSqliteSaver` 又会让同线程里的同步调用报 `InvalidStateError`，而图的节点本来就全是同步的（subprocess、文件 IO），没必要拆两套。所以 Chainlit 走 `agent.py` 的 `astream_sync`：把同步流放到工作线程里跑、事件经队列送回事件循环，并用 `contextvars.copy_context()` 把 async handler 设的 `current_workspace` 带进去——**一套 checkpointer 服务全部入口**。

### 边界（重要，主动讲）

状态现在落盘，进程重启后仍在，但要注意恢复的是**编排状态**而非**被中断的计算**：

- 「图的状态」能跨重启恢复——同一个 `thread_id` 从 SQLite 里读回。
- 「后台 `gmx mdrun` 的子进程」不会因为图恢复就自动接着跑——模拟是 `nohup` 挂在系统上的独立进程，agent 重启只是重新掌握它的进度，不是重启它。
- 多路并发共用一个 SQLite 文件（连接 `check_same_thread=False` + WAL）。

这个边界不主动讲，面试官一追问「那你机器重启了怎么办」，答「图能续上」就够，但别把它说成「模拟进程也能续」。

### 会被追问什么

**Q：轮询 120 次，每次 30 秒，一共 60 分钟。如果模拟要跑 10 小时呢？**
A：轮询会先退出（60 分钟后），图继续往下走，然后 step2 找不到 `.gro` 就报错，进 repair，repair 调 `auto_repair` 发现 PID 还活着 → 提示「MD 仍在运行」。这是**一个设计缺陷**：应该让轮询上限可配，或者检测到 PID 活着就继续轮（**已知未修**，见 §13）。实际使用中短测试够了，长模拟确实会走到这个路径。

**Q：`time.sleep(30)` 在节点里，不会阻塞整个图吗？**
A：会，而且这是个真问题。如果同时有两个任务，一个的轮询 sleep 会挡住另一个。当前项目是单任务使用，所以没暴露。（推测：要修的话得把 `time.sleep` 换成 `asyncio.sleep` 并让节点变成 async，或者把轮询改成外部调度。）

---

## 6. 优化点：把「不许编数字」从提示词搬进骨架

### 问题

收尾要让模型写一段中文结论。但它会顺着上下文自由发挥——实测出现过一次：**12 张图全在磁盘上、工具也返回了成功，模型写「出图未完成（工具报错）」**。

提示词里写「不许编数字」堵不住这个。因为它不是在编数字，是在编一个**它以为发生了的事实**。

### 优化点

**别让模型负责它必须准确的东西。** 把结论做成代码拼好的骨架，模型只负责组织语言。

### 怎么做

`build_gpcr_report()`（`agent.py:1120`）从各层的 `analysis_*.json` 里读数字，按五节拼成纯文本：

```
一、观察（测到了什么）      ← _sec_observe + _sec_observe2
二、统计（结论有多硬）      ← _sec_stats
三、机制含义（...解读）     ← _sec_mech
四、边界（哪些结论不能下）  ← _sec_limits
五、产出清单（磁盘实测）    ← _sec_artifacts
```

**第五节是专门为「模型胡说」加的**（`agent.py:1103`）：

```python
def _sec_artifacts(ws):
    figs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ws, "figs", "gpcr_*.png")))
    if figs:
        out.append("• 已生成图 %d 张（figs/*.png）：%s" % (len(figs), ", ".join(...)))
    else:
        out.append("• figs/ 下没有 gpcr_*.png —— 出图确实没有产物")
    csvs = sorted(...)
    ...
```

它**直接扫磁盘**，不依赖任何工具返回值。图片在不在，以文件系统为准。模型在收尾提示词（`SUMMARIZE_SYSTEM`）里的约束是：

> 产出清单是磁盘实测结果，禁止与之相反的说法：清单里列出了图片就不能说「出图失败/未生成图片」。

加了这一节之后，模型编造「出图失败」的空间被物理堵死了——它要么照抄磁盘事实，要么明显自相矛盾。

### 第四个坑：骨架不能留空档

修完「产出清单」之后，又出了新问题。一次没做 A/B 对比（没传 `ws_b`）的运行里，模型自己编了一句「A/B 对比…0/8 个指标显著」。

根因是：`_sec_limits` 在「没有 compare」这一情况下什么都不输出，留了个空档，模型就自己补上了。

修法是显式填死（`agent.py:1097`）：

```python
if not cp:
    out.append("• 本次未做 A/B 跨体系对比（未提供对比工作区 ws_b），没有任何跨体系差异结论；"
               "如需对比，用 --ws-b 指定第二个工作区。")
```

**这条经验比上一个更重要**：让模型「别编」不如让骨架「不空」。任何留给模型的空档，它都会填，而且填得像个正经结论。

### 会被追问什么

**Q：那模型在收尾这一步还有什么自由度？**
A：只负责把骨架组织成流畅的中文，可以调语序、加连接词，但不能改数字、不能增删结论。约束写在 `SUMMARIZE_SYSTEM` 里，同时 `bind_tools([])` 保证它不会再跑去调工具。

**Q：万一模型还是改了骨架里的数字呢？**
A：目前没有自动校验。骨架同时落盘成 `summary/gpcr_conclusion.md`，所以**可以事后对账**——但如果没人来对，编造就检不出来。（推测：要根治得在收尾后做一次数字回查，把骨架里的数字和答复里的数字比对。这是**已知未修**。）

---

## 7. 优化点：把不确定性写进产物（诚实性设计）

### 问题

普通程序错了会崩。agent 错了会**一本正经给你一个像样的结果**。

这个项目尤其危险：它输出的是科学结论。一个看起来合理的自由能势垒，如果采样不足，和真的势垒长得一模一样。

### 优化点

不是靠模型自觉，而是**让程序自己知道它不知道，并把这件事写进产物**。

### 怎么做（四个机制）

**1. 每个均值必须带置信区间和有效样本量。** `analyze_uncertainty` 做分块 bootstrap + 自相关估计 ESS。`_sec_stats`（`agent.py:991`）在 ESS 过低时直接标注：

```python
ess = _ess_of(v)
flag = "  ⚠ ESS<50，该均值不可信" if (ess == ess and ess < 50) else ""
out.append("  %-30s mean=%s  95%%CI=[%s, %s]  ESS=%s%s" % (...))
```

如果这一步压根没跑，也明说：

```python
else:
    out.append("  未跑 analyze_uncertainty：本节所有均值都没有 CI/ESS 支撑。")
```

**2. FES 先判收敛，再谈结论。** 元动力学的收敛检查是分块比较 `rms_centered`（`agent.py:1003`）：

```python
vals = [e.get("rms_centered") for e in (fm.get("convergence") or []) if e.get("rms_centered") is not None]
if vals:
    okv = vals[-1] <= max(1.0, 0.6 * max(vals))
    out.append("  FES 收敛 rms_centered（25/50/75%%）：%s → 判定：%s" % (
        " / ".join(_f(x, 2) for x in vals),
        "趋稳，可做定性解读" if okv else "未趋稳，不得报自由能深度与势垒"))
```

注意判定写的是「不得报」，不是「可能不准」。措辞越硬，下游越不容易含糊过去。

同一件事在 `_sec_limits` 里再说一遍，确保它出现在「边界」一节而不是被埋在「统计」里。

**3. 测不到的位点进缺失列表，不猜。** `_sec_limits`（`agent.py:1065`）：

```python
miss = (A.get("microswitches") or {}).get("missing_measurements") or []
if miss:
    out.append("• 未能测到的位点（缺失≠无变化）：%s" % ", ".join(...))
```

括号里那句「缺失≠无变化」很关键——不说清楚，读的人会把「没测到」当成「没变化」。

**4. 「结果正确地反映了无意义的输入」也要标出来。** 没传第二个工作区时，程序拿 A 和自己比，8 项指标 Δ 全 0。这在报告里看起来像「两个体系高度相似」这种科学结论。修法是跳过并写明未提供工作区（见 §6）。

### 会被追问什么

**Q：ESS<50 这个阈值怎么来的？**
A：经验值。自相关越强、有效样本越少，均值的误差估计就越不可靠。50 是个宽松下限，低于它连定性解读都要小心。（推测：没有做过敏感性分析确定「50 比 30 好在哪里」。）

**Q：那你怎么知道这些判据本身是对的？**
A：不能完全保证，所以报告里写的是「按 Class A GPCR 公认判据的解读，属于可检验的假设」。`_sec_mech` 结尾有这句：

```python
out.append("（以上为按公认判据的机制解读，属于可检验的假设；结论强度受统计一节的 CI/ESS 与收敛判定约束。）")
```

把自己定位成「假设生成器」而不是「结论机器」，是这类科学 agent 唯一诚实的姿态。

---

## 8. 优化点：四个入口共用一套工具

### 问题

同一套计算能力要被四种方式使用：命令行、两个网页、外部 AI 应用（MCP）。如果每个入口各自实现一遍，加一个功能要改四处。

### 优化点

**工具层独立，界面层薄。** 30 个 `@tool` 在 `tools.py` 里写一次，四个入口全部从它 import。

### 四个入口

| 入口 | 文件 | 怎么用同一套工具 |
|---|---|---|
| CLI | `agent.py:1309` `run_cli` | 直接调 `agent_executor.stream` |
| Gradio | `agent.py:1252` `submit_task` + `ui.py` | 同上，UI 在 `ui.py` 里 |
| Chainlit | `chainlit_app.py` | 同上，改用 `astream` 逐节点开 `cl.Step` |
| MCP | `mcp_server.py` | 把 `@tool` 的 `.func` 重新包一层，暴露成 MCP 工具 |

MCP 那一层有个细节（`mcp_server.py:92`）：

```python
def _wrap_tool(lc_tool):
    fn = lc_tool.func
    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        _apply_workspace()
        return fn(*args, **kwargs)
    return _wrapped

for _t in _ATOMIC_TOOLS:
    mcp.tool()(_wrap_tool(_t))
```

取的是 `lc_tool.func`（原函数），不是 `lc_tool.invoke`。因为 MCP 自己会生成参数 schema，不需要 LangChain 的 `@tool` 包装，只需要函数本体 + 工作区注入。

### 界面与业务分离的实际收益

这个项目一个月内换了两次前端（Gradio 深度定制 → Chainlit）。因为业务逻辑全在 `agent.py` 和 `tools.py`，换前端只动 `ui.py` / `chainlit_app.py`，图、工具、分析后端一行没改。

`ui.py` 的接口就是两个回调（`agent.py:1363`）：

```python
demo = build_ui(submit_task, package_results)
```

### 代价（这是个真教训）

**入口乘以功能数才是测试量。** 这个项目出过一个 bug：命令行一直好用，但 **MCP 入口从第一天起就没起来过**——环境里 `mcp` 升到 2.x，v2 把 `FastMCP` 改名成了 `MCPServer`（挪到 `mcp.server.mcpserver`），代码还按老名字 import。因为只测过 CLI，这个坏掉的入口一直没暴露。

修法是短期的 `pip install "mcp<2"` 钉回 1.30.0。注意 `boltz2-agent`（同骨架的另一个项目）**到现在还没修**。

### 会被追问什么

**Q：为什么不做成 API + 前端分离（后端服务 + HTTP 前端）？**
A：考虑过。但这个项目是单人使用、跑在实验室服务器上，引入 HTTP 层只会增加部署复杂度。而且真正需要「被外部调用」的场景已经由 MCP 覆盖了。**用 MCP 当 API 比自建 HTTP 更好**——因为它是给 AI 用的标准协议，Claude Desktop 直接就能调。

**Q：Chainlit 和 Gradio 你现在留哪个？**
A：Chainlit（7779）。Gradio 那个（7778）已经停了。Chainlit 的优势是步骤流展示——每个节点一个 `cl.Step`，用户能看到「现在在跑第 3 层」，Gradio 里只有一个滚动日志箱。

---

## 9. 优化点：路由的两层设计（关键词 → LLM）

### 问题

判断「用户这句话要干什么」看起来必须用 LLM。但每句话都调一次模型，既花钱又引入不确定性——用户明说「跑一下这个 tpr」，答案是完全确定的。

### 优化点

**第一层用关键词扫描（零 token），第二层才用 LLM。** 这是 §1 那条原则的具体应用。

### 怎么做

`_detect_mode`（`agent.py:574`）先扫关键词：

```python
MD_MARKERS = ["tpr", "step5_1", ".gro", "mdrun", "md 模拟", "分子动力学", "轨迹",
              "rmsd", "run_name", "建md", "跑模拟", "模拟"]
GPCR_MARKERS = ["gpcr", "元动力学", "metadynamics", "plumed", "colvar", "hills",
                "微开关", "microswitch", "tm6", "fes", "dccm", "psn", "lead-lag",
                # ... 共 30 多个
                ]

def _detect_mode(text) -> str:
    t = str(text).lower()
    if ("只做 gpcr" in t) or ("不跑 md" in t):  return "gpcr"
    if ("全流程" in t) or ("模拟+gpcr" in t):   return "full"
    md = any(m in t for m in MD_MARKERS)
    gp = any(m in t for m in GPCR_MARKERS)
    if md and gp: return "full"
    if gp:        return "gpcr"
    return "md" if md else ""
```

命中就直接定意图和模式。`router_node`（`agent.py:603`）只在没命中时才调 LLM：

```python
mode = _detect_mode(last_user)
if mode:
    return {"intent": "pipeline", "mode": mode, "ws_b": _extract_ws_b(last_user)}
# 没命中才走模型，只分 query / chat
```

LLM 这里只区分 `query`（问文档）/ `chat`（寒暄），而且**意图不在三类里就强制落到 `chat`**：

```python
if intent not in ("pipeline", "query", "chat"):
    intent = "chat"
```

失败安全（fail-safe）：模型输出乱七八糟时，最坏结果是当成聊天回一句，而不是误启动几小时的模拟。

### 代价（这个必须主动讲）

**关键词表本身成了新的脆弱点。** 这就是 BUG-4 的成因：`MD_MARKERS` 里有 `"run_name"` 这个通用词，而 Chainlit 前端**无条件**把 `run_name: md` 拼进 prompt（因为 `run_name` 有 `or "md"` 默认值），于是「你好」也命中了关键词 → 判成 MD 任务 → 跑了整条流水线还打了个 zip。

两处修：
1. 前端只在**真填了任务参数**时才拼任务指令（`chainlit_app.py:72`）；
2. 打包也绑到 `is_task` 上（`chainlit_app.py:112`）。

**结论**：用确定性匹配换确定性判定，匹配表就成了新的脆弱点。这是这个设计的固有风险，不是可以回避的。可迁移的做法是——第一层路由用规则，但**规则表的每一个词都要问「它会不会在无关的话里出现」**。

### 会被追问什么

**Q：为什么不用 embedding 相似度代替关键词？**
A：想过。但 embedding 需要调 API（这个项目已经在踩 API 不稳的坑了），而且它的判定不透明——出了问题你不知道是匹配到了什么。关键词匹配的好处是**可解释、可复现**，同一条输入永远给同一个结果。

**Q：除了 `run_name` 还有别的危险词吗？**
A：有。「模拟」「轨迹」这些词在闲聊里也可能出现（「我模拟了一下你的回答」）。但目前用到的场景里，误判的代价只是一次多余的计算，不是错的数据。真正的风险是**误判成任务后自动打包和建目录**——那个已经修了。

---

## 10. 优化点：报错不中断 + 修复回路

### 问题

多步流水线里某一步失败，有两条路：整条链崩掉，还是继续？这个项目里**两种都用了**，取决于上下文。

### 怎么做：两条不同的策略

**MD 四步：失败就进 repair。** 因为步骤有强依赖——`compute_rmsd` 依赖 `analysis.ndx`，没 index 就没法算。所以出口条件边 `_post_tool_route`（`agent.py:253`）在检测到 `【执行报错】` 时转到 `repair`：

```python
def _post_tool_route(next_node):
    def _route(state):
        if _is_tool_error(state["messages"][-1]):
            return "repair"
        return next_node
    return _route
```

repair 回路：`repair`（LLM 决定调哪个诊断工具）→ `repair_tools` → 按 `failed_step` 回到对应步骤，`retry_count > MAX_RETRIES`（2）才真退出。

`failed_step` 的写入在工厂里（`agent.py:209`）：

```python
if last and _is_tool_error(last):
    update["failed_step"] = step_num
    update["retry_count"] = state.get("retry_count", 0) + 1
```

**GPCR 分层链：失败不中断。** 因为层之间**没有强依赖**——景观算不出来不影响耦合分析，耦合算不出来不影响出图。中断只会让整份报告全废。

`_make_det_node` 里每个工具 try/except，失败了写进消息继续下一个：

```python
for t in tools:
    try:
        out = str(t.invoke({}))
    except Exception as e:
        out = "【%s 失败】%s" % (getattr(t, "name", "?"), e)
    msgs.append(HumanMessage(content=out, name="GpcrLayer"))
```

然后由报告的「边界」一节如实写出哪些层失败了。

### 判断依据

**这一步的失败会不会让后续步骤失去意义？** 会 → 中断去修；不会 → 记下来继续。

### 会被追问什么

**Q：`retry_count` 是全局的还是每步的？**
A：全局的（`AgentState` 里就一个 `retry_count`）。这是个小瑕疵——step1 失败重试 2 次后，step3 失败就没预算了。（推测：单步的独立计数更合理，但当前场景下不会触及。）

**Q：repair 用的 LLM 会不会也返回空调用然后静默结束？**
A：会，而且 `repair_route` 的出口也是「没有 tool_call 就 `__end__`」：

```python
def repair_route(state):
    if isinstance(last, AIMessage) and last.tool_calls:
        return "repair_tools"
    return "__end__"
```

区别在于：这里 `__end__` 是**合理的失败退出**（修不了就放弃），不像 §1 那个把「没调用」错当成「做完了」。而且 `repair` 的提示词里明确写了「如确实无法自动恢复，请明确告知放弃」——这是有意的设计，不是漏洞。

---

## 11. 优化点：前端的两处工程细节

这两条偏工程，但很能说明「你在真跑它」而不是只写了个 demo。

### 11.1 每个节点变成一个可折叠的步骤

Chainlit 版把图的每个节点名映射成中文，逐节点开 `cl.Step`（`chainlit_app.py:85`）：

```python
NODE_LABELS = {"router": "判定任务模式", "progress_poll": "等待模拟进度",
               "gpcr_setup": "第1层 编号识别", "gpcr_l2": "第2层 激活坐标",
               "gpcr_l3": "第3层 构象景观", "gpcr_l4": "第4层 耦合机制",
               "gpcr_l5": "第5层 元动力学", "gpcr_plot": "出图", "gpcr_compare": "A/B 对比", ...}

async for event in A.agent_executor.astream(init, config, stream_mode="updates"):
    for node, st in (event or {}).items():
        ...
        async with cl.Step(name=NODE_LABELS.get(node, node), type="tool") as step:
            step.output = body[:2000]
```

**为什么用 `astream(stream_mode="updates")` 而不是 `invoke`**：一次拿到全部结果的话，用户要盯着转圈几分钟。逐节点更新让每一步都能显示出来——一个跑 5 层分析的 agent，看不到进度等于没法用。

`step.output` 截到 2000 字符：UI 层也只给摘要，细节在日志里。

### 11.2 只用 `is_task` 决定要不要走任务的整套动作

这是 BUG-4 的修法，值得单列因为它是个通用模式（`chainlit_app.py:70-112`）：

```python
has_task = bool(tpr or nsteps or traj or ws or ws_b or analyze_only)
if has_task:
    prompt = A._build_prompt(...)   # 只有真填了参数才拼任务指令
else:
    prompt = text                    # 否则原话交给图

is_task = has_task or bool(A._detect_mode(prompt))

if is_task:
    await cl.Message(content="已提交，工作区 `%s`..." % task_id).send()
...
if is_task:                          # ← 打包也绑在同一个判断上
    zip_path, note = A.package_results(task_id)
```

**经验**：一个「要不要做 X」的判断，要做到底。原来只在「显示已提交」用了它，没用到「打包」上，结果打了个空 zip 出来。

---

## 12. 优化点：三个词把人带对地方 —— 提示词的分工

### 提示词分三类，各管各的

| 类别 | 位置 | 管什么 |
|---|---|---|
| 路由 | `INTENT_SYSTEM` | 只分 query/chat（pipeline 由关键词定） |
| 每步 | `STEP1_SYSTEM`…`STEP4_SYSTEM` | 这一步要调什么工具、参数从哪来 |
| 收尾 | `SUMMARIZE_SYSTEM` | 五节骨架怎么组织、哪些不许改 |

### 一个具体的设计：每步提示词里都写「权限」

比如 `STEP1_SYSTEM`：

> 权限：只能调用系统绑定的工具；调用完成后直接结束本轮。若工具返回【执行报错】，系统会自动进入修复分支，不要自行尝试别的工具。

「只能调用绑定的工具」这句其实**没有实际约束力**（§2 里是 `bind_tools` 在做这件事）。写它的目的是**告诉模型别再挣扎**：不然它会尝试调用一个看不见的工具，然后收到一个含糊的错误，浪费一轮。

**提示词解释结构，结构做实际约束。** 这是这两者的正确分工。

### 「禁止臆造」是反复出现的一条

`SYS_RULE`、`STEP1_SYSTEM`、`STEP2_SYSTEM`、`STEP3_SYSTEM` 里都有类似的话：

> tpr 路径、轨迹列表、起点时间、对比工作区必须原样传递，禁止臆造。

因为这是 LLM 在这个项目里最容易造成的实质伤害——它编一个不存在的路径，工具报错，然后整条链停下来。加这句不保证有用，但成本是零。

---

## 13. 还没解决的问题（主动讲，别等着被问）

这一节比前面所有节加起来都有说服力。**能说清自己项目的边界，说明你真的跑过它。**

### 1. 轮询上限对长模拟不友好

60 分钟的上限（120 × 30 秒）对跑 10 小时的模拟太短。到点了图会继续往下走，然后因为找不到 `.gro` 报错。更合理的做法是检测 PID 活着就继续轮，或者把上限做成参数。

### 2. 收尾没有自动的数字校验

骨架里的数字和模型答复里的数字，目前靠人眼对账。要根治得在收尾后做一次自动比对。

### 3. step1–4 的 LLM 仍然可能返回空调用

`_llm_has_tool_call` 的 `__next__ → __end__` 这条路径在 MD 四步里还在。目前的缓解只是「历史里没骨架就走 MD 分支」的兜底，没有重试机制。

### 4. 工具结果截断是按字符切的，不是按语义

16000 字符硬切，可能切掉关键信息。按语义切（保留报错行 + 尾部 N 行）会更稳。

### 5. `thread_id` 会撞

格式精确到分钟，同一分钟提交两个同名任务会共用工作区。

### 6. `boltz2-agent` 的 MCP 入口还坏着

同骨架的另一个项目，`mcp` v2 改名那个 bug 没修。这本身就是「入口多、测试少」的证明。

### 7. CHARMM-GUI 前处理没接

计划里有，一直没做。现在是「吃已经准备好的 `.tpr`」，上游结构准备还得手动跑 CHARMM-GUI。

---

## 14. 常被问、但我没做的（以及为什么）

这一节和上一节互补：上一节是「该做没做」，这一节是「看起来该做但我判断不需要」。

**Q：为什么不用 LangGraph 的 `Send` API 做并行？**
A：MD 四步里有强依赖（index → 分析 → 出图），没法并行。GPCR 层之间理论上可以并行，但每一层都是 CPU 密集的（MDAnalysis 算几十万原子），并行只会让它们在同一个 CPU 上互相抢，不会更快。真要做并行得等到有多机调度。

**Q：为什么不用 LlamaIndex / CrewAI / AutoGen？**
A：核心需求是「固定流程 + 局部用 LLM」，正是 LangGraph 显式图要解决的问题。CrewAI 那种角色对话的范式，流程是隐式的（靠 agent 之间聊天推进），出问题没法定位到某一节点。这个项目最需要的恰恰是「能定位到哪一步」。而且骨架是从另一个项目迁移的，换框架等于重写。

**Q：为什么不做成微调 / 本地模型？**
A：不需要。LLM 在这个项目里只做三件小事（分类意图、提参数、组织语言），这些通用模型已经够好。定制化的部分（分析判据、GPCR 知识）全是代码和领域数据，写成代码比训进模型可靠得多，也可解释。

**Q：为什么不用 function calling 之外的方案，比如让模型生成代码？**
A：想过，否了。让模型生成 shell 命令 = 允许它构造任意命令，安全边界直接没有了。而且 GROMACS 命令的参数语义很刁（2026 改了 `gmx hbond` 的语义，见 BUGLOG 的 BUG-7），生成出来的命令错得不明显。固定工具 + 固定参数，是可以用测试保证的；生成代码不是。

**Q：怎么评估这个 agent 好不好？**
A：目前是**端到端对账**：跑一段真实数据，把 agent 算的量（比如 RMSD 均值）和 `gmx` 原生命令的输出对照（`gmx rms` 给 0.154 nm，脚本给 1.54 nm，这就抓出了 Å/nm 的 bug）。没有做自动化 eval 集。（推测：建立一套「已知答案的轨迹 + 期望输出」的回归集是下一步该做的，工作量不小。）

---

## 15. 自测

合上文档，能答出下面 8 题就算掌握。

1. 主图有哪几个节点？`pipeline` 是什么？
2. 为什么第 2–5 层不交给 LLM？这个决定是被哪一次事故推动的？
3. `bind_tools([])` 用在哪、防什么？
4. 工具结果超过多少字符会被截断？为什么取首尾而不是取前 N？
5. 进度轮询的消息为什么要有固定 `id`？
6. SQLite checkpointer（`SqliteSaver`）的边界是什么？它比 `MemorySaver` 多活了什么、又没解决什么？
7. 报告里「产出清单」这一节为什么必须扫磁盘而不是读工具返回值？
8. 四个入口分别是哪四个文件？它们共用哪一层？

---

## 一句话总结

这个项目里，LLM 负责**听懂意图**和**组织语言**；流程、计算、判断、以及「什么该说、什么不该说」，全部由普通代码负责。

判断一个步骤该给谁，看它有没有需要判断的分支；判断一个结论能不能说，看它有没有不确定性支撑。
