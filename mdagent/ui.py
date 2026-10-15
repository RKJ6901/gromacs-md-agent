"""MD Agent 控制台外观层：主题、CSS、布局。

只负责画界面并接好回调，不含业务逻辑：提交与打包两个回调由 agent.py 传入。
配色沿用分析出图的同一套色（青色主色 + 暖棕辅色），让控制台和图看着是一套东西。
"""
import os

import gradio as gr

_CSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.css")


def _load_css() -> str:
    with open(_CSS_PATH, encoding="utf-8") as f:
        return f.read()


def _theme() -> gr.themes.Base:
    """Fira Sans / Fira Code 打底，中文回退到 Noto Sans SC / 微软雅黑。"""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.stone,
        neutral_hue=gr.themes.colors.stone,
        font=[gr.themes.GoogleFont("Fira Sans"), "Noto Sans SC", "Microsoft YaHei",
              "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Fira Code"), "ui-monospace", "Consolas",
                   "monospace"],
    )


HEADER_HTML = """
<div class="md-head">
  <div class="md-brand">
    <span class="md-dot" aria-hidden="true"></span>
    <span class="md-title">MD Agent 控制台</span>
  </div>
  <div class="md-sub">GROMACS 分子动力学模拟 &middot; GPCR / 元动力学后处理分析</div>
</div>
"""
CSS = _load_css()
THEME = _theme()

import prep_defs as _P


def _choices(d):
    out = []
    for k, v in d.items():
        label = v['label']
        note = v.get('note', '')
        out.append(('%s — %s' % (label, note) if note else label, k))
    return out


FF_CHOICES = _choices(_P.FORCEFIELDS)
ROUTE_CHOICES = _choices(_P.LIGAND_ROUTES)
WATER_CHOICES = _choices(_P.WATER_MODELS)
BOX_CHOICES = _choices(_P.BOX_SHAPES)
LIPID_CHOICES = [('POPC', 'POPC'), ('POPE', 'POPE'), ('DPPC', 'DPPC')]


def build_ui(submit_fn, fetch_fn):
    """画控制台，并把 submit_fn / fetch_fn 接到按钮上。

    Gradio 6 把 theme / css 移到了 launch()：这里只建 Blocks，
    调用方 launch 时传 ui.THEME / ui.CSS。
    """
    with gr.Blocks(title="MD Agent 控制台") as demo:
        gr.HTML(HEADER_HTML)
        _body(submit_fn, fetch_fn)
    return demo


def _body(submit_fn, fetch_fn):
    """左列是控制输入（两页），右列是日志与结果。"""
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=380):
            with gr.Tab("运行模拟"):
                tpr_input = gr.Textbox(label="起始 .tpr 路径", lines=1, placeholder="/data/xxx/step5_1.tpr")
                nsteps_input = gr.Textbox(label="步数", lines=1, placeholder="5000000", info="留空则用 .tpr 原值")
                runname_input = gr.Textbox(label="输出前缀", lines=1, placeholder="md")
            with gr.Tab("分析已有轨迹"):
                ws_input = gr.Textbox(label="分析工作区", lines=1, placeholder="/data2/JRK/md_workspaces/gpcr_test", info="填了就就地分析，不新建任务目录")
                ws_b_input = gr.Textbox(label="对比工作区（可选）", lines=1, placeholder="/data2/JRK/md_workspaces/gpcr_ident", info="A/B 对比的第二套体系")
                analyze_only_input = gr.Checkbox(label="只做 GPCR 分层分析，不跑模拟", value=False)
            with gr.Tab('准备体系'):
                protein_in = gr.Textbox(label='蛋白结构 .pdb', lines=1)
                ligand_in = gr.Textbox(label='配体文件 .mol2 / .sdf', lines=1)
                escalate_in = gr.Checkbox(label="遇到歧义停下来问我（默认：agent 自己决定并记账）", value=False)
                with gr.Row():
                    ff_in = gr.Dropdown(label='蛋白力场', choices=FF_CHOICES, value='charmm36-jul2022')
                    route_in = gr.Dropdown(label='配体路线', choices=ROUTE_CHOICES, value='cgenff')
                water_in = gr.Dropdown(label='水模型', choices=WATER_CHOICES, value='tip3p')
                with gr.Row():
                    box_in = gr.Dropdown(label='盒型', choices=BOX_CHOICES, value='dodecahedron')
                    pad_in = gr.Number(label='边距 nm', value=1.0)
                with gr.Row():
                    salt_in = gr.Dropdown(label='盐类型', choices=['NaCl', 'KCl', 'MgCl2'], value='NaCl')
                    conc_in = gr.Number(label='浓度 M', value=0.15)
                membrane_in = gr.Checkbox(label='建膜（本地晶格双层插入）', value=False)
                lipid_in = gr.Dropdown(label='脂质类型', choices=LIPID_CHOICES, value='POPC')
            traj_input = gr.Textbox(label="分析轨迹 (.xtc)", lines=2, placeholder="step5_1.xtc,step5_2.xtc", info="多段用逗号分隔")
            user_input = gr.Textbox(label="附加指令（可选）", lines=1, placeholder="例如：起点时间 0,20000")
            submit_btn = gr.Button("提交执行", variant="primary")
        with gr.Column(scale=7, min_width=420):
            error_box = gr.Textbox(label="状态", lines=2, interactive=False, elem_id="md-status")
            chatbot = gr.Chatbot(label="执行日志", height=420, elem_id="md-log")
            with gr.Row():
                fetch_btn = gr.Button("打包下载结果", variant="secondary", scale=1)
                download_file = gr.File(label="结果压缩包", interactive=False, scale=2)
    task_state = gr.State("")
    submit_inputs = [tpr_input, nsteps_input, runname_input, traj_input, ws_input, ws_b_input, analyze_only_input, protein_in, ligand_in, ff_in, route_in, water_in, box_in, pad_in, salt_in, conc_in, membrane_in, lipid_in, escalate_in, user_input, chatbot, error_box, task_state]
    submit_outputs = [chatbot, user_input, error_box, task_state]
    submit_btn.click(fn=submit_fn, inputs=submit_inputs, outputs=submit_outputs)
    user_input.submit(fn=submit_fn, inputs=submit_inputs, outputs=submit_outputs)
    fetch_btn.click(fn=fetch_fn, inputs=[task_state], outputs=[download_file, error_box])
