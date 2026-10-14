"""分析工具层端到端测试：在 agent venv 里跑，验证 md_tools -> worker 子进程 的链路。

用法: python e2e_tools.py [workspace]
"""
import os
import sys

sys.path.insert(0, '/data/JRK/md-agent')
os.chdir('/data/JRK/md-agent')
import md_tools as T

WS = sys.argv[1] if len(sys.argv) > 1 else '/data2/JRK/md_workspaces/gpcr_test'
T.current_workspace.set(WS)
print('workspace: ' + WS, flush=True)


def run(name, fn, **kw):
    print('=' * 72)
    print('### ' + name + ' ' + str(kw))
    try:
        out = str(fn.invoke(kw))
    except Exception as e:
        print('调用异常: ' + repr(e))
        return
    print(out[:900])
    if len(out) > 900:
        print('... (共 %d 字符)' % len(out))
    print()


run('detect_receptor_profile', T.detect_receptor_profile)
run('analyze_segments', T.analyze_segments)
run('analyze_microswitches', T.analyze_microswitches)
run('analyze_tm6_opening', T.analyze_tm6_opening)
run('analyze_ligand_retention', T.analyze_ligand_retention)
run('parse_plumed', T.parse_plumed)
run('analyze_fes', T.analyze_fes)
run('analyze_uncertainty', T.analyze_uncertainty,
    series='microswitch_DRY.csv:3.49-3.50,tm6_opening.csv:TM6ic_TM3ic_nm')
run('analyze_leadlag', T.analyze_leadlag)
run('analyze_landscape', T.analyze_landscape, n_states=4)
run('analyze_correlation', T.analyze_correlation)
run('analyze_network', T.analyze_network)
run('analyze_water_network', T.analyze_water_network)
run('compare_systems', T.compare_systems, ws_b='/data2/JRK/md_workspaces/gpcr_ident')
