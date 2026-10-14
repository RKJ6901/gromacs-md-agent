"""GPCR 集成测试：agent 图能否编译 + 四段结论骨架能否生成。"""
import os
import sys

sys.path.insert(0, '/data/JRK/md-agent')
os.chdir('/data/JRK/md-agent')

import matplotlib
matplotlib.use('Agg')

import md_agent as A
from md_tools import current_workspace

WS = sys.argv[1] if len(sys.argv) > 1 else '/data2/JRK/md_workspaces/gpcr_test'
print('ws: ' + WS, flush=True)

nodes = sorted(A.agent_executor.get_graph().nodes.keys())
print('--- graph nodes (%d) ---' % len(nodes))
print(', '.join(nodes))

print('--- mode detection ---')
print('analyze-only ->', A._detect_mode('只做 GPCR 分层分析（不跑 MD 模拟） | 分析轨迹: /x/md.xtc'))
print('md+gpcr     ->', A._detect_mode('tpr 路径: a.tpr | 分子动力学 模拟+GPCR 全流程'))
print('gpcr only   ->', A._detect_mode('看一下 plumed 元动力学 FES 和 TM6 张开'))
print('ws_b        ->', repr(A._extract_ws_b('对比工作区: /data2/JRK/md_workspaces/gpcr_ident | 指令: 比对')))

current_workspace.set(WS)
print('--- gpcr report ---')
print(A.build_gpcr_report())
