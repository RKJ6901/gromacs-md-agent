"""GPCR 出图端到端测试：在 agent venv 里跑 md_tools.plot_gpcr_figures。"""
import os
import sys

sys.path.insert(0, '/data/JRK/md-agent')
os.chdir('/data/JRK/md-agent')

import matplotlib
matplotlib.use('Agg')

import md_tools as T

WS = sys.argv[1] if len(sys.argv) > 1 else '/data2/JRK/md_workspaces/gpcr_test'
T.current_workspace.set(WS)
print('ws: ' + WS, flush=True)
out = T.plot_gpcr_figures.invoke({'families': '', 'out_dir': 'figs'})
print(str(out))
