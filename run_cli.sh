#!/bin/bash
# GROMACS MD 模拟 + 后处理分析 Agent CLI 提交脚本
# 用法: bash run_cli.sh
# 可覆盖的环境变量:
# TPR=step5_1.tpr 起始 tpr 路径(必填)
# NSTEPS=5000000 覆盖步数(可空, 空=用 tpr 原值)
# RUN=md 输出前缀
# TRAJ=step5_1.xtc,step5_2.xtc 分析轨迹(逗号分隔)

TPR="${TPR:-step5_1.tpr}"
NSTEPS="${NSTEPS:-}"
RUN="${RUN:-md}"
TRAJ="${TRAJ:-}"

python mdagent/agent.py --mode cli --tpr "$TPR" --nsteps "$NSTEPS" --run-name "$RUN" --traj "$TRAJ" --msg "请按顺序执行: 1.提交MD 2.建index并合并轨迹 3.算RMSD/RMSF/Rg/Hbond 4.出图"
