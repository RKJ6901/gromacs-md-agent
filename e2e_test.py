"""MD agent Step2(GROMACS mdrun) + Step3(轨迹分析出图) 端到端短测。

直接调用 md_tools 的 @tool（绕过 LLM，先验证机制），时长可控。
"""
import os
import sys
import time
import shutil

sys.path.insert(0, "/data/JRK/md-agent")
import md_tools as T

WS = "/data2/JRK/md_workspaces/e2e_test"
SRC = "/data/real_JRK/888r_batch/OSSM_188503"
NSTEPS = int(os.getenv("E2E_NSTEPS", "200000"))   # 200000 x 2fs = 400 ps

os.makedirs(WS, exist_ok=True)
T.current_workspace.set(WS)
print("workspace:", WS, flush=True)

if not os.path.exists(f"{WS}/step5_1.tpr"):
    shutil.copy(f"{SRC}/step5_1.tpr", WS)
TPR = f"{WS}/step5_1.tpr"

# ---------------- STEP 2: 短 MD ----------------
if os.path.exists(f"{WS}/md.gro") and os.path.exists(f"{WS}/md.xtc"):
    print("\n=== STEP2: 已存在 md.xtc，跳过 mdrun ===", flush=True)
else:
    print("\n=== STEP2: gmx mdrun ===", flush=True)
    print(T.run_gromacs_md.invoke({
        "tpr_path": TPR, "nsteps": NSTEPS, "run_name": "md", "ntomp": 8,
    }), flush=True)

    for i in range(90):
        time.sleep(10)
        if os.path.exists(f"{WS}/md.gro"):
            break
        print("  [%ds] %s" % ((i + 1) * 10, T.check_md_status.invoke({"run_name": "md"}).replace("\n", " | ")), flush=True)
print(T.check_md_status.invoke({"run_name": "md"}), flush=True)
assert os.path.exists(f"{WS}/md.xtc"), "mdrun did not produce md.xtc"
print("STEP2 OK -> md.xtc", os.path.getsize(f"{WS}/md.xtc"), "bytes", flush=True)

# ---------------- STEP 3: 分析 ----------------
print("\n=== STEP3: 分析准备 ===", flush=True)
print(T.prepare_analysis_index.invoke({"tpr_path": TPR}), flush=True)

TRAJ = f"{WS}/md.xtc"
jobs = [
    ("rmsd  Backbone", lambda: T.compute_rmsd.invoke({"tpr_path": TPR, "traj_path": TRAJ})),
    ("rmsf  Backbone", lambda: T.compute_rmsf.invoke({"tpr_path": TPR, "traj_path": TRAJ})),
    ("gyrate Protein", lambda: T.compute_gyrate.invoke({"tpr_path": TPR, "traj_path": TRAJ})),
    ("hbond Prot-LIG", lambda: T.compute_hbond.invoke({"tpr_path": TPR, "traj_path": TRAJ})),
]
for name, fn in jobs:
    print("\n--- %s ---" % name, flush=True)
    print(fn(), flush=True)

print("\n=== STEP3: 出图 ===", flush=True)
xvg_map = {
    "rmsd": f"{WS}/rmsd.xvg",
    "rmsf": f"{WS}/rmsf.xvg",
    "gyrate": f"{WS}/gyrate.xvg",
    "hbond": f"{WS}/hbond_count.xvg",
}
print(T.plot_md_figures.invoke({"xvg_map": xvg_map, "out_prefix": f"{WS}/figs/md"}), flush=True)

print("\n=== DONE ===")
for p in ("figs/md.png", "figs/md.pdf", "rmsd.xvg", "rmsf.xvg", "gyrate.xvg", "hbond_count.xvg"):
    fp = os.path.join(WS, p)
    print(("OK  " if os.path.exists(fp) else "MISS") , p, os.path.getsize(fp) if os.path.exists(fp) else "")
