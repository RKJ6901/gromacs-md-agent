# GROMACS 常用命令与 MD 后处理分析速查

## 通用：分组选择
gmx 分析命令用 -n index.ndx 指定分组，分组名或编号均可，交互式从 stdin 读取。
本流水线用 analysis.ndx，关键分组: System(0) / Protein(1) / Backbone(4) / LIG(13) / Complex(17)。

## 1. 轨迹合并 gmx trjcat
CHARMM-GUI 分段生产(step5_1.xtc ... step5_N.xtc)常从 0 ps 重启，必须用 -settime 手动给每段起点:
printf '0\n20000\n40000\n' | gmx trjcat -f step5_1.xtc,step5_2.xtc,step5_3.xtc -o merged.xtc -settime
若轨迹时间已连续，可省略 -settime: gmx trjcat -f a.xtc,b.xtc -o merged.xtc

## 2. RMSD (gmx rms)
先按 fit 组叠合，再算 calc 组的 RMSD。默认 Backbone<->Backbone:
printf '4\n4\n' | gmx rms -s step5_1.tpr -f merged.xtc -n analysis.ndx -o rmsd.xvg
配体 RMSD 用 calc 组 LIG(13)；复合物用 Complex(17)。单位 nm。

## 3. RMSF (gmx rmsf)
逐残基涨落，-res 输出残基号作为 x 轴:
echo 4 | gmx rmsf -s step5_1.tpr -f merged.xtc -n analysis.ndx -o rmsf.xvg -res

## 4. 回转半径 (gmx gyrate)
蛋白紧致度随时间变化:
echo 1 | gmx gyrate -s step5_1.tpr -f merged.xtc -n analysis.ndx -o gyrate.xvg

## 5. 氢键 (gmx hbond)
蛋白-配体氢键数与几何:
printf '1\n13\n' | gmx hbond -s step5_1.tpr -f merged.xtc -n analysis.ndx -o hbond.xvg -num hbond_count.xvg
-num 给氢键数(出图用)，-o 给距离/几何。

## 6. MM/GBSA (gmx_MMPBSA)
蛋白-配体 GB 结合能 + 逐残基分解。mmpbsa.in 关键项: igb=5, saltcon=0.15, idecomp=1。
gmx_MMPBSA -O -i mmpbsa.in -cs step5_1.tpr -ct merged.xtc -ci analysis.ndx -cg Protein LIG -cp topol.top -nogui
无 topol.top 时省略 -cp，走 tpr/CHARMM-GUI 模式。产物: FINAL_RESULTS_MMPBSA.dat。
注意: -cs 与 -ct 的原子数必须一致；用全体系(fit 后)轨迹，而非 stripped 复合物轨迹。

## 7. 常见报错与处理
- LINCS warning: 时间步过大或体系 clash，减小 dt 或放宽约束；持续警告会引发 NaN。
- NaN: 数值发散，检查 PBC/参数冲突或时间步过大，用更小 dt 重跑。
- lost atoms / number of atoms 不一致: 拓扑与轨迹不匹配，trjcat 合并需给对 start_times。
- CUDA out of memory: 调小 -ntomp，或降低输出频率/换卡重跑。
- trjcat 时间重叠: 分段轨迹从 0 重启，用 -settime 指定每段起点。

## 8. GPCR Nature 风格出图
matplotlib，DejaVu Sans，7.5pt；隐藏 top/right spine；面板字母 A/B/C；
输出 PDF(fonttype 42) + PNG(600dpi)。RMSD/Rg/Hbond 用时间轴，RMSF 用残基号轴。
