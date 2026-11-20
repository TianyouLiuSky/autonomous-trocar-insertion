# CIS II / ISMR 2027 — Codex Engineering Handoff

**交接日期：2026-11-19。目标：在现有 Codex 项目中新建隔离工作分支，重组已存在的 Python 研究代码，首先交付一套能在原实验室电脑上按步骤重新标定并运行 ISMR 双阶段 Trocar 插入实验的代码包，同时解释最终文件架构。**

> 本文件是工程任务交接及已核实的 Google Drive 资料索引。Google Drive 中存在重复命名及内容不完整的副本。请逐项读取实际文件，记录来源，再进行合并。不要凭文件夹名称、上传时间、README 声称的完整性或体积自动判断版本。本文日期表示源码的历史修改时间；2026-11-19 左右的 Drive 上传或复制时间不可用来判定源码更新顺序。

## 1. 用户已确定的范围与前提

1. 复现实验应以用户提供的 **`ISMR27_0007_MS(1).pdf`** 作为方法与实验流程的主要验收依据。论文为 *Making the Port: Autonomous Bi-Planar Scleral Trocar Insertion with Eye-Anchored Microscope Targeting*。这份 PDF 需要用户单独添加到 Codex 工作区；本文件本身没有携带 PDF 二进制数据。
2. 目标机器是原实验室电脑，ROS、SHER 2.1/机器人驱动、RealSense、DeckLink/Leica 的采集链及 Python libraries 已存在。**本轮忽略重新安装这些实验环境依赖及底层驱动工程**；可以记录所需配置和现有接口，但不要让它们阻塞首个代码包交付。
3. **全部 Calibration 从头采集并求解。** 不允许把 2026 年 8–10 月的 `handeye_T.npz`、`tip_offset_*.npz`、`mic_extrinsic_calib.npz`、`ring_to_entry_offset*.npz`、`origin_correction.npz`、`phantom_normal.npz` 等直接作为新实验的有效标定结果。旧文件仅用于研究 schema、路径、回归对照和历史溯源；需要可以在空标定目录下开始工作。
4. 希望复现论文中的显微镜眼球锚定目标、D405 法向量、移动 RCM 的双阶段插入、独立检测及 20-trial 实验记录。论文明确规定每阶段由操作人员确认和启动，保留这一人机确认方式；不要擅自改成无人值守的全自动连续穿刺。
5. 第一阶段优先产出 **可执行的整合代码包** 和 **清楚的目录/调用架构说明**，随后再考虑清理代码风格或进一步模块化。不要为了重构而改动已经验证的运动学、插入速度、偏移模型或阶段逻辑。
6. Codex 应先检查自己当前项目/仓库是否已有更新的本地版本；若发现比以下 Drive 资料更新且有明确证据的修改，先比较差异并报告，再决定合并。操作应在新 Git 分支上开展，保持原仓库、历史数据和源码不被覆盖。

## 2. Google Drive 资料来源与优先级（已核对实际文件）

### A. P0 / 最新主控制源码：2026-10-12 `01_code (1)`

**目录：** https://drive.google.com/drive/folders/187xp7M9GysksogQ3RaxNHgTLa8aCUqqZ

**主程序：** `home_to_rcm_insert_v2_precision.py`，377,882 bytes，修改时间 **2026-10-12 22:17 UTC**。https://drive.google.com/file/d/1aVWXzB6VLVBu7VoSp-UWNqvNhjiBn2ZS/view

**启动脚本：** `run_main.sh`，5,786 bytes，修改时间 **2026-10-12 21:42 UTC**。https://drive.google.com/file/d/1zqWILmCz1516bBEWijoR0Zl6--bc13RS/view

**实验存档：** `trial_archiver.py`，10,477 bytes，修改时间 **2026-10-12 22:25 UTC**。https://drive.google.com/file/d/1WWiuwsq45DWrH3sicIEcmCD-2yXcbMkP/view

**同目录已有：** `measure_trial_holes.py`、`save_manual_still.py`、`handeye_calibrate.py`、`handeye_capture_one.py`、`pivot_calibration.py`、`pivot_residual_fit.py`、`fast_pivot_fit_woodbury.py`、`rcmoffset_fit.py`、`tip_inee_guard.py`、`tip_pin_check_v3_fullpose.py`、`handeye_vs_tip_discriminate.py`、`origin_correction_from_discrim_csv.py`、`intr_util.py`、`jog_only.py`、README/验证规程，以及 `microscope_autonomy/`、`backups/`。

**最新显微镜源码子目录：** https://drive.google.com/drive/folders/133gGMvhDvq9IXu6_tAbrqp1dioFQmbux 。包含 `mic_perception.py`（14,202 bytes，修改于 2026-10-12）、`mic_extrinsic_calib.py`、`mic_ring_offset_verify.py`、`mic_independent_verify.py`、`click_calib_points.py`、`click_calib_points_per_touch.py`、`mic_live_capture.py`、`mic_live_capture_ros.py`、`phantom_normal_repeatability.py`、`correspondence_report.py`。

**基线规则：** 以本包的主程序、`run_main.sh`、`trial_archiver.py` 和上述显微镜模块为首选。早期主程序仅用于查找已丢失的功能、对照历史行为；不要直接覆盖最新版本。`run_main.sh` 已包含 20 trial 的 `INSERT_ANGLE_SCHEDULE` 与 `MIC_RCM_ANGLE_SCHEDULE`，默认 `TIP_MODEL=rcmoffset`、`MIC_RCM_MODE=circle`，并保留区分 free-space 与 `--trocar` 的手动旋转控制开关。

### B. P0 / 最完整的 2026-10-12 历史实验快照：ISMR 2027 `package_2026-10-12`

**目录：** https://drive.google.com/drive/folders/1JgYl7vza8SjEEvf4m9scv3yjfpBvrbz9

**其 `01_code/`：** https://drive.google.com/drive/folders/1bYqivC4h79v_BDy_antl4my8duZ8sAOp 。实际含 377,882-byte 同名主程序、启动脚本和标定工具；应与 A 计算 SHA-256 后比较，不必仅凭尺寸认定二进制完全相同。

**其 `02_calibration/`：** https://drive.google.com/drive/folders/1iwzDxtIW_o6jCNMLHB10ygMXJrVZrcSH 。实际核实有 `handeye_T.npz`、`tip_offset_rcmoffset.npz`、`tip_offset_pivot.npz`、`mic_extrinsic_calib_ACTIVE_microscope_autonomy.npz`、`ring_to_entry_offset_ACTIVE_microscope_autonomy.npz`、`origin_correction.npz`、`phantom_normal.npz`、`mic_calib_touches.jsonl`、`ring_offset_samples.jsonl` 和 `03_results_npz/` 内的 11 个历史模型文件。**这些是旧实验参数，仅作 schema 和依赖关系参考，严禁当作本次新标定结果。**

**其 `03_logs/`：** https://drive.google.com/drive/folders/14-LoNNzWP6H1LE2yUVH3fwIt1R0GYOxC 。实际含 `precision_log.csv`、`correspondence_log.csv`、`orient_trace_last.csv` 和 archiver 日志。

**其 `04_formal_campaign_20trials/`：** https://drive.google.com/drive/folders/13u104P_ydnOxoHFEotsKdwB3KXPBbapp 。实际有 `campaign_trials.csv`、`campaign_summary_phase1.txt` 及 20 个 `trialNN_*` 目录，已抽查 `trial01`、`trial20`，均包含照片、JSON、按 trial 分割的 CSV、旋转轨迹及 `config_snapshot.json`。可据此复制数据结构并设计分析脚本的 fixture，不得将这些历史 trial 作为新的实验结果。

**注意同名不完整副本：** https://drive.google.com/drive/folders/1vNKtSsXptqlhOP0_NIN3hMzM13gBWa58 。之前检查发现其 `01_code/microscope_autonomy`、`02_calibration/03_results_npz` 和若干 trial 子目录为空。此副本不能替代上面的 `1JgY...`。Google Drive 复制时间不代表源码修改时间。

### C. P1 / 从零标定工具与固件算法参考：`ISMR_Complete_Package_2026-08-25`

**目录：** https://drive.google.com/drive/folders/11q0lyi3TkL6MExuVcH0hDpV_LLtOAKXe

**相机内参与 Hand-eye：** https://drive.google.com/drive/folders/1FDs9hczyCJybVx-O6DYXFWj-epWi-Er_ 。包含 `camera_intrinsics_calibrate_charuco.py`、`camera_intrinsics_calibrate.py`、`handeye_calibrate.py`、`handeye_reproj_refine.py`、`handeye_compare_on_val.py`、`handeye_live_recheck.py` 等。优先从这里补齐在 10 月 12 日主代码目录中不存在的 **相机内参重新标定程序** 和独立验证工具。Hand-eye 采集与求解功能 10 月 12 日也已有，默认保留较新版本。

**Pivot 和 tip 样本：** https://drive.google.com/drive/folders/1JHGt2TUonVc6ntA8VCDLPFuJEK9Vb6WT 。含 `pivot_calibration.py`、`pivot_capture_one.py`、`pivot_residual_fit.py`、`pivot_visual_crosscheck.py`、历史 CSV。若 10 月 12 日的主程序需要后者的独立采集函数，可补齐；历史 CSV 只能用于离线测试。

**Tip offset 视觉修正：** https://drive.google.com/drive/folders/1XMl2VmgfKvrmvDwoxNSnpd-LmrlA030P 。含 `rcmoffset_fit.py`、`tip_offset_refit_zeromean.py`、`tip_offset_apply_visual_correction.py`、`tip_pin_check*.py`，用于重建和验证偏移模型。后两个脚本中有历史数据路径及旧实验数值假设，重用前必须参数化。

**固件研究参考：** https://drive.google.com/drive/folders/12UCRMkGvvQEuV53u2ofRTnAyM-KqLDml 。含 `robotTask.cpp`、`robotModel.cpp`、`appTask.cpp` 和差异文件。用户假设现有硬件驱动可用，本轮只用它们核对 TipInEE、RCMoffset 与运动学约定，除非确认必要且获得明确许可，否则不修改机器人固件。

**针轴诊断：** https://drive.google.com/drive/folders/1pikkDfJenhEYUNkRsDcXLBWQQQb0FVL0 ，`needle_axis_diagnose.py`。该旧脚本能从新测得的针杆 3D 方向估计 EE 坐标系内的轴，但主要打印结果；需补齐新的有效轴参数存储/载入与验证接口。

### D. P2 / 2026-10-06 中间演示包：仅用于查回遗漏变化

https://drive.google.com/drive/folders/1iOa_ARMK7qL_X5oDGCWoyG2o2BLYMnqa

此包的主程序约 327 KB，旧于 10 月 12 日 377,882-byte 版本；有 Leica 配准代码和该阶段的技术说明。只用于差异审计、追踪 10 月 6 日的视觉线程/性能改进。README 明确记载当日仍使用 10 月 5 日的显微镜标定，未完成计划中的重新标定，**不要将此包直接当作最终主版本**。

### E. 可选研究参考：其他归档与 `rcm_definition.py`

**2026-10-01 Session：** https://drive.google.com/drive/folders/1WkUI9uvNKjy_z4UaR4SJLe59IMHS7OkF ，代码旧于 10 月 12 日。

**2026-09-29 Eye-Eye Calibration：** https://drive.google.com/drive/folders/1TyJv5Afoi74ngn40XKVe0OYp_5ULKU5B ，包含 `eye_eye_calibration.py` 等。提交版 ISMR 主链路采用独立 D405→机器人 Hand-eye，以及 Leica 像素→机器人触点仿射映射；两相机直接 6-DoF Eye-Eye Calibration **并非本文复现实验的必备前置步骤**，仅当代码/现行硬件明确需要时才纳入。

**`rcm_definition.py`**：用户在这段 ChatGPT 会话上传的独立约 243 行 RCM 几何说明/数值自检文件，须由用户作为附件交给 Codex。它并不是 SHER 的独立运行主程序。注意该文件开头把初始 ORIENT pivot 描述为表面点 E，但 10 月 12 日主程序实际在 ORIENT 时传入 `pivot_point=cur_tip()`（悬停针尖当前位置）；REORIENT 传入 `S.internal_rcm`（通道末端 P）。以经过核对的主程序与提交论文的 **Stages / Holding a pivot** 段落为准，修订模块说明，避免错误覆盖控制逻辑。

## 3. 复现流程的验收定义（以提交论文为准）

论文目标：Leica 1080p 顶视识别 limbus 中心和眼球附着的 fiducial 0，以相对于 fiducial 的角度和半径定义目标；通过新采集的 microscope pixel→robot base 标定映射为表面入针点 E。D405 提供表面法向量和独立的粗目标交叉验证。机器人从 8 mm 悬停点开始，初始 ORIENT 绕悬停针尖转向指定的 25°–45° 斜入角，沿斜向进行 1.5 mm 的 TUNNEL，内部 RCM 取通道末端 P，再绕 P 调整为沿表面内法向的方向，PLUNGE 3.0 mm，然后轴向撤针、抬高避让、返回 HOME。论文称 20 次插入、140 个闭环（每次 7 个）完成，操作人员负责确认目标与启动各阶段。位置/角度停止阈值分别约为 0.4 mm / 0.3°。最终要能生成每次 trial 的目标、法向、速度、姿态、闭环误差、Leica vs D405 差异、照片与独立针杆方向检查及针孔测量结果。

参考 PDF：用户单独上传的 `ISMR27_0007_MS(1).pdf`，论文第 2 页图 2（系统架构）、第 3–4 页（方法与几何）、第 5–6 页（20-trial 实验和评估）。如果论文与 10 月 12 日 README 存在差异，**不要擅自调整旧实验数据**；记录差异，并以论文定义当前复现目标。旧统计结果属于历史参考，不作为本轮硬件上的预期自动输出。

## 4. Calibration / Registration 全新采集：需保留的能力及预计产物

| 次序 | 所需操作和功能 | 已有候选源码 | 新运行应生成或更新的文件 / 要求 |
|---|---|---|---|
| 0 | 新会话初始化、无旧模型运行检查 | 新增简单 preflight/manifest 即可 | 不载入历史 `.npz`；原始文件移入 `reference_only` 或不放入运行搜索路径 |
| 1 | D405 RGB 相机内参（必要时重标） | 8 月包 `camera_intrinsics_calibrate_charuco.py`、`camera_intrinsics_calibrate.py` | `cam_intrinsics.npz`（核对 `intr_util.py` 实际读取的 schema/路径） |
| 2 | Hand-eye：D405→机器人基系 | 10 月包 `handeye_calibrate.py`、`handeye_capture_one.py`，8 月独立验证工具 | 新 `handeye_T.npz`；保留采样 CSV 与拟合/验证报告 |
| 3 | Trocar Tip / Pivot 采样 | `pivot_calibration.py`、`pivot_capture_one.py`、`tip_pin_check_v3_fullpose.py` | 新 `pivot_samples.csv`、姿态样本；核对 GUI/Firmware 报告的 TipInEE |
| 4 | Tip orientation-dependence / RCMoffset 模型 | `rcmoffset_fit.py`、`fast_pivot_fit_woodbury.py`、`tip_offset_refit_zeromean.py` | 新 `tip_offset_rcmoffset.npz` 或经证据选定的兼容模型；测试 `TIP_MODEL` 选择及模型有效区域 |
| 5 | 视觉核对 tip / 原点偏差 | `handeye_vs_tip_discriminate.py`、`tip_offset_apply_visual_correction.py`、主程序 `c` key、`origin_correction_from_discrim_csv.py` | 全新 `origin_correction.npz`（若有必要）；不得保留旧固定校正；照片点击、残差、诊断报告 |
| 6 | 真实 needle axis in EE frame | 主程序独立 3D shaft check、`needle_axis_diagnose.py` | **增加当前轴校准输出文件及主程序加载途径**；方向、单位长度、符号和精度检查 |
| 7 | Leica microscope→机器人基系触点 Registration | 主程序 `1`–`8` 等触点采集键、`click_calib_points*.py`、`mic_extrinsic_calib.py` | 新 `mic_calib_touches.jsonl`、点击 JSON、`mic_extrinsic_calib.npz`；至少按算法要求采点并检查 LOO/held-out 指标 |
| 8 | D405 ring→入针点的粗定位/交叉核验 | `mic_ring_offset_verify.py`、主程序 `0`/反引号键 | 若选用该粗定位模型，则从新配对样本生成 `ring_to_entry_offset.npz`（或 circle 变体）；修复无旧模型时采集被提前拒绝的问题 |
| 9 | 每个 trial 的 phantom surface normal / target lock | 主程序 `n` 键及法向模块、`mic_perception.py` | 当前会话新 `phantom_normal.npz` 与锁定状态；每 trial 检查 freshness 和目标有效性 |
| 10 | 20 trials 的插入实验和归档 | 最新 `run_main.sh`、主程序、`trial_archiver.py` | `precision_log.csv`、`correspondence_log.csv`、trial JSON/图像、每 trial 的 `config_snapshot.json`、汇总 CSV |
| 11 | 独立验证和论文分析 | `mic_independent_verify.py`、`measure_trial_holes.py`、`correspondence_report.py`、`needle_axis_diagnose.py` | 视觉独立验证指标、孔位偏差、针轴偏差、历史统计方法复现；缺失的统计脚本请补全 |

每一步的确切输入/输出 schema、环境变量、函数调用及路径均由 Codex 逐文件核实。**这里列的是应有链路，不代表上述所有脚本目前已实现零改动顺畅衔接。**

## 5. 已发现的具体代码问题：优先修复

**（一）针轴仍使用历史硬编码常量。** 10 月 12 日主程序内含 `NEEDLE_AXIS_MEASURED_20260826 = np.array([-0.0757, 0.1164, -0.9903])`，并覆盖运行时 `NEEDLE_AXIS_LOCAL`。对新器械及新标定必须删除默认沿用旧值的行为，增加当前针轴标定文件/配置的生成、加载、归一化、符号方向、和独立测量校验。旧 `needle_axis_diagnose.py` 可用于计算/诊断，但当前未确认其能够直接写入运行时可加载的校准结果。

**（二）首次 D405 ring offset 标定存在依赖闭环。** 10 月 12 日 `log_ring_offset_sample()` 调用 `calib_freshness_problems()`，当前过滤 `ring_to_entry_offset_circle MISSING`；默认 `MIC_CIRCLE_D405_MODEL=fixed` 模式则报告 `ring_to_entry_offset MISSING`，使首次采集可能因输出文件还没生成而被拒绝。需区分 **collect-phase dependencies** 和 **use-phase dependencies**，首次采集只要求 Hand-eye、Leica Registration、正常目标与新鲜帧等真正的上游依赖，不要求还未拟合的模型已经存在。

**（三）TipInEE guard 依赖 8 月历史常量。** `tip_inee_guard.py` 的默认 `CANONICAL_TIP_INEE` 为 `[0.82968,-2.84431,-13.02703]`；`handeye_vs_tip_discriminate.py` 默认调用 `assert_tip_inee()`。允许在新标定时传入本次 TipInEE 期望值，由当前模型文件决定，并与机器人 GUI/运行时值交叉验证。不能直接关闭必要的物理一致性防护。

**（四）Visual tip correction 旧参数假设。** `tip_offset_apply_visual_correction.py` 中的 `SRC_CSV` 固定指向 `handeye_vs_tip_20260825_180754.csv`、代码硬编码旧相机内参，并有校正向量约 9.4 mm 的旧实验特定阈值；若纳入重建流程，全部变为运行时输入加物理合理性验证，不可强制历史误差幅度重现。

**（五）源文件大量路径和输出位置按旧目录硬编码。** 例如 `measure_trial_holes.py`、`trial_archiver.py` 等使用 `/home/eyebrp/Matthew/microscope_autonomy`；`run_main.sh` 在原位启动主程序。需要统一且可配置的 root、calibration outputs、日志路径；同时尽量保持与实验室现有环境的兼容性，避免破坏已验证的入口脚本。

**（六）完整论文级统计脚本需要核查/补足。** 实际存在测孔、显微镜配准独立验证、D405 差异分析、归档和针轴诊断代码。然而提交稿包含的 leave-one-trial-out 共同偏移估计、置信区间、校正后孔位中位误差、完整 Fig. 7–8 指标等，尚未逐一确认均有可直接运行的离线分析入口。请通过现有 CSV 结构对齐这些输出；不得凭历史论文数字写死结果或臆造测量。

**（七）RCM 描述存在文件间不一致。** `rcm_definition.py` 开头写 ORIENT pivot=E，但最新主程序 ORIENT 用 `cur_tip()`，REORIENT 用 `S.internal_rcm`。论文第 4 页在 Holding a pivot 段也表述初始 Orient 绕当前针尖，Reorient 绕 P。优先修订文档/独立模块以真实算法为准；如要改控制逻辑，先写出几何与运动学差异测试并报告。

## 6. 第一阶段交付物（必须生成，禁止只输出计划）

**交付 A：`dist/CISII_ISMR_Runnable_v1.zip`**。包含从已发现的最新版本重组的完整 Python 用户态源代码、必须的标定工具、Registration 工具、视觉模块、主程序、启动脚本、实验归档与必要的诊断/测量脚本。可以在实验室已有环境中解压运行；硬件上尚未进行过的步骤必须明确标为 `HARDWARE_UNVERIFIED`，不要虚称“成功运行插入”。如果连接测试不具备条件，至少保证 Python 语法、模块导入、文件路径解析、CLI 帮助/离线核心算法测试以及 no-old-calibration preflight 能执行。提供压缩包 checksum。

**交付 B：`docs/FILE_ARCHITECTURE.md`**。逐目录解释文件用途、入口点、导入关系、由谁生成何种标定参数、运行时由谁读取、日志流向，以及各脚本是否必须/可选/仅历史备份。可以附 ASCII 树和 Calibration dependency DAG。

**交付 C：`docs/FRESH_CALIBRATION_RUNBOOK.md`**。按实际脚本给出从无旧 `.npz` 开始的采集、求解、保存、验证、启动及中止条件。区分必须由操作员在实验台操作的步骤和纯离线计算。确保每一步保存新数据到正确路径；预期文件名/schema应通过代码检查确定。若旧历史模型存在，默认拒绝其被自动加载。

**交付 D：`docs/SOURCE_MANIFEST.md` 与 `docs/INTEGRATION_REPORT.md`**。列出每个实际采用的文件来自哪个 Drive URL、版本/时间、原始及整合后的 SHA-256（若可取得）、合并/修改原因、被淘汰的旧版本及未解决风险。报告实现了哪些修复、测试结果和硬件验证缺口；不要仅汇报“看起来完整”。

**交付 E：基本的 `preflight` / `self-test` 入口**，至少检查最新源码入口、校准结果是否齐全/匹配当前会话、必要 schema、参数单位与坐标系约定、运行模式和安全停止逻辑。测试工具不应自动发布机器人运动指令。允许新增运行帮助脚本与 `config.example`，用户明确不要求重装 ROS 或 Python libraries。

## 7. 推荐的第一版文件架构（仅为目标布局，按真实 import/path 调整）

```text
CISII_ISMR_Runnable_v1/
  README.md                         # 如何使用此包；什么已通过离线验证
  home_to_rcm_insert_v2_precision.py  # 2026-10-12 主控制器，首版保留相对位置以降低风险
  run_main.sh                       # 2026-10-12 唯一推荐启动入口
  trial_archiver.py                 # 20-trial 自动归档
  intr_util.py
  tip_inee_guard.py
  handeye_calibrate.py              # 主要运行脚本的顶层相对 import 尽量保持兼容
  handeye_capture_one.py
  pivot_calibration.py
  pivot_residual_fit.py
  rcmoffset_fit.py
  fast_pivot_fit_woodbury.py
  ...                               # 其余已有、验证为必要的 helpers
  microscope_autonomy/
    mic_perception.py
    mic_extrinsic_calib.py
    mic_ring_offset_verify.py
    mic_independent_verify.py
    mic_live_capture_ros.py
    mic_live_capture.py
    click_calib_points.py
    click_calib_points_per_touch.py
    correspondence_report.py
    ...
  calibration_tools/                # 在导入路径检查通过后才考虑归类，防止破坏调用
    camera_intrinsics_calibrate_charuco.py
    camera_intrinsics_calibrate.py
    needle_axis_diagnose.py
    ...
  analysis/
    measure_trial_holes.py
    save_manual_still.py
    ...
  config/
    run.example.env
    calibration_manifest.example.json
  scripts/
    preflight.py
    verify_calibration.py
    offline_smoke_test.py
  data/
    calibration/                   # 新实验生成，初始化为空；读取路径需显式适配
    raw_samples/
    logs/
    trials/
  docs/
    FILE_ARCHITECTURE.md
    FRESH_CALIBRATION_RUNBOOK.md
    SOURCE_MANIFEST.md
    INTEGRATION_REPORT.md
  tests/
  reference_only/                   # 可选择不打包历史 `.npz`，仅记录来源
```

**首版遵循“最小侵入式可运行”原则：** 许多研究脚本依赖同目录的 `intr_util.py`、`tip_inee_guard.py`，并用 `__file__` 计算路径。Codex 必须先分析所有 import 和文件 I/O，再决定是否将脚本搬到子目录。若重构路径成本高，优先维持平铺源码结构，单独生成 FILE_ARCHITECTURE 文档即可。`data/calibration/` 若与旧主程序寻找根目录 `.npz` 的方式冲突，可先使用受控配置变量或兼容链接；所有重定向必须有测试。最终运行包无需附带历史有效矩阵。

## 8. 工作顺序与验收标准

**先审计，再整合，再测试。** 第一次提交先输出现有仓库与 P0 代码的差异、逐功能 source mapping、硬编码/导入/输出位置检查和整合方案；随后立即在分支上完成有界代码改动，打包第一个 v1，不要停在报告阶段。优先修复“空标定状态下首次采样被拒绝”“旧针轴硬编码”“TipInEE 旧值锁定”“历史视觉修正参数”。保留近期主程序中控制阶段的接口和实际 RCM 旋转中心，任何修改需说明前后行为差异。

离线验收：能识别并拒绝旧模型；能够从新的示例/合成标定输入生成正确 schema 的输出；可以校验各个模块从主程序导入且路径正确；新针轴可从参数文件加载并参与计算；首次采样逻辑不要求待拟合的模型预先存在；能解析新 20-trial 的结构、运行独立验证和统计分析。实际机器人插入、毫米级精度、安全限位和组织交互必须由实验室操作者后续现场确认，不能用模拟测试替代。

在实机运动前，应由操作员核实 Trocar 固定、TipInEE 当前值、手眼与视觉标定有效性、显微镜注册置信度、表面法向方向、台架及眼球固定状态、自由空间运动及停止机制。不要让脚本在尚未完成标定检查、缺少人工确认时执行穿刺。

## 9. 可直接作为 Codex 首条消息的任务指令

请读取仓库现状及 `CISII_CODEX_HANDOFF.md`，将已有项目与 handoff 指向的 Google Drive 源码逐文件比较。创建独立 Git 工作分支，以 **2026-10-12 的 `01_code (1)` 主程序、启动器、显微镜模块** 为首选基线；从 **2026-08-25 Complete Package** 补齐从零重标所需的相机内参、Pivot 和工具标定脚本；以 **`ISMR 2027/package_2026-10-12`** 作为当时运行路径、标定 schema、20-trial 数据结构的历史参考。所有旧 `.npz` 禁止作为本次有效运行参数。先生成并交付 `CISII_ISMR_Runnable_v1.zip`、`FILE_ARCHITECTURE.md`、`FRESH_CALIBRATION_RUNBOOK.md`、`SOURCE_MANIFEST.md`、`INTEGRATION_REPORT.md`，以及可离线运行的 preflight/smoke tests；先求“可在原实验室电脑进行新标定并执行论文流程”，再考虑大型重构。检查并修复旧针轴默认覆盖、首次 ring-offset 采集依赖死锁、TipInEE 默认历史值、视觉校正旧数值/路径依赖。明确列出实际完成/无法验证的内容。如果 Codex 无法直接访问这些 Drive 资源，请马上按 manifest 列出需要用户下载上传的具体文件，继续处理本地可获得的部分，不要猜测缺失代码的实现或假装完成实机测试。
