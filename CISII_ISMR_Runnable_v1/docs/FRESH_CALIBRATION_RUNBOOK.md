# 从零标定、Registration与插入操作手册

全链路 **HARDWARE_UNVERIFIED**。这是实验室现有设备上的台架复现流程；本机没有连接 SHER、D405 或 Leica，不能以离线测试代替现场运动/急停/精度验收。论文随包在 `docs/ISMR27_0007_MS.pdf`；旧 README/协议中的历史数值不可直接套用。

## 0. 新会话与路径

解压后进入包根目录，在实验室终端加载原有 ROS/驱动环境。无需在此重装环境。当前 GUI TipInEE 必须从实机读取；以下变量先由操作者填写（不要照抄历史矩阵或旧针轴）：

```bash
# X Y Z 是实际 GUI TipInEE，单位 mm；替换后执行
python3 scripts/session.py init lab_YYYYMMDD_toolID --tip X Y Z
export ISMR_SESSION=lab_YYYYMMDD_toolID
source scripts/env.sh
python3 scripts/preflight.py --phase calibration
python3 scripts/session.py paths
```

`C/R/L/T` 分别指本会话 calibration/raw_samples/logs/trials。每个终端都设置同一 `ISMR_SESSION` 并 `source scripts/env.sh`。会话名只能用字母、数字、下划线、短横线。换针、改变 GUI TipInEE、相机/支架重装、机器人原点变化：另建会话并重新标定，不覆盖旧会话。

`session.py accept` 表示操作者已查看新数据与独立检查，并登记 SHA-256 与证据；它不会控制机器人。所有证据文件必须在本会话下、晚于初始化时间。`--max-rms` 必须填本次台架预先确定的正数验收限值：内参是 px，其余相应模型是 mm。下面 `$CAM_LIMIT_PX`、`$TIP_LIMIT_MM`、`$MIC_LIMIT_MM`、`$RING_LIMIT_MM`、`$AXIS_LIMIT_DEG`、`$CORR_LIMIT_MM`、`$CORR_LIMIT_PX`、`$CORR_SIGMA_MM` 由操作者依实验方案设置；没有统一默认值。

先核对急停、零速停止和控制权；同一时间只有一个运动控制进程。D405 采集程序之间也应互斥使用相机。标定模式主界面仍允许人工 jog，`g/t/i/p` 被禁用；不要把“calibration”理解为无运动。

## 1. D405 RGB内参（台架采集+离线求解）

使用真实测量过的标定板。ChArUco脚本默认6×6、10mm方格/6.8mm marker，并不保证与你的板相同；设置 `CHARUCO_SQUARES_X/Y`、`CHARUCO_SQUARE_MM`、`CHARUCO_MARKER_MM`、`CHARUCO_DICT` 后再运行。不要把旧眼眼标定板的参数当成本脚本默认。RGB模式为640×480，与主程序一致。

```bash
python3 camera_intrinsics_calibrate_charuco.py
# 无ChArUco板时，使用经确认规格的棋盘格替代：
# python3 camera_intrinsics_calibrate.py
```

保持板平整，覆盖图像中心、边缘、不同距离及倾角，建议20张以上。ChArUco窗口 `c` 采集、`u` 撤销、`l` 列出、`x` 删除、`s` 求解、`q` 退出；程序保存每次 capture 到 `$R/captures_session/`。查看覆盖率及LOO，不能只看训练误差。输出 `$C/cam_intrinsics.npz`。棋盘格入口按自己的窗口提示采样；两入口输出同schema。新输出系 OpenCV forward Brown 畸变，本版内参读取器用同一约定；不能静默回退 factory。

```bash
python3 scripts/session.py accept cam_intrinsics.npz --sources "$R"/captures_session/* --max-rms "$CAM_LIMIT_PX" --note '实际板规格、覆盖率、LOO及RGB模式已检查'
```

上面通配符必须只包含文件；若 capture 目录另有子目录，逐项列出其中原始图像/角点文件。棋盘格入口未持久化逐帧原图，需操作者额外保存板照片/求解终端报告作证据，登记这些文件；推荐ChArUco入口。

## 2. Hand-eye：D405→robot base（台架）

固定针杆上的ArUco/实际标记，保持标记与工具刚性关系，核对FrameEE坐标、mm单位和xyzw四元数。所有手眼脚本使用当前内参。

```bash
python3 handeye_calibrate.py
```

在原GUI控制机器人到不同位置、不同姿态，静止后空格/`r`记录；`m`切自动ArUco/点击模式，`u`撤销，`s`求解保存，`q`退出。至少12个有效多样姿态，推荐覆盖实际工作区；CSV `$R/handeye_data/pairs.csv`，照片在其 `frames/`，输出 `$C/handeye_T.npz`。`x`会清空当前采样，勿误按。

独立采集新姿态到单独验证文件，不能重复训练数据：

```bash
# 主采集程序退出、释放相机；每个新静止姿态执行一次
python3 handeye_capture_one.py "$R/handeye_val.csv"
# 收集多姿态后；不发布机器人运动，但会短暂打开相机读取当前内参
python3 handeye_compare_on_val.py "$R/handeye_val.csv" "$C/handeye_T.npz" "$C/handeye_T.npz" > "$L/handeye_validation.txt"
python3 scripts/session.py accept handeye_T.npz --sources "$R/handeye_data/pairs.csv" "$R/handeye_val.csv" "$L/handeye_validation.txt" --note '独立姿态重投影与空间精度已检查'
```

重复同一候选参数是调用原比较工具所需的两个候选入口，只用于输出独立集分数，不声称比较出了更优模型。也可用 `handeye_live_recheck.py` 做新图像核验。`handeye_reproj_refine.py` 属可选优化；默认会提升参数，研究时设置 `PROMOTE=0`，检查 held-out 后再有意提升并重新登记所有下游。独立点击只能约束相机射线横向误差，不能证明深度方向准确。

## 3. Trocar Pivot / RCMoffset（台架采样+离线拟合）

确认本会话初始 TipInEE 与GUI一致，针固定、支点不滑动，避免把受力弯曲当作真实无载针尖。不要为复现旧值调GUI。采样工具不命令机器人运动：操作者按原实验方式固定针尖并在自由空间变换姿态。

```bash
python3 pivot_calibration.py
```

终端空格或 `c` 记录、`p` 打印、`u`撤销、`s`保存、`q`退出。输出 `$R/pivot_samples.csv` 与 `$C/tip_offset_pivot.npz`。覆盖工作姿态、多个轴向，建议至少20–40个稳定姿态并查看条件数/LOO。替代单次采集：

```bash
python3 pivot_capture_one.py "$R/pivot_independent.csv"
python3 rcmoffset_fit.py "$R/pivot_samples.csv" --save > "$L/rcmoffset_fit.txt"
python3 scripts/session.py accept tip_offset_rcmoffset.npz --sources "$R/pivot_samples.csv" "$L/rcmoffset_fit.txt" --max-rms "$TIP_LIMIT_MM" --note '当前TipInEE、姿态覆盖与LOO核验'
```

新模型的 `tip_inee_at_calib` 来自本会话显式基线，不能为空/NaN。保存拟合是离线计算，GUI基线不能在采样中途改变。独立 pin sweep 用 `tip_pin_check_v3_fullpose.py` 检查；该工具读 rigid pivot 并与实时基线核对。`fast_pivot_fit_woodbury.py`、`pivot_residual_fit.py` 是比较/诊断；首版不自动选择augmented。

## 4. 视觉针尖/原点核对（台架，可能人工点动）

先运行 `python3 handeye_vs_tip_discriminate.py`，在多个方向与空间位置点击真实针尖，`s`保存CSV（打印具体路径），`q`退出。该程序自带角向jog，会发真实速度命令；不要同时开主控制器。比较base/tool常量残差，避免在单姿态上凭叠图调参。

工具坐标常量偏差确有证据时，离线产生新候选，使用当前CSV，拒绝任何历史9.4mm幅度假设：

```bash
python3 tip_offset_apply_visual_correction.py --csv "$R/本次discriminator文件.csv" --out "$C/tip_offset_corrected_candidate.npz" --max-correction-mm "$CORR_LIMIT_MM" --max-rms-px "$CORR_LIMIT_PX" --max-sigma-mm "$CORR_SIGMA_MM"
```

输出delta、bootstrap sigma、姿态spread、LOO及新GUI TipInEE。先保留原模型，核验候选并记录GUI变更。提升候选前退出所有控制器，备份本会话旧模型到raw_samples，再将候选复制为 `$C/tip_offset_rcmoffset.npz`，GUI按输出设定并独立重新核验，然后重新 `accept` 模型（把本次CSV/报告列入证据）。从此所有后续registration必须按新模型重做；若已完成，则哈希依赖会阻止旧注册继续使用。更换器械应直接开新会话。

若base原点偏差有证据，可在主程序 `c` 键按提示点击并测量；已有背景深度/40mm上限仍保留。`c`产生 `$C/origin_correction.npz` 后必须登记（sources可用相关日志/照片及诊断报告）。仅base横向诊断向量也可用 `origin_correction_from_discrim_csv.py VX VY VZ --handeye "$C/handeye_T.npz" --out "$C/origin_correction.npz" --source-note '本次诊断'`；不能从历史抄向量。登记命令：`python3 scripts/session.py accept origin_correction.npz --sources "$L/本次诊断报告.txt" --note '当前原点修正已独立核验'`。没有证据时不生成该可选文件，默认零修正。若运行中改了 `c`，退出、检查并登记后才能继续自动阶段。

## 5. 独立针轴（先采集，再允许自动ORIENT）

为了避免“必须先ORIENT才能标定ORIENT针轴”的新闭环，本版提供只读采集入口。退出所有占用D405的程序。在GUI中摆放自由空间的不同静止姿态，重新打开采集或保持窗口运行：

```bash
python3 scripts/capture_needle_axis.py --out "$R/axis_fit.csv"
```

点击针杆ROI中心，静止时 `c` 保存一次通过原3D线拟合质量gate的测量及照片；`q`退出。此脚本不创建Publisher，但GUI人工运动由操作者掌控。至少3个有效姿态且覆盖≥20°，推荐更多方向。物理确定EE局部从针尾朝针尖的大致方向，用它解析3D线的正负号，**不是旧针轴常量**：

```bash
# AX AY AZ 替换为本次物理方向的粗参考（只用来选正负）
python3 needle_axis_calibrate.py "$R/axis_fit.csv" --sign-reference AX AY AZ --max-spread-deg "$AXIS_LIMIT_DEG" --report "$R/axis_fit_report.json"
# 独立采集新的姿态到另一份CSV
python3 scripts/capture_needle_axis.py --out "$R/axis_validation.csv"
python3 needle_axis_calibrate.py "$R/axis_validation.csv" --sign-reference AX AY AZ --max-spread-deg "$AXIS_LIMIT_DEG" --check "$C/needle_axis.npz" --report "$R/axis_validation_report.json"
python3 scripts/session.py accept needle_axis.npz --sources "$R/axis_fit.csv" "$R/axis_validation.csv" "$R/axis_validation_report.json" --note '物理符号与独立姿态验证已通过'
```

不同文件但相同字节会被拒绝作为独立验证。axis加载器校验EE坐标、单位长度、符号、样本数与离散度；不默默丢弃离群值。主程序ORIENT/REORIENT实际使用新文件。独立3D测量仍共享hand-eye旋转，其系统偏差需现场评估。

## 6. Leica Registration（台架触点+逐图点击+离线拟合）

先启用原有DeckLink ROS视频源。在一个终端运行relay，默认只保留最近5帧；此阶段不要同时选择 `MIC_ROS_LIVE=1` 来依赖不同照片路径：

```bash
python3 microscope_autonomy/mic_live_capture_ros.py --interval 0.3
# 另一终端，完成同一会话环境设置后：
bash run_main.sh
```

人工jog触及实际位置并静止，在主窗口按 `1..8`，再用 `, . / ; ' - =` 对应点9..15。**无需Shift**，避免触发小写jog。建议15点，覆盖目标区域并保留独立验证触点。每个触点同步保存当前针尖base位置到 `$R/mic_calib_touches.jsonl` 和对应Leica照片到 `$R/mic_calib_touch_stills/`。检查照片时间、针尖位置、触点标签对应，不能凭记忆在另一张图猜触点。退出主程序再计算：

```bash
python3 microscope_autonomy/click_calib_points_per_touch.py --touches "$R/mic_calib_touches.jsonl" --stills-dir "$R/mic_calib_touch_stills" --out "$R/mic_clicks.json"
python3 microscope_autonomy/mic_extrinsic_calib.py --fit --loo --touches "$R/mic_calib_touches.jsonl" --clicks "$R/mic_clicks.json" --out "$C/mic_extrinsic_calib.npz" > "$L/mic_registration.txt"
python3 scripts/session.py accept mic_extrinsic_calib.npz --sources "$R/mic_calib_touches.jsonl" "$R/mic_clicks.json" "$L/mic_registration.txt" --max-rms "$MIC_LIMIT_MM" --note '逐触点配对、LOO与工作区覆盖核验'
```

数学最少3点，登记/LOO至少5点；不能以恰好可解代替准确度。独立检查：用未参加拟合的静止图执行 `mic_independent_verify.py --gate 图1 图2 图3 --calib "$C/mic_extrinsic_calib.npz"`；主界面`9`采集独立campaign触点到 `$R/mic_campaign_touches.jsonl`，再 `--campaign --touches "$R/mic_campaign_touches.jsonl" --calib "$C/mic_extrinsic_calib.npz" --log "$L/mic_campaign_log.csv"` 按工具提示点击。具体gate由原代码执行，照片文件名需替换成当前采集文件。

## 7. 首次 ring offset（无需旧输出）

重新启动标定模式主界面。`z`锁定D405 ring，`n`采本次法向，确认Leica帧新鲜、两个fiducial识别正确后按`0`。这是静态配对，不运动。重复采集≥5组，建议更多独立重新摆放/锁定；fixed-vector模型只适用于本次训练姿态附近，不能任意转动phantom。新采样会记录mark0角度/半径和两个mark的base坐标，不要求ring模型已经存在。

```bash
python3 microscope_autonomy/mic_ring_offset_verify.py --fit --loo --samples "$R/ring_offset_samples.jsonl" --out "$C/ring_to_entry_offset.npz" > "$L/ring_fit.txt"
python3 microscope_autonomy/mic_ring_offset_verify.py --report --npz "$C/ring_to_entry_offset.npz"
python3 scripts/session.py accept ring_to_entry_offset.npz --sources "$R/ring_offset_samples.jsonl" "$L/ring_fit.txt" --max-rms "$RING_LIMIT_MM" --note '固定模型适用姿态、两fiducial几何、LOO及未遮挡状态核验'
```

退出重启主程序以加载新模型几何。训练yaw超限、遮挡、法向不稳定或held-out误差恶化时重新采样，不放宽阈值来追旧论文数字。`--fit-circle`保留为可选分析，首版插入不允许切换tangent模型。

## 8. 20 trials插入（全部为实机人工确认）

先完成新标定和自由空间停止检查，并核对针具固定、当前TipInEE、原点、显微镜注册、目标可见区域、法向朝外及phantom固定。不要在标定尚未完成时执行穿刺。

```bash
python3 scripts/preflight.py --phase insertion
# 新终端同一会话，启动归档：
python3 trial_archiver.py > "$L/campaign_archiver.log" 2>&1 &
# 控制终端：
bash run_main.sh --insertion
```

每个trial流程：

1. 在安全的初始HOME位置按 `h`，锁定HOME并应用本trial角度表；新trial会清除旧法向fresh状态和视觉目标锁。
2. `z`锁D405 ring，`n`重新估计法向。法向应朝外；确认入针用 `-n` 指向phantom内侧。没有可靠新法向就停止。
3. 按 **反引号** 做Leica/D405交叉检查并锁定显微镜眼球附着目标。确认fiducial身份、scheduled site和照片新鲜。显微镜目标驱动控制；D405估计仅作粗核验。不要以 `x` 的粗ring目标替代论文目标。
4. 人工确认后按 `g` 到8mm悬停。检查空间和目标。
5. 人工确认后按 **`t`：ORIENT绕当前悬停针尖，然后自动TUNNEL至E后1.5mm**。中间没有第二次按键暂停。成功时内部RCM取实测通道末端P，不是未到达的命令点。`i`仅为人工确认后的TUNNEL重试；不要重复穿刺。
6. 人工确认通道成功后按 **`p`：绕P REORIENT到内法向，再自动PLUNGE 3mm**。成功后自动采集仍插入时的trial照片/JSON。
7. 人工确认可撤针后按 `b`：轴向撤针（1mm/s）、抬至clearance（默认20mm）、返回HOME，保存post-retract照片及结束时间。
8. 检查本trial归档、独立轴测量状态和照片，再开始下一trial。20次只是计划长度，不能把缺失/失败trial补成成功。

`run_main.sh`保留10月12日20项site与25–45°入针角表、PIERCE_SPEED=6mm/s；位置/角度阈值0.4mm/0.3°。这些来自论文/原代码，未在本机证明适合现场。临时改角度/速度必须记录，不能宣称是同一实验。`--trocar`仅在当前已有真实插入RCM且需恢复手动旋转补偿时使用，语法 `bash run_main.sh --trocar --insertion`；组织内不要任意手动jog。

空格会中止当前自动动作并发零速；`q`退出，finally多次零速并释放相机。中止后不要依赖屏幕旧状态直接继续；检查针与组织位置、重新核验再决定回退/重试，必要时用硬件急停。原程序仍保留在缺乏撤针轴信息时直接HOME的历史回退，本版没有更改控制数学；这种异常状态不得按 `b`，应现场安全处置。Shift键在旧X11窗口有记录的不可靠行为，`H/C/M`等修改键操作要先确认focus；触点9–15已使用无修饰标点。

## 9. 归档、测量与离线分析

`trial_archiver.py`轮询trial JSON，归档照片、按时间窗口截取的precision/correspondence CSV、姿态轨迹、config_snapshot（含SHA-256、会话与环境）和视频。检查 `campaign_trials.csv` 的finalized及缺失字段，查看归档日志是否有error。停机后终止该后台归档进程。补拍：`python3 save_manual_still.py`；测孔：`python3 measure_trial_holes.py`（点击针孔，按窗口提示确认，`--summary`汇总）；输出 `$T/trial_plunge_stills/hole_measurements.csv`。

```bash
python3 microscope_autonomy/correspondence_report.py "$L/correspondence_log.csv" --no-plot > "$L/correspondence_report.txt"
python3 microscope_autonomy/phantom_normal_repeatability.py "$R/phantom_normal_samples.jsonl" > "$L/normal_repeatability.txt"
python3 needle_axis_diagnose.py > "$L/needle_axis_diagnostic.txt"
python3 analysis/campaign_report.py --campaign "$T/campaign" --holes "$T/trial_plunge_stills/hole_measurements.csv" --calib "$C/mic_extrinsic_calib.npz" --out "$L/campaign_report.json"
```

新增report检查20-trial结构、closed-loop reached误差/时间、失败行、LOTO公共偏移与误差CI。孔位像素经本次A的XY部分转换，只能报告**投影2D**偏移；不能凭2D孔照片恢复论文的完整3D公共偏移。若另有独立3D测量CSV，列为 `trial,entry_base_x,entry_base_y,entry_base_z,target_base_x,target_base_y,target_base_z`（mm），用 `--entries "$R/entry_3d.csv"` 额外计算3D LOTO。禁止用原论文统计填空。Fig.7/8的全部视频标注、轨迹图和测量协议还需后续数据与人工对齐；本版不声称完全复现图表。

## 中止/重采条件

标定产物缺失、非有限数、坐标/分辨率/单位不一致、哈希或上游已变化、LOO超过预定限值、独立验证失败、针具/GUI参数改变、phantom/相机移动超模型适用范围、帧过期/遮挡、法向反号、运动stall、停止机制未确认，均先停止并定位。不要把“程序返回reached”当作物理精度证据，也不要重新拷入历史npz解锁。
