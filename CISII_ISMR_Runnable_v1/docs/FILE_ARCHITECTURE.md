# 文件与调用架构

工作分支 `codex/ismr-runnable-v1`，基于原 `Autonomous-Trocar-Insertion` 的 `fc4e04a`，独立 worktree 为已有 `next step/ISMR_Integration/`。原工作区及 `next step/code` 没有改写。运行包在该 worktree 的 `CISII_ISMR_Runnable_v1/`。

```text
CISII_ISMR_Runnable_v1/
  README.md
  run_main.sh                         实验入口，保留 20-trial 角度表
  home_to_rcm_insert_v2_precision.py   ROS控制、D405几何、阶段状态机、触点/法向采集
  runtime_session.py                  会话路径、schema、哈希与上游依赖检查
  intr_util.py / tip_inee_guard.py     共享当前内参、GUI TipInEE一致性检查
  camera_intrinsics_calibrate*.py     ChArUco / 棋盘格内参采集求解
  handeye_*.py                        手眼采集、求解、诊断、视觉偏差区分
  pivot_*.py / rcmoffset_fit.py        固定针尖采样、拟合与交叉检查
  fast_pivot_fit_woodbury.py           可选快速模型比较；首版不自动提升 augmented
  tip_offset_apply_visual_correction.py  新数据视觉修正，另存候选模型
  needle_axis_calibrate.py            新轴求解、符号/多姿态检查、独立集验证
  needle_axis_diagnose.py             原历史分析工具，仅诊断，不作有效轴生成入口
  rcm_definition.py                  几何说明/数值示例，不是机器人控制器
  trial_archiver.py                  从 trial JSON/CSV/照片生成 campaign
  measure_trial_holes.py / save_manual_still.py  人工测孔/补拍
  microscope_autonomy/
    mic_perception.py                limbus/fiducial与眼球附着目标
    mic_extrinsic_calib.py            触点像素→base仿射；主程序及ring工具导入
    click_calib_points_per_touch.py   按每次触点照片点击（推荐）
    click_calib_points.py             单图点击（可选）
    mic_ring_offset_verify.py         paired ring→mark模型、LOO及新几何参数
    mic_independent_verify.py         重复图像gate、独立触点验证
    mic_live_capture*.py              Leica实时帧relay，ROS版/视频设备版二选一
    phantom_normal_repeatability.py   法向重复性（离线）
    correspondence_report.py          两视觉目标差异、CI、Bland–Altman（离线）
  scripts/
    session.py / preflight.py         初始化、产物登记与分阶段检查
    capture_needle_axis.py            只读ROS+D405针杆采样，不发运动命令
    env.sh / with_session.sh          终端路径与Python导入辅助
    offline_smoke_test.py             无硬件回归入口
    build_release.py                 可重复打包；排除所有实测数据和历史npz
  analysis/campaign_report.py         trial完整性、阶段误差、CI、LOTO偏移
  config/run.example.env              配置说明；无历史标定值
  tests/                             数学、拒绝路径、状态边界与归档回归
  docs/                              新手册、来源表、整合报告、原论文及历史说明
  data/README.md
  data/sessions/<name>/               仅实验室初始化/运行后生成
    session.json                     会话身份、初始TipInEE、验收限值、哈希和证据
    calibration/                     新npz
    raw_samples/                     CSV/JSONL/点击/触点照片/相机capture/法向历史
    logs/                            precision/correspondence/orient trace/debug
    trials/
      trial_plunge_stills/            trial JSON、照片、hole_measurements.csv
      recordings/                    视频
      campaign/                      trialNN_*/配置快照、分割CSV、轨迹、汇总
```

保留顶层工具平铺，避免破坏 `intr_util`、`tip_inee_guard` 导入；显微镜代码保留子目录。各脚本由 `__file__` 找代码根，运行时解析当前会话路径，不依赖旧 `/home/eyebrp/...`。用户设置的旧路径环境变量在主程序 preflight 被拒绝。CLI 显式输入/输出使用手册里的相对路径。

依赖：

```text
实际标定板 → cam_intrinsics → handeye → 独立针杆采样 → needle_axis
当前GUI TipInEE → pinned pivot samples → RCMoffset → 视觉核对/可选修正
                                                ↓
                            Leica触点+每触点照片+点击 → mic_extrinsic
handeye + mic_extrinsic + 当前法向 + live配对样本 → ring offset + fiducial几何
上述已登记模型 + 每trial新normal/target lock → 分阶段插入 → trial JSON/CSV/照片
                                     ↓
               archiver → campaign → 独立测量、correspondence、LOTO统计
```

`cam_intrinsics.npz`：fx/fy/ppx/ppy、coeffs(5)、width/height、n、fit_rms_px、loo_rms_px、distortion_model；`handeye_T.npz`：R(3×3)、t(3, mm)、m及求解指标。`tip_offset_rcmoffset.npz`：d_const(3, mm)、K(mm)、tip_inee_at_calib(3, mm)、n和LOO。`needle_axis.npz`：axis_ee(3, 单位向量)、frame=EE、units=unitless、n、max_spread_deg、max_allowed_deg、orientation_spread_deg、source_sha256。`mic_extrinsic_calib.npz`：A(3×3)、px_uv、base_xyz、LOO、n_calib_points；A乘[u,v,1]输出base XYZ(mm)，不是SE(3)变换。`ring_to_entry_offset.npz`：offset_mark0(3, mm)、n_samples_mark0、loo_rms_mm_mark0、mark0_theta_deg、mark0_radius_px、training_yaw_deg（均来自新样本）。`phantom_normal.npz`：point_base、normal_base、normal_cam、t、plane_rms_mm等。

实验必需：当前主链及上表新模型。可选：origin correction（实测需要时）、视觉常量修正、augmented比较、reprojection refine、tangent circle模型。历史工具的可用性不等于已通过本轮硬件验证。首版 insertion gate 只接受 rcmoffset + circle/fixed；其他模型可离线研究，不能静默切入实验。
