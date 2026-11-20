# ISMR Runnable v1整合与验证报告

状态：**HARDWARE_UNVERIFIED / 已完成源码整合及离线验证，待实验室验收**。交付日期按用户时区为2026-11-19。本版没有发布任何机器人运动命令，没有调用实机驱动，不能据此声称完成物理插入或毫米级精度验证。

## 源码决策与工作区

独立分支 `codex/ismr-runnable-v1`，父提交 `fc4e04a`，worktree `next step/ISMR_Integration`。初始审计提交 `d43d8ac` 保存P0原始代码，随后在该分支整合。根工作目录本身不是Git仓库；原 `Autonomous-Trocar-Insertion` 仍在main，原有 `.DS_Store` 修改未动。`next step/code` 与历史数据未改写。运行时仅需拷贝内层 `CISII_ISMR_Runnable_v1`，外层仓库不是运行依赖。

A/P0、B历史快照、本地next step/code和附件的主程序/启动器/archiver逐字节相同。A显微镜模块优先，8月工具补齐缺失能力，不用较旧handeye覆盖10月版本。11月2日本地仓库新增相机诊断的曝光设置及readback已存在于采用主程序，未重复覆盖或替换探测器。所有实际采用文件的来源、源时间、原/整合SHA-256见SOURCE_MANIFEST及机器清单；上传/复制时间未用于判断新旧。

已读取附件论文，并渲染查看第4页的Stages/Holding a pivot与独立测量段；与实际主程序核对：ORIENT绕当前hover tip，REORIENT绕实测P。论文将每trial描述为7个闭环，代码还会执行返回HOME的RETRACT；统计应把HOME回程单列，不能把所有reached行机械统计为7个。原代码` t `连ORIENT→TUNNEL，`p`连REORIENT→PLUNGE，与逐阶段手动确认的文字表述存在粒度差异，本版保留代码行为并在README/HUD/手册明确提示。需要逐闭环分别确认的实验方案应另行修改并做现场验证。

## 完成的修复

|问题|当前行为|
|---|---|
|历史针轴覆盖|删除20260826轴默认覆盖；加载本会话needle_axis.npz。无有效新轴时不允许自动ORIENT。新增只读shaft采集、带符号/姿态覆盖/离散度检查的求解与独立验证。|
|首次ring-offset依赖闭环|collect阶段排除所有ring模型依赖，fixed/tangent均回归覆盖；handeye/registration/法向等上游和帧有效性保留。use阶段仍要求输出模型。|
|旧TipInEE canonical|默认取本次已选模型；尚无模型时取初始化显式记录的GUI值；继续与live ROS交叉核验，不关闭guard，拒绝NaN。|
|视觉修正旧数值|输入本次CSV、当前handeye/内参，另存候选；去除旧CSV/内参/9.4mm幅度/固定spread假设。加入多姿态、LOO、bootstrap、可配置物理限值。|
|ring旧geometry|额外发现nominal mark angle/radius和训练yaw也是旧实验常量；改由新配对样本拟合并随ring模型加载。|
|旧参数/路径自动加载|每会话隔离；主入口拒绝未登记、哈希变化、上游变化、初始化前参数及旧目录override；运行包不含npz。raw来源哈希和接受限值登记在session.json。可选origin一旦变化会使相关registration失效。|
|内参链不一致|手眼采集/验证改用同一本次self-cal；缺失或不兼容文件不静默回factory。新OpenCV自标定输出明确forward Brown模型，与runtime及视觉修正保持一致；这是相机几何接口修复，需要实机重投影复核。|
|归档位置|日志/照片/视频/campaign统一本会话，SHA256记录实际模型/代码/会话；空CSV不再导致归档报错。|
|分析缺口|提供campaign完整性、阶段指标、bootstrap CI及按trial留出的公共偏移分析，显式区分2D与3D输入。|

七个关键原函数 `auto_move`、`auto_orient_about_rcm`、`do_tunnel`、`compute_goals`、`desired_insertion_dir`、`zero`、`measure_shaft_direction_3d` 与源基线AST完全相同；保存其AST用于回归。没有修改固件、运动学、PI参数、目标姿态几何、插入深度、速度表或闭环终止规律。新增gate只在自动动作按键入口检查当前会话，calibration模式保留人工jog和原有撤回/停止路径。

## 离线验证

`python3 scripts/offline_smoke_test.py`：**21项测试全部通过**。详见 `OFFLINE_TEST_RESULTS.txt`。

覆盖全包Python编译、两个shell语法、9个Python CLI与shell帮助、复制到含空格新路径的初始化/标定CLI、空标定启动、禁止空参数插入、历史/未登记/篡改/外部路径拒绝、无效rotation/NaN拒绝、TipInEE mismatch、正负号与多姿态针轴、合成RCMoffset参数恢复、15触点仿射+LOO及实际保存schema、ring首采依赖fixed/tangent、合成ring模型与新几何输出、小于1mm新视觉修正恢复（不依赖旧9.4mm）、七函数AST一致、停止/旋转中心保留、合成trial归档及20-trial解析/LOTO。

测试机器Python3.11、NumPy/SciPy；OpenCV安装于临时测试目录（不进入ZIP、不改变实验室环境）。主模块导入测试使用显式ROS/RealSense消息导入替身，Publisher/Subscriber/init/pipeline均禁止实际调用。真实cv2参与仿射、投影和视觉修正测试；ROS/RealSense真实导入、设备打开、X11按键、触点采集、急停与驱动兼容性均未测试。旧ChArUco入口保留实验室OpenCV旧版ArUco API；若实验室升级OpenCV，需要先核对该入口兼容性，不声称在本机新OpenCV下可采集。

B历史六份标定文件仅用来读schema，trial01/trial20抽查JSON/config/precision字段结构；提取字段/类型/shape/哈希写入参考JSON，不把参数、照片或旧结果放进运行搜索路径。打包器排除所有npz、会话数据、缓存和macOS metadata，并输出独立SHA256文件。

## 首版限制与实验室验收

- 首版插入gate只支持`TIP_MODEL=rcmoffset`、`MIC_RCM_MODE=circle`、`MIC_CIRCLE_D405_MODEL=fixed`及双阶段模式；保留其他算法供离线比较，不能当作已通过的新实验模式。fixed offset受phantom训练姿态限制。
- 软件哈希/会话证据可阻止意外旧参数复用，不能证明采样实际质量或识别人为把旧数值另存为“新数据”。必须真实重新采集并独立核验，不能仅更新文件时间或登记说明。
- 有效schema和低LOO不证明无载tip准确、手眼深度准确、法向符号正确或安全限位有效。可配置接受阈值必须由实验方案预先确定；本版不强行重现历史误差幅度。
- 原撤针逻辑在缺少可确定轴时可能回退直接HOME，保留未改；异常状态不得盲目按`b`。实机停止机制、运动限制、两阶段接触、组织交互、器械固定与回程路径须逐项现场确认。
- Shift按键在旧X11有失效记录；旧`H/C/M`命令仍需注意focus，新增手册强调触点9–15用无修饰标点。
- 原`handeye_reproj_refine.py`的PROMOTE默认行为保留为可选工具；手册要求研究时PROMOTE=0。原`fast_pivot`/augmented工具不自动纳入运行基线。
- 完整论文Fig.7/8仍需要真实20-trial原始记录、视频时间对齐及独立entry测量。新增报告不生成缺失观测、不硬编码论文统计。仅有hole clicks时输出投影2D偏移，3D offset需额外独立XYZ数据。现有测孔工具使用post-retract still，而论文entry指标使用Tunnel末端录像标注，二者不能直接视为相同测量协议。
- `capture_needle_axis.py`为本轮新增硬件采集入口，复用原shaft算法但采集同步/GUI使用未经过台架验证。须先在自由空间确认照片、ROS姿态和符号的一致性。

没有因Google Drive权限无法读取而缺失的必需源码。若要进一步严格复现论文全部图表，仍需提供/采集对应的原始视频标注、trial级独立针孔位置及三维测量证据，并明确排除/失败trial规则。这不阻塞本轮源码包交付，也不将历史结果算作本次实验。
