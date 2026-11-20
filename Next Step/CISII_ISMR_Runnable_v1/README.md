# CISII ISMR Runnable v1

用于原实验室电脑的重新标定与 ISMR 双阶段插入代码包。**状态：HARDWARE_UNVERIFIED**。本机完成离线检查，不代表已成功连接机器人、完成插入或验证精度。已有 ROS、SHER 2.1、D405、DeckLink/Leica 环境由实验室继续使用。

从本目录开始操作。复制到实验室时，只需整个 `CISII_ISMR_Runnable_v1/`，无需外层 Git worktree。所有新数据落在 `data/sessions/<会话名>/`；不随包附带历史矩阵、样本或 trial 数据，也不预分配存储空间。

1. 阅读 [完整操作手册](docs/FRESH_CALIBRATION_RUNBOOK.md)，先记录当前 GUI TipInEE、实际标定板尺寸及台架配置。
2. `python3 scripts/session.py init 会话名 --tip X Y Z`（X/Y/Z 替换为当前实测 GUI 数值，单位 mm）。按输出 `export ISMR_SESSION=会话名`，再 `source scripts/env.sh`。每个终端都要设置。
3. 从相机内参开始，依次完成 hand-eye、pivot/RCMoffset、独立针轴、Leica registration、ring 交叉验证，并登记新产物及原始证据。
4. 标定/人工点动界面：`bash run_main.sh`。**人工 jog 仍会真实运动**；本模式禁用 `g/t/i/p` 自动阶段。
5. 全部新标定通过并在实验台确认后：`bash run_main.sh --insertion`。逐 trial 使用 `h → z → n → 反引号 → g → t → p → b`。**`t` 连续执行 ORIENT+TUNNEL；`p` 连续执行 REORIENT+PLUNGE**，按键前必须确认该组动作。空格中止当前自动运动，`q` 退出并发送零速；硬件急停仍须现场确认。
6. 归档、测孔、独立验证与统计命令见手册最后部分。

离线复查：`python3 scripts/offline_smoke_test.py`。需要现有 NumPy、SciPy、OpenCV；不需要 ROS/相机。测试中的 ROS/RealSense 只是禁止调用硬件的导入替身，不能证明硬件 API 兼容。

[文件架构](docs/FILE_ARCHITECTURE.md) · [源码与 SHA-256](docs/SOURCE_MANIFEST.md) · [整合、测试与局限](docs/INTEGRATION_REPORT.md)

原始 README/协议保留作历史说明；其中日期、绝对路径、旧参数和“已成功”描述不代表本版状态，以本 README 和新手册为准。固件没有修改。直接 Eye-eye 6-DoF 配准不属于本文必需链路。
