# Pivot Calibration

This folder estimates `F_tip`: the fixed translation from the SHER end-effector
frame to the physical trocar tip. It fills the gap between the hand-eye
calibration (`F_b` / `T_cam2base`) and commanding the actual trocar tip to a
planned insertion point.

## Method

Seat the trocar tip in a fixed dimple, then record many end-effector poses while
changing the tool orientation. The tip should stay physically fixed.

For each recorded pose:

```text
p_tip_base = R_base_gripper * t_gripper_tip + p_gripper_base
```

Because the tip is fixed in the dimple, all predicted tip positions should equal
one unknown pivot point:

```text
R_i * t_gripper_tip + p_i = c_base
```

The script stacks all samples and solves:

```text
[R_i  -I] [t_gripper_tip] = -p_i
          [c_base       ]
```

The important output is `t_gripper_tip_mm`, the tool-tip/TCP offset in the
end-effector frame.

## Live Collection

`pivot_calibration.py` is manual/semi-manual. It does **not** command robot
motion. It only listens to `FrameEE`, records poses when you press Enter, then
solves automatically.

Prerequisites:

- The actual trocar/tool is mounted exactly as it will be used.
- The physical tip is seated in a stable dimple.
- ROS is publishing the SHER end-effector transform, usually
  `/SHER20/eye_robot/FrameEE`.

Run:

```bash
cd Autonomous-Trocar-Insertion/pivot-calibration
python3 pivot_calibration.py --robot-name SHER20
```

Controls:

- `Enter`: record current settled pose
- `s`: solve with current samples
- `w`: save result
- `d`: delete last sample
- `p`: print current pose
- `q`: quit

Recommended sampling:

- Record at least 12 samples.
- Use diverse roll/pitch orientations while keeping the tip seated.
- Avoid only translating the robot; pivot calibration needs orientation
  diversity.
- If the tip slips in the dimple, delete that sample.

You can print a manual pose prompt list with:

```bash
python3 pivot_pose_prompts.py
```

This prompt script does not move the robot. That is intentional: before the
TCP is known, commanding end-effector rotations can sweep the trocar tip.

For a more guided fully manual workflow, use:

```bash
python3 manual_pivot_calibration.py --robot-name SHER20
```

See `README_manual.md`.

## Offline Re-Solve

Every save writes a sample CSV. Recompute from CSV with:

```bash
python3 pivot_calibration.py --from-csv output/pivot_calibration_<timestamp>_samples.csv
```

## Outputs

Each saved calibration writes:

- `.npz`: machine-readable calibration and residual arrays
- `.json`: human-readable summary
- `_samples.csv`: raw recorded poses

Key fields:

- `t_gripper_tip_mm`: fixed end-effector-to-tip translation, in mm
- `pivot_base_mm`: estimated dimple point in robot base frame, in mm
- `T_gripper_tip_mm`: identity rotation plus `t_gripper_tip_mm`
- `residual_norms_mm`: per-sample tip consistency errors
- `rms_residual_mm`, `max_residual_mm`: quality metrics

## How To Use The Result

If you want the trocar tip at `p_tip_des_base_mm` with desired end-effector
orientation `R_base_gripper_des`, command the end-effector origin to:

```text
p_gripper_des_base_mm =
    p_tip_des_base_mm - R_base_gripper_des * t_gripper_tip_mm
```

This is the missing step that converts planned insertion points into robot
end-effector targets.

## Quality Checks

Good calibration should have:

- low residual RMS and max residuals relative to the project accuracy budget
- meaningful orientation spread, ideally around 20 deg or more between some
  sample pairs
- no loose tool mount or slipping dimple contact

If residuals are high, recollect with steadier dimple contact and more roll/pitch
diversity.
