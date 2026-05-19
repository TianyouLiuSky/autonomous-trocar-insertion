# Manual Pivot Calibration

Use this workflow when a person will physically hold or cooperatively move the
trocar tool through the calibration motions. The script does **not** command any
robot motion. It only listens to the SHER end-effector pose topic and records
poses when you press Enter.

The output is the same `F_tip` quantity used by the rest of the project:

```text
t_gripper_tip_mm
```

This is the fixed translation from the SHER end-effector frame to the physical
trocar tip.

## Physical Setup

1. Mount the real trocar/tool exactly as it will be used for insertion.
2. Create or use a small stable dimple.
3. Place the trocar tip into the dimple.
4. Keep the tip seated in the same point for the entire calibration.
5. Move only by gently changing roll/pitch around the dimple.

The dimple is the fixed pivot point. If the tip slides, that sample should be
deleted or the dataset should be recollected.

## Run

```bash
cd /Users/tobiichi-orieda/Documents/CIS-II/Autonomous-Trocar-Insertion/pivot-calibration
python3 manual_pivot_calibration.py --robot-name SHER20
```

If the robot pose topic is not `/SHER20/eye_robot/FrameEE`, pass it directly:

```bash
python3 manual_pivot_calibration.py --topic /SHER20/eye_robot/FrameEE
```

If your transform topic is in meters rather than millimeters:

```bash
python3 manual_pivot_calibration.py --translation-scale-to-mm 1000
```

The current SHER code in this project appears to use millimeters, so the default
scale is `1.0`.

## Prompted Movements

The script prompts these manual poses:

- neutral
- positive roll
- negative roll
- positive pitch
- negative pitch
- four combined roll/pitch poses
- smaller roll/pitch poses
- neutral repeat

At each prompt:

1. Move the tool to the requested orientation.
2. Keep the physical tip seated in the dimple.
3. Let the pose settle.
4. Press Enter to record.

Commands during collection:

- `Enter`: record current pose
- `p`: print current pose
- `s`: skip the current prompt
- `b`: delete previous sample and go back
- `q`: stop early and solve with collected samples

## Saving And Outputs

By default the script solves and saves at the end:

- `manual_pivot_calibration_<timestamp>.npz`
- `manual_pivot_calibration_<timestamp>.json`
- `manual_pivot_calibration_<timestamp>_samples.csv`
- `manual_pivot_calibration_<timestamp>_samples.csv.labels.txt`

Important fields:

- `t_gripper_tip_mm`: end-effector to trocar-tip translation
- `pivot_base_mm`: estimated dimple location in robot base coordinates
- `rms_residual_mm`: overall pivot consistency error
- `max_residual_mm`: worst sample error
- `max_pairwise_rotation_deg`: how much orientation diversity you achieved

## Quality Rules

Aim for:

- at least 12 samples
- max pairwise orientation spread of at least 15-20 degrees
- low residuals; the default warning threshold is `0.5 mm`

High residuals usually mean one of:

- the tip slipped in the dimple
- the tool mount moved
- the robot pose stream has unit/frame issues
- the samples did not include enough roll/pitch diversity

## How The Result Is Used

For a desired trocar-tip target in robot base coordinates:

```text
p_gripper_des_base_mm =
    p_tip_des_base_mm - R_base_gripper_des * t_gripper_tip_mm
```

That gives the end-effector origin position needed to put the physical trocar tip
at the planned target.
