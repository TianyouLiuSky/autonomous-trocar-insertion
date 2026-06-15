# Force Data Collection

This directory contains a standalone workflow for collecting trocar penetration
force at a fixed direct-down or 30-degree entry angle.

## Directory Layout

```text
force_data_collection/
├── code/
│   ├── direct_down_insertion.py
│   ├── insertion_30deg.py
│   ├── fixed_angle_teleop.py
│   ├── force_recorder_ui.py
│   ├── analyze_force_session.py
│   └── tests/
└── data/
    └── <timestamped experiment directories>
```

Generated experiment data under `data/` is ignored by Git.

## Important Angle Convention

The software cannot determine the phantom surface normal from robot pose alone.
Before launching either insertion script, physically align the tool so that it
is directly down and normal to the phantom surface. The startup orientation is
captured as the **0-degree reference**.

- `direct_down_insertion.py` locks this startup orientation.
- `insertion_30deg.py` rotates 30 degrees from this startup orientation about
  the tool-local Y axis, then locks the resulting orientation.

Use `--tilt-axis local-x` or `--tilt-sign -1` if the experimental fixture needs
the tilt in a different plane or direction. Verify the direction with the
phantom clear of the tool before an insertion trial.

## ROS Topics

Defaults for `--robot-name SHER20`:

| Purpose | Topic |
|---|---|
| Robot pose | `/SHER20/eye_robot/FrameEE` |
| FBG force | `/SHER20/eye_robot/FBGForcesTip` |
| Linear velocity command | `/SHER20/eyerobot2/desiredTipVelocities` |
| Angular velocity command | `/SHER20/eyerobot2/desiredTipVelocitiesAngular` |

The recorder stores all four available `FBGForcesTip` values. The physical
meaning and sign of each channel must be confirmed against the installed force
sensor calibration before interpreting the results.

## Operator Workflow

Use two terminals on the robot computer.

1. Place the phantom and tool in their correct experimental positions.
2. With the tool clear of the phantom, align it directly down relative to the
   phantom surface. This pose defines 0 degrees.
3. Start the recorder UI:

   ```bash
   cd force_data_collection/code
   python3 force_recorder_ui.py --robot-name SHER20
   ```

4. Start one fixed-angle teleoperation script:

   Direct down:

   ```bash
   python3 direct_down_insertion.py --robot-name SHER20
   ```

   30-degree entry:

   ```bash
   python3 insertion_30deg.py --robot-name SHER20
   ```

5. For the 30-degree script, confirm that the tool has clearance to rotate.
   The script rotates in place and waits until the locked orientation settles.
6. Use the keyboard to move to the desired starting position:

   | Key | Motion |
   |---|---|
   | `W` / `S` | Forward / backward in the locked tool frame |
   | `A` / `D` | Left / right in the locked tool frame |
   | Up / Down arrow | Retract/up / insert/down along the locked tool axis |
   | Space | Stop and hold the current position |
   | `Q` | Stop and quit |

   No key changes the tool orientation.

7. In the recorder, optionally press **Tare** or `T`. A manual tare is retained
   for that trial. If you do not tare, Start automatically uses the most recent
   one second of force data as the baseline. Press **Start Collection** or `S`.
8. Perform the insertion with the teleoperation keyboard.
9. Press **Finish and Save** or `F` in the recorder.
10. Retract the tool and press `Q` in the teleoperation terminal.

Always keep a hand on the physical emergency stop. Test with no phantom first.

## Motion Defaults

The conservative defaults are:

- Position step: `0.05 mm` per key event
- Maximum linear velocity: `0.20 mm/s`
- Maximum angular velocity while holding angle: `0.05 rad/s`
- Maximum insertion from teleoperation start: `3.0 mm`
- Maximum total displacement from teleoperation start: `5.0 mm`
- Pose-staleness stop: `0.5 s`

Example overrides:

```bash
python3 insertion_30deg.py \
  --step-mm 0.025 \
  --max-linear-vel 0.10 \
  --max-insertion-mm 2.0
```

## Saved Session Format

Finishing a recording creates:

```text
data/20260615_143000_angle_p30deg/
├── force_samples.csv
├── metadata.json
├── summary.csv
├── force_and_depth_vs_time.png
└── force_vs_insertion_depth.png
```

`force_samples.csv` is triggered by every force callback. Each row includes:

- Receipt timestamp and elapsed time
- Four raw force channels
- Four EMA-filtered force channels
- Per-channel tare baseline and baseline-subtracted values
- Latest robot pose, Euler orientation, and pose age
- Insertion depth and lateral displacement from recording start
- Target entry angle and latest keyboard action
- Latest commanded linear and angular velocity

`Float64MultiArray` has no ROS header, so force time is the subscriber receipt
time. Robot pose is the most recent pose at that force callback; use
`pose_age_s` to evaluate synchronization quality.

## Session Check

```bash
python3 analyze_force_session.py ../data/<session_directory>
```

This reports force callback rate, pose age, insertion depth range, and
baseline-subtracted force ranges.

## Dependencies

The scripts use ROS1 `rospy`, NumPy, SciPy, PyQt5, PyQtGraph, and Matplotlib.
These libraries are already used elsewhere in the current project and
`EyeRobot` codebase.
