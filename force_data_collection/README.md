# Force Data Collection

This directory contains a standalone workflow for collecting trocar penetration
force at fixed direct-down, perpendicular-to-eye, or 30-degree entry angle.

## Directory Layout

```text
force_data_collection/
├── code/
│   ├── direct_down_insertion.py
│   ├── perpendicular_insertion.py
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

The straight/direct-down robot orientation is the absolute XYZ Euler
orientation `(roll, pitch, yaw) = (0, -13, 0)` degrees.

- `direct_down_insertion.py` moves to and locks `(0, -13, 0)`.
- `perpendicular_insertion.py` moves to and locks the perpendicular-to-eye
  orientation `(0, +20, 0)`. This is treated as an absolute robot RPY target,
  and `C/V` insertion follows the locked tool axis.
- `insertion_30deg.py` represents the 30-degree-from-horizontal oblique
  condition. Since straight/direct-down is vertical, the script applies
  60 degrees about tool-local Y from straight. With the default positive tilt
  this is equivalent to approximately `(0, +47, 0)`.

Use `--tilt-axis local-x` or `--tilt-sign -1` if the experimental fixture needs
the tilt in a different plane or direction. Verify the direction with the
phantom clear of the tool before an insertion trial.

## ROS Topics

Defaults for `--robot-name SHER20`:

| Purpose | Topic |
|---|---|
| Robot pose | `/SHER20/eye_robot/FrameEE` |
| Force | Use `/SHER20/eye_robot2/HandleForces` for final SHER2.0 collection |
| Raw FBG wavelengths | Auto-detects `/eye_robot/WavelengthsRaw` or `/SHER20/eye_robot/WavelengthsRaw` |
| Linear velocity command | `/SHER20/eyerobot2/desiredTipVelocities` |
| Angular velocity command | `/SHER20/eyerobot2/desiredTipVelocitiesAngular` |

The recorder stores the active force topic's first four values. Existing
EyeRobot force-control scripts often use `FBGForcesTip`, but the SHER2.0
logger also watches `ScleraForces` and `HandleForces`. The recorder UI displays
raw force values, raw wavelength values, selected topics, message rates, message
ages, and recent peak-to-peak motion so a topic, calibration, or sensor problem
is visible. Debugging confirmed that `/SHER20/eye_robot2/HandleForces` is the
working SHER2.0 force topic for this collection workflow; keep auto-detection
for troubleshooting, but pin this topic for final experimental runs.

## Operator Workflow

Use two terminals on the robot computer.

1. Place the phantom and tool in their correct experimental positions.
2. Keep the tool clear of the phantom. The control script will rotate to the
   selected locked orientation for the chosen insertion mode.
3. Start the recorder UI:

   ```bash
   cd force_data_collection/code
   python3 force_recorder_ui.py \
     --robot-name SHER20 \
     --force-topic /SHER20/eye_robot2/HandleForces
   ```

4. Start one fixed-angle teleoperation script:

   Direct down:

   ```bash
   python3 direct_down_insertion.py --robot-name SHER20
   ```

   Perpendicular-to-eye:

   ```bash
   python3 perpendicular_insertion.py --robot-name SHER20
   ```

   30-degree-from-horizontal entry:

   ```bash
   python3 insertion_30deg.py --robot-name SHER20
   ```

5. For any insertion script, confirm that the tool has clearance to rotate.
   The perpendicular script rotates to the absolute `(0, +20, 0)` RPY target.
   The 30-degree script rotates 60 degrees away from straight/direct-down. All
   scripts wait until the locked orientation settles. After orientation setup,
   the script automatically moves the tip to its pre-teleop start target within
   `0.5 mm` tolerance, unless launched with `--skip-centering`.

   - Direct down uses the workspace midpoint `(-16.0, -109.0, 8.5)` mm.
   - Perpendicular-to-eye uses the workspace midpoint `(-16.0, -109.0, 8.5)`
     mm.
   - 30-degree oblique uses `(-5.5, -109.0, 17.0)` mm. This is intentionally
     closer to the observed upper/right fixed-orientation limit so the tool has
     more usable shaft-direction insertion travel before the lower/left
     translation limit around `x = -19.7, z = 3.9`.
6. A dedicated movement-control window opens. Click that window, then hold the
   keys to move to the desired starting position:

   | Key | Motion |
   |---|---|
   | Hold `W` / `S` | Robot-base X+ / X- |
   | Hold `A` / `D` | Robot-base Y+ / Y- |
   | Hold `C` / `V` | Insert / retract |
   | Space | Stop and hold the current position |
   | `Q` | Stop and quit |

   `W/S` and `A/D` are lateral robot-base motions. In the direct-down script,
   `C/V` are pure robot-base Z down/up, even though the locked straight
   orientation is `(0, -13, 0)`. In the perpendicular and 30-degree scripts,
   `C/V` follow the locked insertion axis computed from the tool orientation.
   Translation stops as soon as the movement key is released and also stops if
   the control window loses focus. No key changes the tool orientation.

   The motion window reports `Command gate`. This code has no force/contact
   stop. It can suppress translation if orientation error exceeds 2 degrees, a
   relative travel limit is reached, or the command would push farther outside
   the configured workspace. For tool-axis insertion, it also suppresses the
   Z-down component if commanded horizontal progress stalls while `C` is held;
   the gate reports `horizontal stall guard` and clears when `C` is released.
   SHER's lower-level force-control/contact behavior is outside this script and
   must remain enabled unless the robot's responsible operator changes its
   controller mode.

7. In the recorder, optionally press **Tare** or `T`. A manual tare is retained
   for that trial. If you do not tare, Start automatically uses the most recent
   one second of force data as the baseline. Press **Start Collection** or `S`.
   The recorder UI shows the active condition at the top: `DIRECT DOWN
   (0 deg)`, `PERPENDICULAR (20 deg)`, or `30 DEG OBLIQUE`.
   The force chart keeps a fixed recent-time width from `--plot-seconds` while
   automatically rescaling its vertical force range to the visible raw and EMA
   traces. Tune the vertical padding with `--plot-y-padding-fraction` and the
   minimum displayed span with `--plot-y-min-span`.
8. Perform the insertion with the teleoperation keyboard.
9. Press **Finish and Save** or `F` in the recorder.
10. Retract the tool and press `Q` in the movement-control window.

Always keep a hand on the physical emergency stop. Test with no phantom first.

## Contact and Force Diagnosis

- If `Command gate` reports an orientation or travel limit, this teleoperation
  script is suppressing the requested translation.
- If `Command gate` reports a workspace limit, the current key command would
  move the tip past the configured workspace boundary.
- If `Command gate` reports `horizontal stall guard`, the tool-axis insertion
  command needs horizontal motion, but the measured horizontal pose progress is
  too small. The script blocks Z-down motion to prevent tool bending. Release
  `C`, retract or reposition, and do not keep pushing into the obstruction.
- If the displayed command vector changes while `C` is held but the robot does
  not advance along the tool axis, SHER's lower-level controller is limiting
  the motion.
- If force is `missing` or `stale`, launch with an explicit topic, for example
  `--force-topic /SHER20/eye_robot2/HandleForces` for the validated SHER2.0
  setup.
- If messages are current but every raw channel and recent peak-to-peak value
  remains unchanged while load is applied, the issue is upstream of the
  recorder: sensor publication, sensor calibration, or robot operating mode.
- If raw wavelengths change but force stays flat, the FBG sensor is likely
  publishing but the force conversion/calibration path is not producing a
  changing force estimate.

## Motion Defaults

The conservative defaults are:

- Hold-to-move linear velocity: `0.50 mm/s`
- Maximum angular velocity while holding angle: `0.05 rad/s`
- Workspace bounds: `x = -42..10 mm`, `y = -133..-85 mm`,
  `z = -13..30 mm`
- Direct-down pre-teleop target: `(-16.0, -109.0, 8.5) mm`
- Perpendicular pre-teleop target: `(-16.0, -109.0, 8.5) mm`
- 30-degree pre-teleop target: `(-5.5, -109.0, 17.0) mm`
- Workspace and centering tolerance: `0.5 mm`
- Maximum insertion from teleoperation start: `20.0 mm`
- Maximum retraction from teleoperation start: `20.0 mm`
- Direct-down C/V axis: robot-base Z
- Perpendicular and oblique C/V axis: locked tool insertion axis
- Maximum total displacement from teleoperation start: `25.0 mm`
- Pose-staleness stop: `0.5 s`
- Tool-axis horizontal-stall guard: enabled for tool axes with horizontal
  component at least `0.35`; measured over `0.6 s`

Example overrides:

```bash
python3 insertion_30deg.py \
  --max-linear-vel 0.10 \
  --max-insertion-mm 15.0
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
- Four latest raw wavelength channels
- Four EMA-filtered force channels
- Per-channel tare baseline and baseline-subtracted values
- Latest robot pose, Euler orientation, and pose age
- Insertion depth and lateral displacement from recording start
- Insertion axis used for depth/lateral projection
- Target entry angle and latest keyboard action
- Latest commanded linear and angular velocity

`Float64MultiArray` has no ROS header, so force time is the subscriber receipt
time. Robot pose is the most recent pose at that force callback; use
`pose_age_s` to evaluate synchronization quality.

`metadata.json` stores `insertion_condition`, `target_entry_angle_deg`, and
`record_insertion_axis_base_frame`. The session directory name also includes
the angle label, for example `angle_p0deg` or `angle_p30deg`.

## Session Check

```bash
python3 analyze_force_session.py ../data/<session_directory>
```

This reports the recorded condition, insertion axis, force callback rate, pose
age, insertion depth range, and baseline-subtracted force ranges.

## Dependencies

The scripts use ROS1 `rospy`, NumPy, SciPy, PyQt5, PyQtGraph, and Matplotlib.
These libraries are already used elsewhere in the current project and
`EyeRobot` codebase.
