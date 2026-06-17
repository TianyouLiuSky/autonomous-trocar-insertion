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

The straight/direct-down robot orientation is the absolute XYZ Euler
orientation `(roll, pitch, yaw) = (0, -13, 0)` degrees.

- `direct_down_insertion.py` moves to and locks `(0, -13, 0)`.
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
| FBG force | Auto-detects `/eye_robot/FBGForcesTip` or `/SHER20/eye_robot/FBGForcesTip` |
| Raw FBG wavelengths | Auto-detects `/eye_robot/WavelengthsRaw` or `/SHER20/eye_robot/WavelengthsRaw` |
| Linear velocity command | `/SHER20/eyerobot2/desiredTipVelocities` |
| Angular velocity command | `/SHER20/eyerobot2/desiredTipVelocitiesAngular` |

The recorder stores all four available `FBGForcesTip` values. Existing
EyeRobot force-control scripts treat channel 0 as the tip/axial force in
newtons. The recorder UI displays all four raw force values, raw wavelength
values, selected topics, message rates, message ages, and recent peak-to-peak
motion so a topic, calibration, or sensor problem is visible.

## Operator Workflow

Use two terminals on the robot computer.

1. Place the phantom and tool in their correct experimental positions.
2. Keep the tool clear of the phantom. The control script will rotate to the
   configured straight orientation `(0, -13, 0)`.
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

   30-degree-from-horizontal entry:

   ```bash
   python3 insertion_30deg.py --robot-name SHER20
   ```

5. For either script, confirm that the tool has clearance to rotate. The
   30-degree script rotates 60 degrees away from straight/direct-down. Both
   scripts wait until the locked orientation settles.
6. A dedicated movement-control window opens. Click that window, then hold the
   keys to move to the desired starting position:

   | Key | Motion |
   |---|---|
   | Hold `W` / `S` | Robot-base X+ / X- |
   | Hold `A` / `D` | Robot-base Y+ / Y- |
   | Hold `C` / `V` | Robot-base Z- (down) / Z+ (up) |
   | Space | Stop and hold the current position |
   | `Q` | Stop and quit |

   Translation is commanded in the robot base frame. It stops as soon as the
   movement key is released and also stops if the control window loses focus.
   No key changes the tool orientation.

   The motion window reports `Command gate`. This code has no force/contact
   stop. It can suppress translation if orientation error exceeds 2 degrees or
   a travel limit is reached. SHER's lower-level force-control/contact behavior
   is outside this script and must remain enabled unless the robot's responsible
   operator changes its controller mode.

7. In the recorder, optionally press **Tare** or `T`. A manual tare is retained
   for that trial. If you do not tare, Start automatically uses the most recent
   one second of force data as the baseline. Press **Start Collection** or `S`.
8. Perform the insertion with the teleoperation keyboard.
9. Press **Finish and Save** or `F` in the recorder.
10. Retract the tool and press `Q` in the movement-control window.

Always keep a hand on the physical emergency stop. Test with no phantom first.

## Contact and Force Diagnosis

- If `Command gate` reports an orientation or travel limit, this teleoperation
  script is suppressing the requested translation.
- If the displayed Z command remains negative while `C` is held but the robot
  does not advance, SHER's lower-level controller is limiting the motion.
- If force is `missing` or `stale`, launch with an explicit topic, for example
  `--force-topic /eye_robot/FBGForcesTip`.
- If messages are current but every raw channel and recent peak-to-peak value
  remains unchanged while load is applied, the issue is upstream of the
  recorder: sensor publication, sensor calibration, or robot operating mode.
- If raw wavelengths change but force stays flat, the FBG sensor is likely
  publishing but the force conversion/calibration path is not producing a
  changing force estimate.

## Motion Defaults

The conservative defaults are:

- Hold-to-move linear velocity: `0.20 mm/s`
- Maximum angular velocity while holding angle: `0.05 rad/s`
- Maximum downward Z travel from teleoperation start: `20.0 mm`
- Maximum upward Z travel from teleoperation start: `20.0 mm`
- Maximum total displacement from teleoperation start: `25.0 mm`
- Pose-staleness stop: `0.5 s`

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
