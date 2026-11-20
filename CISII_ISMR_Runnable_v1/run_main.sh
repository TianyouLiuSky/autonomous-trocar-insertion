#!/bin/bash
# 2026-10-07: canonical launcher for home_to_rcm_insert_v2_precision.py.
#
# WHY THIS EXISTS: two silent regressions in two days, both from the same
# root cause (restarting with `python3 home_to_rcm_insert_v2_precision.py`
# directly and relying on remembering a long list of env vars each time):
#   - 10-06: forgot MIC_RCM_MODE=circle -> silently fell back to the old
#     "mark" (physical ink marker) RCM detection mode
#   - 10-07: forgot MANUAL_JOG_RCM_PIVOT=0 -> silently fell back to "1"
#     (pivot-compensation ON), reintroducing rotation-causes-translation
#     during manual jog
# Both env vars have safe-looking defaults in the script itself, so neither
# mistake produced an error -- just quietly wrong behavior. This script is
# the fix: ALWAYS use it (or copy/extend it) instead of invoking the python
# file directly, so the full correct set of env vars is one command, not a
# fact to remember.
#
# Usage:
#   ./run_main.sh                  # free-space / calibration mode (default)
#   ./run_main.sh --trocar         # real trocar/RCM inserted -> re-enables
#                                   # MANUAL_JOG_RCM_PIVOT=1 (rotate-about-
#                                   # live-RCM protection) for that session
#   FOO=bar ./run_main.sh          # any env var can still be overridden by
#                                   # setting it before calling this script
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

MANUAL_JOG_RCM_PIVOT_DEFAULT=0
if [ "$1" = "--trocar" ]; then
    MANUAL_JOG_RCM_PIVOT_DEFAULT=1
    echo "run_main.sh: --trocar mode -> MANUAL_JOG_RCM_PIVOT=1 (rotate-about-live-RCM protection ON)"
    shift
fi

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/eyebrp/.Xauthority}"
source /opt/ros/melodic/setup.bash
source /home/eyebrp/catkin_ws/devel/setup.bash 2>/dev/null || true

export PULSE_S="${PULSE_S:-2.0}"
# 2026-10-11: target = point on mark0's own limbus-concentric circle, 25deg from the
# screw (the trocar cannot pierce the screw head itself; trial 2 slid off it).
# D405 side: validated mark0 fixed vector (XY LOO 0.91mm) + constant circle step.
export MIC_RCM_MODE="${MIC_RCM_MODE:-circle}"
export MIC_TARGET_MARK="${MIC_TARGET_MARK:-0}"
export MIC_TARGET_RADIUS_MARK="${MIC_TARGET_RADIUS_MARK:-0}"
# Pose-robust: angle is relative to mark0's measured angle (+25deg), z from the D405
# ring plane at the target (follows phantom tilt/height). No pose limits enforced.
export MIC_TARGET_ANGLE_REL_MARK="${MIC_TARGET_ANGLE_REL_MARK:-0}"
export MIC_RCM_ANGLE_DEG="${MIC_RCM_ANGLE_DEG:-25}"
# 2026-10-12: 20 distinct sites on mark0's limbus-concentric circle (lower arc, inside the
# microscope view, >=15deg from both screws), 7.5deg = ~1.6mm apart, interleaved so
# neighbouring sites are not consecutive trials. Trial N uses entry N.
# 2026-10-12: faster puncture -> less pre-rupture tissue deflection (globe pushed 2-4 mm at 3 mm/s).
export PIERCE_SPEED="${PIERCE_SPEED:-6.0}"
# 2026-10-12: oblique (tunnel) angle per trial, 25-45deg x4 in randomized blocks of 5;
# correlation with the site schedule below and with trial order = 0.000 (seed 15312).
export INSERT_ANGLE_SCHEDULE="${INSERT_ANGLE_SCHEDULE:-30,35,40,45,25,30,45,25,35,40,35,40,45,25,30,30,40,35,25,45}"
# 2026-10-12: mark0 screw too bright for the 0.72 dark threshold under today's lighting.
export MARK_DARK_FRACTION="${MARK_DARK_FRACTION:-0.80}"
# Sites between about -55 and -115 deg put the target past the bottom edge of the microscope frame
# (y > 1020 px), so they can never lock; those entries were replaced by upper-arc sites before trial 2.
# Phase 1 (phantom fixed, restarted at trial 1): sites -15, 135, -22.5, 40 were already punctured in
# phase 0, so entries 1-4 are -60, 127.5, -165, -67.5; with the phantom fixed slightly high in the
# view, 62.5 and 55 fall off the top edge and were replaced by -67.5 and -75.
export MIC_RCM_ANGLE_SCHEDULE="${MIC_RCM_ANGLE_SCHEDULE:--60,127.5,-165,-67.5,-30,142.5,-37.5,-112.5,-45,-120,-52.5,-127.5,47.5,-135,150,-142.5,-75,-150,157.5,-157.5}"
export MIC_Z_SOURCE="${MIC_Z_SOURCE:-d405plane}"
export MIC_CIRCLE_D405_MODEL="${MIC_CIRCLE_D405_MODEL:-fixed}"
export MIC_ROS_LIVE="${MIC_ROS_LIVE:-0}"
export MANUAL_JOG_RCM_PIVOT="${MANUAL_JOG_RCM_PIVOT:-$MANUAL_JOG_RCM_PIVOT_DEFAULT}"
# 2026-10-07: script's own internal default is TIP_MODEL=rcmoffset, but the
# two tip-offset npz files (both refit 2026-09-21 on the SAME post-trocar-
# swap data, n=2273) show augmented is ~3x more accurate on orientation-
# dependent bias: LOO RMS rcmoffset=0.473mm vs augmented=0.168mm (and
# tip_offset_rcmoffset.npz's own provenance_note admits "K only explains 9%
# of the orientation-dependent bias here ... weaker physical fit than
# before, treat with more caution"). This default override makes the
# better-validated model the one actually used, without touching the
# script's own internal default (so TIP_MODEL=rcmoffset still works if ever
# needed for comparison).
# 2026-10-11: back to rcmoffset. augmented's LOO only covers the pivot-data
# orientations; at the ORIENT pose (45deg from the calibration-touch pose) it
# disagrees with rcmoffset by 3.1mm XY / 2.3mm Z, and the physical tip drifted
# exactly that way (toward the limbus) during ORIENT. origin_correction.npz was
# shifted by aug(q_touch)-rcm(q_touch) so cur_tip at the touch pose is unchanged
# (backup: 03_results_npz/origin_correction.npz.bak_augmented_182010).
export TIP_MODEL="${TIP_MODEL:-rcmoffset}"

echo "run_main.sh launching with:"
echo "  PULSE_S=$PULSE_S  MIC_RCM_MODE=$MIC_RCM_MODE  MIC_TARGET_MARK=$MIC_TARGET_MARK  MIC_RCM_ANGLE_DEG=$MIC_RCM_ANGLE_DEG"
echo "  MIC_ROS_LIVE=$MIC_ROS_LIVE  MANUAL_JOG_RCM_PIVOT=$MANUAL_JOG_RCM_PIVOT  TIP_MODEL=$TIP_MODEL"

exec python3 home_to_rcm_insert_v2_precision.py "$@"
