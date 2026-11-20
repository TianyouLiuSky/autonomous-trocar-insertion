#!/usr/bin/env python3
"""jog_only.py -- headless-of-camera keyboard jog, extracted verbatim from
home_to_rcm_insert_v2_precision.py's own jog logic (same LIN/ANG key map,
same SPEED/ASPEED defaults, same pulse-hold behavior), so it can run in a
SEPARATE terminal AT THE SAME TIME as handeye_calibrate.py (or any other
script that needs the RealSense exclusively) -- this script touches ONLY
ROS topics, never opens the camera, so there is no exclusive-device
conflict. Needed because:
  - handeye_calibrate.py's own interactive loop holds the camera AND has
    no jog keys of its own (by design -- it's a passive recorder).
  - home_to_rcm_insert_v2_precision.py has jog keys but ALSO holds the
    camera continuously for its own display, so it can't run at the same
    time as handeye_calibrate.py either.
  - This script has NO camera dependency at all, so it's free to run
    alongside handeye_calibrate.py: jog here, watch handeye_calibrate.py's
    OWN window for ArUco-visible confirmation, press SPACE over there to
    record, without any toggling/relaunching of either script.

Usage:
    python3 jog_only.py

Keys (identical to home_to_rcm_insert_v2_precision.py):
  w/s : jog +X / -X (linear)
  a/d : jog +Y / -Y (linear)
  r/f : jog +Z / -Z (linear)
  k/m : jog +roll / -roll (angular, about X)
  j/l : jog +pitch / -pitch (angular, about Y)
  u/o : jog +yaw / -yaw (angular, about Z)
  ]/[ : speed up / down
  space : stop
  q / Esc : quit (zeroes velocity first)

Does NOT touch handeye_data/pairs.csv or any calibration file -- recording
happens over in handeye_calibrate.py's window (SPACE there), this window
only moves the robot.
"""
from __future__ import print_function
import os
import time

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Transform, Vector3

VEL_TOPIC = os.environ.get("VEL_TOPIC", "/SHER21/eyerobot2/desiredTipVelocities")
ANG_TOPIC = os.environ.get("ANG_TOPIC", "/SHER21/eyerobot2/desiredTipVelocitiesAngular")
FRAMEEE_TOPIC = os.environ.get("FRAMEEE_TOPIC", "/SHER21/eye_robot/FrameEE")

SPEED = float(os.environ.get("JOG_SPEED", "3.0"))
SPEED_MAX = float(os.environ.get("JOG_MAX", "8.0"))
ASPEED = float(os.environ.get("JOG_ASPEED", "0.02"))
ASPEED_MAX = float(os.environ.get("JOG_AMAX", "0.3"))
PULSE_S = float(os.environ.get("PULSE_S", "0.20"))

LIN = {ord('w'): (1, 0, 0), ord('s'): (-1, 0, 0), ord('a'): (0, 1, 0),
       ord('d'): (0, -1, 0), ord('r'): (0, 0, 1), ord('f'): (0, 0, -1)}
ANG = {ord('k'): (1, 0, 0), ord('m'): (-1, 0, 0),
       ord('j'): (0, 1, 0), ord('l'): (0, -1, 0),
       ord('u'): (0, 0, 1), ord('o'): (0, 0, -1)}

_ee = {"p": None, "n_msgs": 0}


def _cb_ee(msg):
    _ee["p"] = np.array([msg.translation.x, msg.translation.y, msg.translation.z], float)
    _ee["n_msgs"] += 1


def main():
    global SPEED, ASPEED
    rospy.init_node("jog_only", anonymous=True, disable_signals=True)
    pub = rospy.Publisher(VEL_TOPIC, Vector3, queue_size=1)
    pub_a = rospy.Publisher(ANG_TOPIC, Vector3, queue_size=1)
    rospy.Subscriber(FRAMEEE_TOPIC, Transform, _cb_ee, queue_size=1)
    rospy.sleep(0.3)

    def zero():
        pub.publish(Vector3(0, 0, 0)); pub_a.publish(Vector3(0, 0, 0))

    win = "JOG ONLY (no camera -- run alongside handeye_calibrate.py)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 640, 260)

    jog = {"lin": None, "ang": None, "until": 0.0}
    msg = ""
    print("jog_only.py running. FrameEE topic: %s  (%s)" %
          (FRAMEEE_TOPIC, "receiving" if _ee["p"] is not None else "NOT receiving yet -- check bringup"))
    print("Keys: w/a/s/d/r/f=lin  k/m/j/l/u/o=ang  ]/[=speed  space=stop  q=quit")

    try:
        while not rospy.is_shutdown():
            canvas = np.zeros((260, 640, 3), np.uint8)
            p = _ee["p"]
            p_txt = ("[%.2f %.2f %.2f]mm" % tuple(p)) if p is not None else "NO FrameEE YET"
            frameee_col = (0, 255, 80) if p is not None else (0, 0, 255)
            cv2.putText(canvas, "FrameEE: %s" % p_txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, frameee_col, 1, cv2.LINE_AA)
            cv2.putText(canvas, "lin=%.1fmm/s  ang=%.3frad/s" % (SPEED, ASPEED), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "w/a/s/d/r/f = jog XYZ (linear)", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, "k/m/j/l/u/o = jog roll/pitch/yaw (angular)", (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, "]/[ = speed up/down   space = stop   q = quit", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, "This window does NOT touch the camera or record anything --", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "jog here, press SPACE in the handeye_calibrate.py window to record.", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 255), 1, cv2.LINE_AA)
            if msg:
                cv2.putText(canvas, msg, (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            k = cv2.waitKey(20) & 0xFF
            now = time.time()

            if k == 255:
                if now < jog["until"] and (jog["lin"] is not None or jog["ang"] is not None):
                    lv = jog["lin"] or (0, 0, 0)
                    av = jog["ang"] or (0, 0, 0)
                    pub.publish(Vector3(lv[0] * SPEED, lv[1] * SPEED, lv[2] * SPEED))
                    pub_a.publish(Vector3(av[0] * ASPEED, av[1] * ASPEED, av[2] * ASPEED))
                else:
                    zero(); jog["lin"] = None; jog["ang"] = None
                continue

            if k in (ord('q'), 27):
                zero()
                break
            elif k == ord(' '):
                zero(); jog["lin"] = None; jog["ang"] = None; msg = "STOP"
            elif k == ord(']'):
                SPEED = min(SPEED * 1.4, SPEED_MAX); ASPEED = min(ASPEED * 1.4, ASPEED_MAX)
                msg = "lin=%.1f ang=%.3f" % (SPEED, ASPEED)
            elif k == ord('['):
                SPEED = max(SPEED / 1.4, 0.2); ASPEED = max(ASPEED / 1.4, 0.005)
                msg = "lin=%.1f ang=%.3f" % (SPEED, ASPEED)
            elif k in LIN:
                jog["lin"] = LIN[k]; jog["ang"] = None; jog["until"] = now + PULSE_S
            elif k in ANG:
                jog["ang"] = ANG[k]; jog["lin"] = None; jog["until"] = now + PULSE_S
    finally:
        zero()
        cv2.destroyAllWindows()
        print("jog_only.py exiting, velocities zeroed.")


if __name__ == "__main__":
    main()
