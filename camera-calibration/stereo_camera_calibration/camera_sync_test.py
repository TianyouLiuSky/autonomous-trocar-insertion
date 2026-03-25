#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from collections import deque

last_left  = deque(maxlen=1)
last_right = deque(maxlen=1)
pair_count = [0]

def try_print():
    if last_left and last_right:
        tl, tr = last_left[0], last_right[0]
        dt_pair = abs(tl - tr) * 1000          # sync quality — should be 0
        pair_count[0] += 1
        print(f"[{pair_count[0]:04d}]  L={tl:.4f}  R={tr:.4f}  dt={dt_pair:.2f}ms")

def cb_left(msg):
    last_left.append(msg.header.stamp.to_sec())
    if last_right and last_left[0] != last_right[0]:
        return  # right hasn't updated yet, skip
    try_print()

def cb_right(msg):
    last_right.append(msg.header.stamp.to_sec())
    if last_left and last_left[0] != last_right[0]:
        return  # left hasn't updated yet, skip
    try_print()
    
rospy.init_node("check_sync")
rospy.Subscriber("/camera_array/cam_left/image_raw",  Image, cb_left)
rospy.Subscriber("/camera_array/cam_right/image_raw", Image, cb_right)
rospy.spin()