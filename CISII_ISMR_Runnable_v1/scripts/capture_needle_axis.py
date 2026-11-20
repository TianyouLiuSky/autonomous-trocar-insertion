#!/usr/bin/env python3
"""Read-only ROS + D405 shaft capture before any automatic ORIENT is enabled.
Move to a new free-space pose using lab GUI, stop, click shaft ROI, press c.
No publishers and no robot commands. HARDWARE_UNVERIFIED.
"""
import argparse,csv,json,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args()
import cv2,numpy as np,rospy,pyrealsense2 as rs
from geometry_msgs.msg import Transform
import home_to_rcm_insert_v2_precision as control
from intr_util import load_best_intrinsics
from runtime_session import data_path,registered_problems
probs=registered_problems(('cam_intrinsics.npz','handeye_T.npz'))
if probs:raise SystemExit('; '.join(probs))
rospy.init_node('capture_needle_axis',anonymous=True,disable_signals=True)
state={'q':None,'t':0};roi={'p':None}
def cb(m):state.update(q=np.array([m.rotation.x,m.rotation.y,m.rotation.z,m.rotation.w]),t=time.time())
rospy.Subscriber('/SHER21/eye_robot/FrameEE',Transform,cb,queue_size=1)
with np.load(data_path('handeye_T.npz')) as z:rh=z['R'];th=z['t']
pipe=rs.pipeline();cfg=rs.config();cfg.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30);cfg.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
profile=pipe.start(cfg);intr=load_best_intrinsics(profile);scale=profile.get_device().first_depth_sensor().get_depth_scale()*1000.;align=rs.align(rs.stream.color)
cv2.namedWindow('axis capture');cv2.setMouseCallback('axis capture',lambda event,x,y,flags,param:roi.update(p=(x,y)) if event==cv2.EVENT_LBUTTONDOWN else None)
try:
 while not rospy.is_shutdown():
  frames=align.process(pipe.wait_for_frames());color=frames.get_color_frame();depth=frames.get_depth_frame()
  if not color or not depth:continue
  image=np.asanyarray(color.get_data());view=image.copy()
  if roi['p']:cv2.drawMarker(view,roi['p'],(0,255,0))
  cv2.imshow('axis capture',view);key=cv2.waitKey(1)&255
  if key in (27,ord('q')):break
  if key!=ord('c') or roi['p'] is None:continue
  if state['q'] is None or time.time()-state['t']>0.2:print('stale FrameEE');continue
  q=state['q'].copy();time.sleep(0.3)
  if time.time()-state['t']>0.2 or abs(q@state['q'])<np.cos(np.deg2rad(0.25)):print('not steady: reject');continue
  result=control.measure_shaft_direction_3d(image,np.asanyarray(depth.get_data()).astype(float)*scale,rh,th,intr,roi['p'])
  print(result)
  if not result.get('ok'):continue
  row={'timestamp':time.time(),'vis3d_ok':True,'action':'independent_capture','vis3d_unc_deg':result['angular_uncertainty_deg']}
  row.update({'vis3d_d'+c:float(v) for c,v in zip('xyz',result['direction'])});row.update({'ee_q'+c:float(v) for c,v in zip('xyzw',q)})
  new=not Path(a.out).exists()
  with open(a.out,'a',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(row));
   if new:w.writeheader()
   w.writerow(row)
  cv2.imwrite(a.out+'.'+str(int(time.time()*1000))+'.png',image)
finally:pipe.stop();cv2.destroyAllWindows()
