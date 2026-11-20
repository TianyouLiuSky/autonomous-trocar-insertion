#!/usr/bin/env python3
"""Initialize or accept newly measured artifacts; never publishes motion."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_session import session_root,data_path,manifest,digest,validate_schema,vector

DEPENDENCIES={
 'handeye_T.npz':['cam_intrinsics.npz'],
 'tip_offset_rcmoffset.npz':[],
 'needle_axis.npz':['handeye_T.npz'],
 'mic_extrinsic_calib.npz':['tip_offset_rcmoffset.npz'],
 'ring_to_entry_offset.npz':['handeye_T.npz','mic_extrinsic_calib.npz'],
 'origin_correction.npz':['handeye_T.npz','tip_offset_rcmoffset.npz']}

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
 i=sub.add_parser('init');i.add_argument('name');i.add_argument('--tip',nargs=3,type=float,required=True)
 a=sub.add_parser('accept');a.add_argument('artifact');a.add_argument('--sources',nargs='+',required=True);a.add_argument('--note',required=True);a.add_argument('--max-rms',type=float,help='bench-approved limit, pixels for intrinsics; mm for tip/registration/ring')
 sub.add_parser('paths')
 args=p.parse_args()
 if args.cmd=='init':
  os.environ['ISMR_SESSION']=args.name;root=session_root()
  if root.exists():raise SystemExit('Session already exists; choose a new name')
  vector(args.tip)
  for part in ['calibration','raw_samples','logs','trials','raw_samples/handeye_data','raw_samples/mic_calib_touch_stills','raw_samples/mic_live_stills','trials/trial_plunge_stills','trials/recordings','trials/campaign']:(root/part).mkdir(parents=True,exist_ok=True)
  d={'session_id':args.name,'created_at':time.time(),'bootstrap_tip_inee_mm':args.tip,'frame':'robot base, mm; quaternion xyzw','status':'HARDWARE_UNVERIFIED','artifacts':{}}
  (root/'session.json').write_text(json.dumps(d,indent=2))
  print('export ISMR_SESSION='+args.name)
 elif args.cmd=='paths':
  manifest()
  for n in ['cam_intrinsics.npz','handeye_T.npz','tip_offset_rcmoffset.npz','needle_axis.npz','mic_extrinsic_calib.npz','ring_to_entry_offset.npz','phantom_normal.npz','mic_calib_touches.jsonl','ring_offset_samples.jsonl','precision_log.csv']:print(n,data_path(n))
 else:
  d=manifest();out=Path(data_path(args.artifact));validate_schema(out)
  if out.stat().st_mtime<d['created_at']:raise SystemExit('Artifact predates current session')
  sources={}
  for name in args.sources:
   src=Path(name).resolve();rel=src.relative_to(session_root().resolve())
   if src.stat().st_mtime<d['created_at']:raise SystemExit('Source predates session')
   sources[str(rel)]=digest(src)
  if args.artifact=='needle_axis.npz':
   import numpy as np
   with np.load(out,allow_pickle=False) as z:fit_hash=str(z['source_sha256']);limit=float(z['max_allowed_deg'])
   reports=[]
   for filename in args.sources:
    if filename.endswith('.json'):
     try:reports.append(json.loads(Path(filename).read_text()))
     except (ValueError,OSError):pass
   if not any(r.get('validation') is True and r.get('source_sha256')!=fit_hash and r.get('n',0)>=3 and r.get('max_error_deg',float('inf'))<=limit for r in reports):raise SystemExit('Need independent needle_axis_calibrate --check JSON report among --sources')
  quality={}
  metric={'cam_intrinsics.npz':'loo_rms_px','tip_offset_rcmoffset.npz':'loo_rms_mm','mic_extrinsic_calib.npz':'loo_rms_mm','ring_to_entry_offset.npz':'loo_rms_mm_mark0'}.get(args.artifact)
  if metric:
   import numpy as np
   if args.max_rms is None or not np.isfinite(args.max_rms) or args.max_rms<=0:raise SystemExit('--max-rms must be the positive bench-approved tolerance')
   with np.load(out,allow_pickle=False) as z:value=float(z[metric])
   if not np.isfinite(value) or value<0 or value>args.max_rms:raise SystemExit('LOO RMS outside configured tolerance')
   quality={'metric':metric,'value':value,'max_rms':args.max_rms}
  deps={}
  for dep in DEPENDENCIES.get(args.artifact,[]):
   entry=d['artifacts'].get(dep)
   if not entry or digest(data_path(dep))!=entry['sha256']:raise SystemExit('Accept current upstream first: '+dep)
   deps[dep]=entry['sha256']
  if args.artifact in ('mic_extrinsic_calib.npz','ring_to_entry_offset.npz'):
   origin=Path(data_path('origin_correction.npz'))
   if origin.exists() and 'origin_correction.npz' not in d['artifacts']:raise SystemExit('Accept current origin correction first')
   deps['origin_correction.npz']=digest(origin) if origin.exists() else 'ABSENT'
  d['artifacts'][args.artifact]={'sha256':digest(out),'sources':sources,'dependencies':deps,'quality':quality,'accepted_at':time.time(),'note':args.note}
  (session_root()/'session.json').write_text(json.dumps(d,indent=2));print('Accepted '+args.artifact)
if __name__=='__main__':main()
