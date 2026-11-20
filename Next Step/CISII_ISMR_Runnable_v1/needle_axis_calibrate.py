#!/usr/bin/env python3
"""Fit a signed EE-frame unit shaft axis from independent depth measurements.
No ROS or motion. Use a separately collected CSV for --check.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from runtime_session import data_path,digest,vector

def load_samples(path, reference):
    axes=[];rotations=[]
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get('vis3d_ok','').lower() not in ('true','1'):continue
            d=vector([float(r['vis3d_d'+c]) for c in 'xyz']);d/=np.linalg.norm(d)
            q=vector([float(r['ee_q'+c]) for c in 'xyzw'],4)
            if abs(np.linalg.norm(q)-1)>0.01:raise ValueError('invalid quaternion')
            rot=Rotation.from_quat(q);a=rot.as_matrix().T @ d
            if abs(a @ reference)<0.1:raise ValueError('axis sign ambiguous against reference; verify physical direction')
            if a @ reference<0:a=-a
            axes.append(a);rotations.append(rot)
    if len(axes)<3:raise ValueError('need >=3 accepted independent measurements')
    spread=max((a.inv()*b).magnitude() for a in rotations for b in rotations)*180/np.pi
    return np.array(axes),spread

def solve(path,reference,max_spread_deg,min_orientation_deg):
    ref=vector(reference)
    if np.linalg.norm(ref)<1e-8:raise ValueError('nonzero physical sign reference required')
    ref/=np.linalg.norm(ref)
    axes,orientation=load_samples(path,ref)
    mean=axes.mean(axis=0);mean/=np.linalg.norm(mean)
    errors=np.degrees(np.arccos(np.clip(axes@mean,-1,1)))
    if orientation<min_orientation_deg:raise ValueError('insufficient orientation diversity')
    if errors.max()>max_spread_deg:raise ValueError('inconsistent shaft measurements; no automatic outlier deletion')
    return mean,errors,orientation

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv');p.add_argument('--sign-reference',nargs=3,type=float,required=True);p.add_argument('--max-spread-deg',type=float,required=True);p.add_argument('--min-orientation-deg',type=float,default=20);p.add_argument('--out',default=data_path('needle_axis.npz'));p.add_argument('--check',help='independent validation of existing axis; does not refit');p.add_argument('--report');a=p.parse_args()
    if not np.isfinite(a.max_spread_deg) or a.max_spread_deg<=0:raise SystemExit('positive finite tolerance required')
    mean,errors,orientation=solve(a.csv,a.sign_reference,a.max_spread_deg,a.min_orientation_deg)
    if a.check:
        with np.load(a.check,allow_pickle=False) as z:
            if str(z['source_sha256'])==digest(a.csv):raise SystemExit('validation CSV must differ from fit CSV')
            axes,_=load_samples(a.csv,vector(a.sign_reference)/np.linalg.norm(a.sign_reference));errors=np.degrees(np.arccos(np.clip(axes@z['axis_ee'],-1,1)))
        if errors.max()>a.max_spread_deg:raise SystemExit('independent axis validation failed')
    else:
        np.savez(a.out,axis_ee=mean,frame='EE',units='unitless',sign_reference=vector(a.sign_reference),n=len(errors),max_spread_deg=errors.max(),max_allowed_deg=a.max_spread_deg,orientation_spread_deg=orientation,source_sha256=digest(a.csv))
    report={'n':len(errors),'axis_ee':mean.tolist(),'max_error_deg':float(errors.max()),'median_error_deg':float(np.median(errors)),'orientation_spread_deg':orientation,'source_sha256':digest(a.csv),'validation':bool(a.check),'status':'HARDWARE_UNVERIFIED'}
    print(json.dumps(report,indent=2))
    if a.report:Path(a.report).write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
