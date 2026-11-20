#!/usr/bin/env python3
"""Offline visual tool correction using fresh CSV, hand-eye, and Brown intrinsics.
Preserves d_const/K. Output is a NEW model; operator must update GUI baseline
and independently recheck before accepting it. HARDWARE_UNVERIFIED.
"""
import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from runtime_session import data_path,digest,vector

def fit_correction(csv_path,model,handeye,intrinsics):
    with np.load(model,allow_pickle=False) as z:old={k:z[k] for k in z.files}
    with np.load(handeye,allow_pickle=False) as z:rh=np.asarray(z['R'],float);th=np.asarray(z['t'],float).reshape(3)
    with np.load(intrinsics,allow_pickle=False) as z:
        k=np.array([[float(z['fx']),0,float(z['ppx'])],[0,float(z['fy']),float(z['ppy'])],[0,0,1.]])
        dist=np.asarray(z['coeffs'],float).reshape(-1)
    with open(csv_path) as source:
        rows=list(csv.DictReader(source))
    if len(rows)<8:raise ValueError('need >=8 diverse poses')
    qs=np.array([[float(r['q'+c]) for c in 'xyzw'] for r in rows]);rots=Rotation.from_quat(qs);rs=rots.as_matrix()
    spread=max((a.inv()*b).magnitude() for a in rots for b in rots)*180/np.pi
    if spread<20:raise ValueError('need >=20 deg orientation diversity')
    pos=np.array([[float(r['p'+c]) for c in 'xyz'] for r in rows]);click=np.array([[float(r['click_u']),float(r['click_v'])] for r in rows])
    q4=np.arctan2(-rs[:,1,2],rs[:,1,1]);u=np.column_stack([np.zeros(len(rows)),np.sin(q4),np.cos(q4)])
    pred=pos+np.einsum('nij,nj->ni',rs,vector(old['d_const'])+float(old['K'])*u)
    def residual(delta,ids=None,frame='tool'):
        corrected=pred+(np.einsum('nij,j->ni',rs,delta) if frame=='tool' else delta)
        cam=(corrected-th)@rh
        if np.any(cam[:,2]<=0):raise ValueError('projection behind camera')
        uv=cv2.projectPoints(cam,np.zeros(3),np.zeros(3),k,dist)[0].reshape(-1,2)
        e=uv-click
        return (e if ids is None else e[ids]).ravel()
    fit=least_squares(residual,np.zeros(3),xtol=1e-10,ftol=1e-10,gtol=1e-10)
    base=least_squares(lambda d:residual(d,frame='base'),np.zeros(3))
    if not fit.success or np.linalg.matrix_rank(fit.jac)<3:raise ValueError('ill-conditioned fit')
    rms=lambda e:float(np.sqrt(np.mean(np.sum(e.reshape(-1,2)**2,axis=1))))
    # Cross-validation refits on each training subset, never on held-out clicks.
    held=np.zeros((len(rows),2))
    for i in range(len(rows)):
        ids=np.arange(len(rows))!=i
        f=least_squares(lambda d:residual(d,ids),np.zeros(3))
        held[i]=residual(f.x).reshape(-1,2)[i]
    rng=np.random.RandomState(0);boots=[]
    for _ in range(100):
        idx=rng.randint(0,len(rows),len(rows));boots.append(least_squares(lambda d:residual(d,idx),fit.x).x)
    report={'delta_mm':fit.x.tolist(),'sigma_mm':np.std(boots,axis=0).tolist(),'n':len(rows),'spread_deg':float(spread),'fit_rms_px':rms(fit.fun),'base_rms_px':rms(base.fun),'loo_rms_px':rms(held),'source_sha256':digest(csv_path)}
    return old,report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',required=True);p.add_argument('--model',default=data_path('tip_offset_rcmoffset.npz'));p.add_argument('--handeye',default=data_path('handeye_T.npz'));p.add_argument('--intrinsics',default=data_path('cam_intrinsics.npz'));p.add_argument('--out',required=True);p.add_argument('--max-correction-mm',required=True,type=float);p.add_argument('--max-rms-px',required=True,type=float);p.add_argument('--max-sigma-mm',required=True,type=float);a=p.parse_args()
    if Path(a.out).resolve()==Path(a.model).resolve():raise SystemExit('Output must differ from input; review before promotion')
    if any(not np.isfinite(v) or v<=0 for v in (a.max_correction_mm,a.max_rms_px,a.max_sigma_mm)):raise SystemExit('positive finite limits required')
    old,r=fit_correction(a.csv,a.model,a.handeye,a.intrinsics);print(json.dumps(r,indent=2))
    delta=vector(r['delta_mm'])
    if np.linalg.norm(delta)>a.max_correction_mm or r['loo_rms_px']>a.max_rms_px or max(r['sigma_mm'])>a.max_sigma_mm:raise SystemExit('Measured correction fails configured physical/uncertainty limits')
    if r['fit_rms_px']>=r['base_rms_px']:raise SystemExit('Tool-frame cause not supported; investigate hand-eye/origin first')
    old.update(tip_inee_at_calib=vector(old['tip_inee_at_calib'])+delta,visual_correction_mm=delta,visual_correction_sigma_mm=r['sigma_mm'],visual_correction_n=r['n'],visual_correction_spread_deg=r['spread_deg'],visual_correction_rms_px=r['fit_rms_px'],visual_correction_loo_rms_px=r['loo_rms_px'],visual_correction_source_sha256=r['source_sha256'])
    np.savez(a.out,**old);Path(a.out+'.report.json').write_text(json.dumps(r,indent=2))
    print('New GUI TipInEE (mm):',old['tip_inee_at_calib']);print('Independently verify, then promote and reaccept downstream registrations.')
if __name__=='__main__':main()
