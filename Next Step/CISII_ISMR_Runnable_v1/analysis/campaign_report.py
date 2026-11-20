#!/usr/bin/env python3
"""Offline campaign completeness, closed-loop errors, CIs and LOTO entry offset.
Hole clicks provide a 2D projected offset only. A 3D common-offset estimate
requires independently measured entry_base/target_base XYZ columns.
"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from runtime_session import data_path

def stats(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if not len(x):return {'n':0}
    rng=np.random.RandomState(0);boot=np.array([np.median(rng.choice(x,len(x),replace=True)) for _ in range(2000)])
    return {'n':len(x),'mean':float(x.mean()),'std':float(x.std(ddof=1)) if len(x)>1 else None,'median':float(np.median(x)),'rms':float(np.sqrt(np.mean(x*x))),'median_bootstrap95':np.percentile(boot,[2.5,97.5]).tolist()}

def loto(vectors,trial_ids):
    v=np.asarray(vectors,float);ids=np.asarray(trial_ids);unique=np.unique(ids)
    if len(unique)<3:raise ValueError('LOTO needs >=3 independent trials')
    # Each training trial has equal weight, even if repeated measurements exist.
    means=np.array([v[ids==i].mean(axis=0) for i in unique]);errors=[]
    for i in unique:
        train=means[unique!=i].mean(axis=0);errors.extend(np.linalg.norm(v[ids==i]-train,axis=1))
    return {'dimensions':int(v.shape[1]),'trial_count':len(unique),'common_offset':means.mean(axis=0).tolist(),'raw_error_mm':stats(np.linalg.norm(v,axis=1)),'loto_corrected_mm':stats(errors),'note':'CI resamples measured errors; does not model calibration systematic uncertainty'}

def report(campaign,holes=None,calib=None,entries=None):
    root=Path(campaign);trials=sorted(root.glob('trial*/*.json'));trial_data=[]
    for p in trials:
        d=json.loads(p.read_text())
        if 'trial' in d and 'timestamp' in d:trial_data.append(d)
    ids=[str(d['trial']) for d in trial_data]
    stages={};failures=[]
    for p in root.glob('trial*/precision_rows.csv'):
        for r in csv.DictReader(open(p)):
            name=r.get('action','')
            if r.get('result')=='reached':
                for key in ('err_mm','err_deg','elapsed_s'):
                    try:stages.setdefault(name,{}).setdefault(key,[]).append(float(r[key]))
                    except (ValueError,KeyError):pass
            elif r.get('result') in ('stall','stalled','aborted','timeout','failed'):failures.append(r)
    out={'status':'OFFLINE_ANALYSIS_HARDWARE_UNVERIFIED','trial_count':len(set(ids)),'duplicate_trial_ids':sorted({i for i in ids if ids.count(i)>1}),'finalized_count':sum('post_retract_timestamp' in d for d in trial_data),'expected_trials':20,'stages':{k:{m:stats(v) for m,v in d.items()} for k,d in stages.items()},'failed_rows':failures}
    if holes:
        with np.load(calib,allow_pickle=False) as z:linear=z['A'][:2,:2]
        rs=[r for r in csv.DictReader(open(holes)) if r.get('status')=='ok']
        vec=[linear@np.array([float(r['hole_u'])-float(r['target_u']),float(r['hole_v'])-float(r['target_v'])]) for r in rs]
        out['entry_projected_xy']=loto(vec,[r['trial'] for r in rs]);out['excluded_hole_rows']=sum(r.get('status')!='ok' for r in csv.DictReader(open(holes)))
    if entries:
        rs=list(csv.DictReader(open(entries)));vec=[[float(r['entry_base_'+c])-float(r['target_base_'+c]) for c in 'xyz'] for r in rs];out['entry_base_3d']=loto(vec,[r['trial'] for r in rs])
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',default=data_path('campaign'));p.add_argument('--holes');p.add_argument('--calib',default=data_path('mic_extrinsic_calib.npz'));p.add_argument('--entries');p.add_argument('--out',required=True);a=p.parse_args();r=report(a.campaign,a.holes,a.calib,a.entries);Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
