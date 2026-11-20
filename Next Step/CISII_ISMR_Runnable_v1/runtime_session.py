"""Relocatable paths and fail-closed current-session calibration provenance.
No ROS imports, subscriptions or robot motion in this module.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parent

def session_root():
    name = os.environ.get('ISMR_SESSION', 'UNINITIALIZED')
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('ISMR_SESSION must be a simple session name')
    return ROOT / 'data' / 'sessions' / name

def data_path(*parts):
    name = str(parts[0]) if parts else ''
    if name.endswith('.npz') and not name.startswith('ring_') or name in ('ring_to_entry_offset.npz', 'ring_to_entry_offset_circle.npz'):
        area = 'calibration'
    elif name in ('precision_log.csv','correspondence_log.csv','orient_trace_last.csv','handeye_debug.jsonl') or name.endswith('_log.csv'):
        area = 'logs'
    elif name in ('trial_plunge_stills','campaign','recordings'):
        area = 'trials'
    else:
        area = 'raw_samples'
    return str(session_root() / area / Path(*parts))

# All original environment-based readers/writers share these defaults.
PATH_ENV = {
 'HANDEYE_NPZ':'handeye_T.npz','CAM_INTRINSICS_NPZ':'cam_intrinsics.npz',
 'TIP_OFFSET_NPZ':'tip_offset.npz','TIP_OFFSET_PIVOT_NPZ':'tip_offset_pivot.npz',
 'TIP_OFFSET_PIVOT_AUGMENTED_NPZ':'tip_offset_pivot_augmented.npz',
 'TIP_OFFSET_RCMOFFSET_NPZ':'tip_offset_rcmoffset.npz','AUTO_TARGET_NPZ':'auto_target.npz',
 'NORMAL_NPZ':'phantom_normal.npz','ORIGIN_CORRECTION_NPZ':'origin_correction.npz',
 'MIC_EXTRINSIC_CALIB_NPZ':'mic_extrinsic_calib.npz',
 'RING_TO_ENTRY_OFFSET_NPZ':'ring_to_entry_offset.npz',
 'RING_TO_ENTRY_OFFSET_CIRCLE_NPZ':'ring_to_entry_offset_circle.npz',
 'MIC_CALIB_LOG':'mic_calib_touches.jsonl','MIC_CALIB_TOUCH_STILLS_DIR':'mic_calib_touch_stills',
 'MIC_LIVE_STILLS_DIR':'mic_live_stills','MIC_ENTRY_POINT_JSON':'mic_entry_point.json',
 'RING_OFFSET_SAMPLES_LOG':'ring_offset_samples.jsonl','HANDEYE_DATA_DIR':'handeye_data',
 'PRECISION_LOG':'precision_log.csv','CORRESPONDENCE_LOG':'correspondence_log.csv',
 'ORIENT_TRACE_CSV':'orient_trace_last.csv','OVERLAY_RECORD_DIR':'recordings',
 'TRIAL_STILLS_DIR':'trial_plunge_stills','NORMAL_SAMPLES_LOG':'phantom_normal_samples.jsonl',
 'SESSION_DIR':'captures_session','HANDEYE_PAIRS_CSV':'handeye_data/pairs.csv'}
for _env,_name in PATH_ENV.items():
    os.environ.setdefault(_env,data_path(_name))
os.environ.setdefault('MIC_AUTONOMY_DIR', str(ROOT/'microscope_autonomy'))

def manifest():
    p = session_root() / 'session.json'
    if not p.is_file():
        raise ValueError('No initialized session; use scripts/session.py init NAME --tip x y z')
    d = json.loads(p.read_text())
    if d['session_id'] != os.environ.get('ISMR_SESSION'):
        raise ValueError('session identity mismatch')
    return d

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def vector(v, size=3):
    a = np.asarray(v, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError('expected finite vector of length %d' % size)
    return a

def validate_schema(path):
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        for k in z.files:
            a = z[k]
            if a.dtype.kind in 'fci' and not np.isfinite(a).all():
                # Unknown optional historical metrics may contain NaN; required values checked below.
                if k in ('R','t','d','d_const','K','tip_inee_at_calib','A','axis_ee','normal_base','point_base','offset_mark0','fx','fy','ppx','ppy','coeffs'):
                    raise ValueError('%s has non-finite %s' % (p.name,k))
        def mat(k, shape):
            a = np.asarray(z[k],float)
            if a.shape != shape or not np.isfinite(a).all():raise ValueError('bad '+k)
            return a
        n = p.name
        if n == 'cam_intrinsics.npz':
            for k in ('fx','fy','width','height'):
                if float(z[k]) <= 0:raise ValueError('invalid '+k)
            for k in ('ppx','ppy','coeffs'):np.asarray(z[k],float)
            if int(z['n'])<5 or not np.isfinite(float(z['loo_rms_px'])) or float(z['loo_rms_px'])<0:raise ValueError('camera lacks LOO evidence')
        elif n == 'handeye_T.npz':
            r = mat('R',(3,3));t=np.asarray(z['t'],float).reshape(-1);vector(t)
            if not np.allclose(r.T @ r,np.eye(3),atol=1e-5) or abs(np.linalg.det(r)-1)>1e-5:raise ValueError('invalid rotation')
        elif n.startswith('tip_offset'):
            vector(z['tip_inee_at_calib'])
            if n == 'tip_offset_rcmoffset.npz':vector(z['d_const']);float(z['K'])
            elif n == 'tip_offset_pivot.npz':vector(z['d'])
        elif n == 'needle_axis.npz':
            a=vector(z['axis_ee'])
            if abs(np.linalg.norm(a)-1)>1e-5 or str(z['frame'])!='EE' or str(z['units'])!='unitless':raise ValueError('axis frame/unit/norm')
            if not np.isfinite([float(z['max_spread_deg']),float(z['max_allowed_deg'])]).all() or float(z['max_allowed_deg'])<=0 or int(z['n'])<3 or float(z['max_spread_deg'])>float(z['max_allowed_deg']):raise ValueError('axis evidence failed')
        elif n == 'mic_extrinsic_calib.npz':
            a=mat('A',(3,3))
            if np.linalg.matrix_rank(a[:2,:2])<2 or int(z['n_calib_points'])<5 or not np.isfinite(float(z['loo_rms_mm'])) or float(z['loo_rms_mm'])<0:raise ValueError('registration lacks held-out evidence')
        elif n == 'ring_to_entry_offset.npz':
            vector(z['offset_mark0'])
            if int(z['n_samples_mark0'])<5 or not np.isfinite(float(z['loo_rms_mm_mark0'])) or float(z['loo_rms_mm_mark0'])<0:raise ValueError('ring model lacks LOO')
            if not np.isfinite(float(z['training_yaw_deg'])):raise ValueError('missing fresh paired-fiducial training yaw')
            if not np.isfinite(float(z['mark0_theta_deg'])) or not np.isfinite(float(z['mark0_radius_px'])) or float(z['mark0_radius_px'])<=0:raise ValueError('missing fresh nominal mark geometry')
        elif n == 'phantom_normal.npz':
            a=vector(z['normal_base']);vector(z['point_base'])
            if abs(np.linalg.norm(a)-1)>0.01:raise ValueError('normal must be unit')
        elif n == 'origin_correction.npz':
            if np.linalg.norm(vector(z['correction']))>40:raise ValueError('origin correction >40 mm')

REQUIRED = ('cam_intrinsics.npz','handeye_T.npz','tip_offset_rcmoffset.npz','needle_axis.npz','mic_extrinsic_calib.npz','ring_to_entry_offset.npz')

def registered_problems(names=REQUIRED):
    try:d=manifest()
    except (ValueError, OSError, KeyError) as e:return [str(e)]
    errors=[]
    for name in names:
        p=Path(data_path(name));entry=d['artifacts'].get(name)
        try:
            if not p.is_file() or not entry:raise ValueError('missing/unregistered '+name)
            if digest(p)!=entry['sha256']:raise ValueError('modified since acceptance: '+name)
            for source,h in entry['sources'].items():
                sp=session_root()/source
                if not sp.is_file() or digest(sp)!=h:raise ValueError('source changed; refit/reaccept '+name)
            for dep,h in entry.get('dependencies',{}).items():
                if h=='ABSENT':
                    if Path(data_path(dep)).exists():raise ValueError('new upstream correction; refit '+name)
                elif digest(data_path(dep))!=h:raise ValueError('upstream changed; refit '+name)
            validate_schema(p)
            q=entry.get('quality',{})
            if q and q['value']>q['max_rms']:raise ValueError('accepted quality exceeds limit')
        except (ValueError,KeyError,OSError,TypeError) as e:errors.append(str(e))
    return errors

def startup_check():
    d=manifest()
    if os.environ.get('TIP_MODEL','rcmoffset')!='rcmoffset':raise SystemExit('v1 preflight supports TIP_MODEL=rcmoffset only; alternatives require separate validation')
    for env,name in PATH_ENV.items():
        if Path(os.environ[env]).resolve() != Path(data_path(name)).resolve():
            raise SystemExit('Path override not permitted in v1 session: '+env)
    if Path(os.environ['MIC_AUTONOMY_DIR']).resolve() != (ROOT/'microscope_autonomy').resolve():
        raise SystemExit('MIC_AUTONOMY_DIR must use packaged source')
    for p in (session_root()/'calibration').glob('*.npz'):
        if p.name in REQUIRED or p.name == 'origin_correction.npz':
            problems=registered_problems((p.name,))
            if problems:raise SystemExit('; '.join(problems))
    # No legacy NPZ is silently imported, even in calibration mode.
    for p in (session_root()/'calibration').glob('*.npz'):
        if p.stat().st_mtime < d['created_at']:raise SystemExit('Reject pre-session calibration: '+p.name)
    if os.environ.get('ISMR_MODE','calibration')=='insertion':
        if os.environ.get('MIC_RCM_MODE','circle')!='circle' or os.environ.get('MIC_CIRCLE_D405_MODEL','fixed')!='fixed' or os.environ.get('BIPLANAR_MODE','1')!='1':raise SystemExit('v1 insertion requires circle/fixed and BIPLANAR_MODE=1')
        problems=registered_problems()
        if problems:raise SystemExit('; '.join(problems))

def motion_problems():
    if os.environ.get('ISMR_MODE','calibration')!='insertion':return ['automatic stages disabled in calibration mode']
    names=list(REQUIRED)
    if Path(data_path('origin_correction.npz')).exists():names.append('origin_correction.npz')
    return registered_problems(names)

def expected_tip(explicit=None):
    if explicit is not None:return vector(explicit)
    d=manifest();p=Path(data_path('tip_offset_rcmoffset.npz'))
    if p.is_file():
        with np.load(p,allow_pickle=False) as z:return vector(z['tip_inee_at_calib'])
    return vector(d['bootstrap_tip_inee_mm'])

def load_axis(required=False):
    p=Path(data_path('needle_axis.npz'))
    if not p.is_file():
        if required:raise ValueError('No current independent needle axis')
        return None
    validate_schema(p)
    with np.load(p,allow_pickle=False) as z:return vector(z['axis_ee'])
