import csv,json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));import runtime_session

class CLITests(unittest.TestCase):
 def test_relocated_paths_fresh_cli_schema_accept_and_reject(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'copied package with spaces';shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('__pycache__','data'))
   env=os.environ.copy()
   for k in list(runtime_session.PATH_ENV)+['MIC_AUTONOMY_DIR','ISMR_MODE']:env.pop(k,None)
   env['ISMR_SESSION']='test_only_synthetic'
   def run(*args,ok=True):
    result=subprocess.run([sys.executable,*args],cwd=root,env=env,text=True,capture_output=True)
    if ok:self.assertEqual(result.returncode,0,result.stdout+result.stderr)
    else:self.assertNotEqual(result.returncode,0)
    return result
   run('scripts/session.py','init','test_only_synthetic','--tip','1','2','-20')
   run('scripts/preflight.py','--phase','calibration');run('scripts/preflight.py','--phase','insertion',ok=False)
   session=root/'data/sessions/test_only_synthetic';calib=session/'calibration';raw=session/'raw_samples';evidence=raw/'synthetic_evidence.txt';evidence.write_text('SYNTHETIC ONLY; not hardware evidence')
   np.savez(calib/'cam_intrinsics.npz',fx=500,fy=500,ppx=320,ppy=240,coeffs=np.zeros(5),width=640,height=480,n=20,loo_rms_px=.2)
   run('scripts/session.py','accept','cam_intrinsics.npz','--sources',str(evidence),'--max-rms','.5','--note','synthetic only')
   rng=np.random.RandomState(0);uv=rng.uniform(0,100,(15,2));a=np.array([[.1,.01,1],[-.02,.1,2],[.001,.002,3]])
   touches=raw/'mic_calib_touches.jsonl';clicks=raw/'clicks.json';cl={}
   with open(touches,'w') as f:
    for i,p in enumerate(uv):
     label='calib_pt_'+str(i+1);f.write(json.dumps({'label':label,'cur_tip_base_mm':(a@np.r_[p,1]).tolist()})+'\n');cl[label]={'u':p[0],'v':p[1]}
   clicks.write_text(json.dumps(cl))
   run('microscope_autonomy/mic_extrinsic_calib.py','--fit','--loo','--touches',str(touches),'--clicks',str(clicks),'--out',str(calib/'mic_extrinsic_calib.npz'))
   with np.load(calib/'mic_extrinsic_calib.npz') as z:np.testing.assert_allclose(z['A'],a,atol=1e-8)
   samples=raw/'ring_offset_samples.jsonl'
   with open(samples,'w') as f:
    for i in range(8):f.write(json.dumps({'ring_base_mm':[i,0,0],'mark0_base_mm':[i+2,3,1],'mark1_base_mm':[i+8,6,1],'mark0_theta_deg':-150,'mark0_radius_px':400,'occluded':False})+'\n')
   run('microscope_autonomy/mic_ring_offset_verify.py','--fit','--loo','--samples',str(samples),'--out',str(calib/'ring_to_entry_offset.npz'))
   with np.load(calib/'ring_to_entry_offset.npz') as z:self.assertIn('training_yaw_deg',z.files);self.assertEqual(float(z['mark0_radius_px']),400)
   # Copied/tampered model after acceptance cannot pass controller startup.
   np.savez(calib/'cam_intrinsics.npz',fx=600,fy=500,ppx=320,ppy=240,coeffs=np.zeros(5),width=640,height=480,n=20,loo_rms_px=.2)
   run('scripts/preflight.py','--phase','calibration',ok=False)
 def test_offline_cli_help(self):
  for file in ['scripts/session.py','scripts/preflight.py','scripts/capture_needle_axis.py','needle_axis_calibrate.py','tip_offset_apply_visual_correction.py','analysis/campaign_report.py','microscope_autonomy/mic_extrinsic_calib.py','microscope_autonomy/mic_ring_offset_verify.py','microscope_autonomy/mic_independent_verify.py']:
   result=subprocess.run([sys.executable,str(ROOT/file),'--help'],capture_output=True,text=True)
   self.assertEqual(result.returncode,0,file+'\n'+result.stderr)
  result=subprocess.run(['bash',str(ROOT/'run_main.sh'),'--help'],capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
if __name__=='__main__':unittest.main()
