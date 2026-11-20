import ast,csv,importlib,json,os,sys,tempfile,time,types,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'microscope_autonomy'),str(ROOT/'analysis')]
import runtime_session as runtime
from needle_axis_calibrate import solve
from campaign_report import loto,report
import rcmoffset_fit
import mic_extrinsic_calib as mic
import mic_ring_offset_verify as ring

# Hardware import substitutes only. Any attempted hardware API access fails.
def forbidden(*a,**kw):raise AssertionError('Offline tests must not access hardware')
def install_import_stubs():
    for name in ['rospy','pyrealsense2']:
        m=types.ModuleType(name);m.Publisher=forbidden;m.Subscriber=forbidden;m.init_node=forbidden;m.pipeline=forbidden;sys.modules[name]=m
    for pkg,names in [('geometry_msgs',['Vector3','Transform']),('std_msgs',['Float64MultiArray']),('sensor_msgs',['Image'])]:
        a=types.ModuleType(pkg);b=types.ModuleType(pkg+'.msg')
        for name in names:setattr(b,name,type(name,(),{}))
        a.msg=b;sys.modules[pkg]=a;sys.modules[pkg+'.msg']=b
install_import_stubs()
os.environ.setdefault('MIC_RCM_ANGLE_DEG','25')
import home_to_rcm_insert_v2_precision as controller
import tip_inee_guard as guard

class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.oldroot=runtime.ROOT;self.oldenv=os.environ.copy()
        runtime.ROOT=Path(self.temp.name);os.environ['ISMR_SESSION']='synthetic';os.environ['ISMR_MODE']='calibration';os.environ['TIP_MODEL']='rcmoffset'
        self.session=runtime.session_root()
        for area in ['calibration','raw_samples','logs','trials']:(self.session/area).mkdir(parents=True)
        self.d={'session_id':'synthetic','created_at':time.time()-1,'bootstrap_tip_inee_mm':[1,2,-20],'artifacts':{}}
        self.write_manifest()
        for env,name in runtime.PATH_ENV.items():os.environ[env]=runtime.data_path(name)
        os.environ['MIC_AUTONOMY_DIR']=str(runtime.ROOT/'microscope_autonomy')
    def tearDown(self):
        runtime.ROOT=self.oldroot;os.environ.clear();os.environ.update(self.oldenv);self.temp.cleanup()
    def write_manifest(self):(self.session/'session.json').write_text(json.dumps(self.d))
    def register(self,name,**values):
        p=Path(runtime.data_path(name));np.savez(p,**values)
        self.d['artifacts'][name]={'sha256':runtime.digest(p),'sources':{},'dependencies':{}};self.write_manifest();return p
    def axis_csv(self,name='axis.csv',bad=False):
        path=self.session/'raw_samples'/name;rows=[];axis=np.array([0.05,0.1,-1.]);axis/=np.linalg.norm(axis)
        for i in range(8):
            r=Rotation.from_euler('xy',[i*7,i*3],degrees=True);d=r.apply(axis if not bad or i!=7 else np.array([1.,0,0]))
            row={'vis3d_ok':'True'};row.update({'vis3d_d'+c:v for c,v in zip('xyz',d)});row.update({'ee_q'+c:v for c,v in zip('xyzw',r.as_quat())});rows.append(row)
        with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return path,axis
    def test_empty_calibration_bootstraps_without_models(self):
        runtime.startup_check();self.assertEqual(runtime.expected_tip().tolist(),[1,2,-20]);self.assertIsNone(runtime.load_axis())
        self.assertTrue(runtime.motion_problems())
    def test_insertion_refuses_empty(self):
        os.environ['ISMR_MODE']='insertion'
        with self.assertRaises(SystemExit):runtime.startup_check()
    def test_legacy_model_rejected(self):
        p=Path(runtime.data_path('handeye_T.npz'));np.savez(p,R=np.eye(3),t=np.zeros(3));os.utime(p,(1,1))
        with self.assertRaises(SystemExit):runtime.startup_check()
    def test_external_path_override_rejected(self):
        os.environ['HANDEYE_NPZ']='/tmp/history.npz'
        with self.assertRaises(SystemExit):runtime.startup_check()
    def test_hash_and_dependency_change_rejected(self):
        p=self.register('handeye_T.npz',R=np.eye(3),t=np.zeros(3));self.assertFalse(runtime.registered_problems(('handeye_T.npz',)))
        np.savez(p,R=np.eye(3),t=np.ones(3));self.assertTrue(runtime.registered_problems(('handeye_T.npz',)))
    def test_invalid_rotation_or_nan_rejected(self):
        p=self.register('handeye_T.npz',R=np.eye(3)*2,t=np.zeros(3))
        with self.assertRaises(ValueError):runtime.validate_schema(p)
        p=self.register('handeye_T.npz',R=np.eye(3),t=[0,0,np.nan])
        with self.assertRaises(ValueError):runtime.validate_schema(p)
    def test_current_tip_guard_and_nan(self):
        old=guard.read_live_tip_inee;guard.read_live_tip_inee=lambda:np.array([1,2,-20.])
        try:
            np.testing.assert_allclose(guard.assert_tip_inee(),[1,2,-20])
            with self.assertRaises(ValueError):guard.assert_tip_inee(expected=[0,0,np.nan])
            guard.read_live_tip_inee=lambda:np.array([0,0,100.])
            with self.assertRaises(SystemExit):guard.assert_tip_inee()
        finally:guard.read_live_tip_inee=old
    def test_axis_fit_sign_and_loader(self):
        p,axis=self.axis_csv();a,err,spread=solve(p,[0,0,-1],1.,20.)
        np.testing.assert_allclose(a,axis,atol=1e-7);self.assertGreater(spread,20)
        self.register('needle_axis.npz',axis_ee=a,frame='EE',units='unitless',n=8,max_spread_deg=err.max(),max_allowed_deg=1.)
        np.testing.assert_allclose(runtime.load_axis(True),axis)
    def test_bad_axis_no_silent_outlier_removal(self):
        p,_=self.axis_csv(bad=True)
        with self.assertRaises(ValueError):solve(p,[0,0,-1],2.,20.)
    def test_ring_first_collection_fixed_and_tangent(self):
        old={k:getattr(controller,k,None) for k in ['NPZ','NORMAL_NPZ','MIC_EXTRINSIC_CALIB_NPZ','RING_TO_ENTRY_OFFSET_NPZ','RING_TO_ENTRY_OFFSET_CIRCLE_NPZ','ORIGIN_CORRECTION','MIC_RCM_MODE','MIC_CIRCLE_D405_MODEL']}
        try:
            controller.NPZ=str(self.session/'he');Path(controller.NPZ).touch()
            controller.NORMAL_NPZ=str(self.session/'normal');Path(controller.NORMAL_NPZ).touch()
            controller.MIC_EXTRINSIC_CALIB_NPZ=str(self.session/'mic');Path(controller.MIC_EXTRINSIC_CALIB_NPZ).touch()
            controller.RING_TO_ENTRY_OFFSET_NPZ=str(self.session/'missing_fixed');controller.RING_TO_ENTRY_OFFSET_CIRCLE_NPZ=str(self.session/'missing_tangent');controller.ORIGIN_CORRECTION=np.zeros(3);controller.MIC_RCM_MODE='circle'
            for mode in ['fixed','tangent']:
                controller.MIC_CIRCLE_D405_MODEL=mode
                self.assertFalse(controller.calib_freshness_problems('collect'));self.assertTrue(controller.calib_freshness_problems('use'))
            Path(controller.MIC_EXTRINSIC_CALIB_NPZ).unlink();self.assertTrue(controller.calib_freshness_problems('collect'))
        finally:
            for k,v in old.items():setattr(controller,k,v)
    def test_affine_registration_with_loo(self):
        rng=np.random.RandomState(4);uv=rng.uniform(0,1000,(15,2));a=np.array([[.1,.02,-10],[-.01,.08,3],[.001,.002,8]])
        xyz=np.c_[uv,np.ones(15)]@a.T;fit,e=mic.fit_affine(uv,xyz)
        np.testing.assert_allclose(fit,a,atol=1e-10);self.assertLess(max(mic.leave_one_out_errors(uv,xyz)),1e-9)
    def test_ring_fit_loo(self):
        p=np.arange(30).reshape(10,3);offset=np.array([2,3,1]);r=ring.fit_one_mark(p,p+offset,0,True)
        np.testing.assert_allclose(r['offset'],offset);self.assertLess(r['loo_rms_mm'],1e-9)
    def test_rcmoffset_fit_from_fresh_synthetic_poses(self):
        rng=np.random.RandomState(3);q3=rng.uniform(-.7,.7,30);q4=rng.uniform(-.8,.8,30)
        rs=np.array([(Rotation.from_euler('y',a)*Rotation.from_euler('x',b)).as_matrix() for a,b in zip(q3,q4)])
        d=np.array([1.,2.,3.]);k=-2.;u=np.c_[np.zeros(30),np.sin(q4),np.cos(q4)];pivot=np.array([20.,10.,30.]);ps=pivot-np.einsum('nij,nj->ni',rs,d+k*u)
        fit,kk,pp=rcmoffset_fit.fit([(rs,q4,ps)]);np.testing.assert_allclose(fit,d,atol=1e-9);self.assertAlmostEqual(kk,k);np.testing.assert_allclose(pp[0],pivot)
    def test_controller_geometry_unchanged(self):
        # Compare actual functions against audited baseline commit file, ignoring path changes.
        old=ROOT/'tests'/'controller_geometry_baseline.json';baseline=json.loads(old.read_text());tree=ast.parse((ROOT/'home_to_rcm_insert_v2_precision.py').read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.FunctionDef) and n.name in baseline:self.assertEqual(ast.dump(n,include_attributes=False),baseline[n.name],n.name)
    def test_stop_and_stage_pivots_preserved(self):
        text=(ROOT/'home_to_rcm_insert_v2_precision.py').read_text()
        self.assertIn('pivot_point=pivot_now',text);self.assertIn('pivot_now = cur_tip()',text);self.assertIn('pivot_point=S.internal_rcm',text)
        tree=ast.parse(text);zero=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='zero');self.assertIn('publish',ast.unparse(zero))
    def test_loto_and_trial_completeness(self):
        v=np.tile([2.,3.],(20,1));r=loto(v,list(range(20)));self.assertEqual(r['trial_count'],20);self.assertEqual(r['loto_corrected_mm']['median'],0)
        camp=self.session/'trials'/'campaign';camp.mkdir()
        for i in range(1,21):
            folder=camp/('trial%02d_x'%i);folder.mkdir();(folder/'trial.json').write_text(json.dumps({'trial':i,'timestamp':i,'post_retract_timestamp':i+1}))
        self.assertEqual(report(camp)['finalized_count'],20)
    def test_intrinsics_requires_measured_loo(self):
        p=self.register('cam_intrinsics.npz',fx=500,fy=500,ppx=320,ppy=240,coeffs=np.zeros(5),width=640,height=480,n=20,loo_rms_px=.2);runtime.validate_schema(p)
        self.assertEqual(Path(runtime.data_path('cam_intrinsics.npz')).parent,self.session/'calibration')
    def test_visual_correction_synthetic_new_magnitude(self):
        import cv2
        from tip_offset_apply_visual_correction import fit_correction
        rng=np.random.RandomState(9);rots=Rotation.from_euler('xy',rng.uniform(-35,35,(10,2)),degrees=True);rs=rots.as_matrix()
        model=self.register('tip_offset_rcmoffset.npz',d_const=np.array([.1,.2,.3]),K=-1.,tip_inee_at_calib=np.array([0,0,-20.]))
        he=self.register('handeye_T.npz',R=np.eye(3),t=np.zeros(3));intr=self.register('cam_intrinsics.npz',fx=500.,fy=510.,ppx=320.,ppy=240.,coeffs=np.zeros(5),width=640,height=480,n=20,loo_rms_px=.1)
        pos=rng.normal(0,3,(10,3))+[0,0,150];q4=np.arctan2(-rs[:,1,2],rs[:,1,1]);u=np.c_[np.zeros(10),np.sin(q4),np.cos(q4)];delta=np.array([.3,-.4,.5])
        points=pos+np.einsum('nij,nj->ni',rs,np.array([.1,.2,.3])-u+delta)
        uv=cv2.projectPoints(points,np.zeros(3),np.zeros(3),np.array([[500.,0,320],[0,510.,240],[0,0,1]]),np.zeros(5))[0].reshape(-1,2)
        path=self.session/'raw_samples'/'visual.csv';rows=[]
        for point,q,click in zip(pos,rots.as_quat(),uv):
            row={};row.update({'p'+c:v for c,v in zip('xyz',point)});row.update({'q'+c:v for c,v in zip('xyzw',q)});row.update(click_u=click[0],click_v=click[1]);rows.append(row)
        with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        _,r=fit_correction(path,model,he,intr);np.testing.assert_allclose(r['delta_mm'],delta,atol=1e-4);self.assertLess(r['loo_rms_px'],.001)
    def test_archiver_on_fresh_trial(self):
        import trial_archiver as ar
        old={k:getattr(ar,k) for k in ['STILLS','CAMPAIGN','PREC','CORR','ORIENT_TRACE','REC_DIR','HASHED','controller_env']}
        try:
            stills=self.session/'trials'/'trial_plunge_stills';stills.mkdir();camp=self.session/'trials'/'campaign';camp.mkdir()
            ar.STILLS=str(stills);ar.CAMPAIGN=str(camp);ar.PREC=str(self.session/'logs'/'p.csv');ar.CORR=str(self.session/'logs'/'c.csv');ar.REC_DIR=str(self.session/'trials'/'recordings');ar.HASHED=[];ar.controller_env=lambda:None
            Path(ar.PREC).touch();self.assertEqual(ar.log_rows(ar.PREC,0,10),(None,[]))
            jp=stills/'trial01_synth.json';jp.write_text(json.dumps({'trial':1,'timestamp':time.time(),'post_retract_timestamp':time.time()+1}))
            row,_=ar.archive(str(jp),None);self.assertEqual(row['finalized'],1);self.assertTrue((camp/jp.stem/'config_snapshot.json').exists())
        finally:
            for k,v in old.items():setattr(ar,k,v)
if __name__=='__main__':unittest.main(verbosity=2)
