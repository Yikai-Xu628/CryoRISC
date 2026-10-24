import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from cryorisc import backend
from cryorisc.io import write_stack,read_mrc_memmap
from cryorisc.losses import supervised_direction_contrastive_loss
from cryorisc.metrics import intra_class_angular_consistency as iac
from cryorisc.pipeline import initial_candidates
from cryorisc.profiles import get_profile

torch.set_num_threads(2)


def test_mrc_roundtrip(tmp_path):
    values=np.random.default_rng(1).normal(size=(4,16,16)).astype('f4')
    path=tmp_path/'input.mrcs';write_stack(path,values,2.68)
    np.testing.assert_array_equal(read_mrc_memmap(path),values)


def test_iac_known_answer_and_singletons():

    v=np.array([[1,0,0],[1,0,0],[0,1,0.]])
    r=iac(v,np.array([0,0,1]))
    assert r['singleton_particles']==1
    assert r['denominator']==3
    for x in r['iac_percent'].values():assert x==pytest.approx(200/3)


def test_iac_antipodal_equivalence():
    r=iac(np.array([[1.,0,0],[-1.,0,0]]),np.array([0,0]),[0,5])
    assert list(r['iac_percent'].values())==[100.,100.]


def test_iac_particle_not_pair_weighting():

    v=np.array([[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,0,1.]])
    assert iac(v,np.array([0,0,1,1,1]),[5])['iac_percent']['5']==pytest.approx(40)


def test_iac_block_invariance():
    rng=np.random.default_rng(1);v=rng.normal(size=(37,3));y=np.arange(37)%4
    a=iac(v,y,block_size=2);b=iac(v,y,block_size=512)
    np.testing.assert_array_equal(a['per_particle'],b['per_particle'])


@pytest.mark.parametrize('v,y',[(np.zeros((2,3)),np.array([0,0])),
    (np.ones((2,3)),np.array([-1,0])),(np.ones((2,3)),np.array([0.5,0.5]))])
def test_iac_rejects_invalid(v,y):
    with pytest.raises(ValueError):iac(v,y)


def test_contrastive_excludes_other_structures():
    z=torch.nn.functional.normalize(torch.randn(8,6),dim=1)
    v=torch.tensor([[1.,0,0]]*2+[[0.,1,0]]*2+[[1.,0,0]]*2+[[0.,1,0]]*2)
    g=torch.arange(8)//2;s=torch.arange(8)//4
    joint=supervised_direction_contrastive_loss(z,v,g,structure_ids=s)
    separate=torch.stack([supervised_direction_contrastive_loss(z[k:k+4],v[k:k+4],g[k:k+4]) for k in (0,4)]).mean()
    torch.testing.assert_close(joint,separate,atol=1e-6,rtol=1e-6)


def test_rescue_keeps_16_unique_and_current():
    scores=np.arange(400,dtype='f4').reshape(2,200)
    candidates=backend.rescue_candidates(scores,np.array([0,199]))
    assert candidates.shape==(2,16)
    assert all(len(set(row))==16 for row in candidates)
    assert 0 in candidates[0] and 199 in candidates[1]


def test_neutral_candidates_have_no_feature_scores():
    labels=np.arange(200)
    c,s=backend.neutral_candidates(labels,32,200,np.random.default_rng(9))
    assert np.all(s==0)
    assert np.all(np.any(c==labels[:,None],axis=1))
    assert all(len(set(row))==32 for row in c)


def test_no_network_initialization_never_opens_metadata(tmp_path):
    images=np.random.default_rng(1).normal(size=(400,16,16)).astype('f4')
    c=dict(source=tmp_path/'DOES_NOT_EXIST.npz',particles=400,size=16,radius=5,edge=1)
    labels,anchors,emb,classes,scores,templates=backend.prepare(c,'B_no_network',11,images)
    assert emb is None and np.all(scores==0)
    np.testing.assert_array_equal(np.bincount(labels),np.full(200,2))
    np.testing.assert_array_equal(labels[anchors],np.arange(200))
    assert templates.shape==(200,16,16)


def test_angle_grid_and_metadata(tmp_path):
    p=tmp_path/'meta.npz';c=np.array([[0,3],[2,1]])
    backend.full_angles(p,np.zeros((4,16,16),'f4'),c,6)
    with np.load(p) as z:
        assert z['candidate_classes'].shape==(2,120)
        np.testing.assert_array_equal(z['pair_rotations_deg'][0,:60],np.arange(0,360,6))


def test_profile_settings_not_collapsed():
    a,b,c=(get_profile(x) for x in ('10028','10219','65045'))
    assert a['historical'] and not b['historical']
    assert a['fine_iterations']==3 and b['fine_iterations']==2
    assert a['union'] and c['union'] and not b['union']
    assert a['background_radius']==b['background_radius']==75
    assert c['background_radius']==pytest.approx(75*128/180)


def test_historical_top32_matches_original_builder():
    z=np.random.default_rng(1).normal(size=(240,96)).astype('f4')
    anchors=np.arange(200)
    c,s=initial_candidates(z,anchors)
    sim=z@z[anchors].T
    expected=np.argpartition(-sim,31,axis=1)[:,:32]
    order=np.argsort(-np.take_along_axis(sim,expected,1),axis=1)
    np.testing.assert_array_equal(c,np.take_along_axis(expected,order,1))
    np.testing.assert_array_equal(s,np.take_along_axis(sim,c,1))


def test_tiny_candidate_budgets(tmp_path):
    rng=np.random.default_rng(3)
    particles=rng.normal(size=(32,16,16)).astype('f4')
    embeddings=rng.normal(size=(32,96)).astype('f4')
    labels=np.arange(32)%8;anchors=np.arange(8)
    source=tmp_path/'source.npz'
    np.savez_compressed(source,embeddings=embeddings,initial_kmeans_labels=labels,selected_anchor_ids=anchors)
    config=dict(source=source,particles=32,size=16,radius=5,edge=1)
    label,anchor,z,classes,sims,templates=backend.prepare(config,'A_full',3,particles,classes=8,candidate_budget=4)
    assert classes.shape==(32,4) and templates.shape==(8,16,16)
    inp=tmp_path/'input.npz';backend.full_angles(inp,templates,classes,30,sims)
    fit=dict(pair_posterior=np.full((32,48),1/48,'f4'),pair_residual_angle=np.zeros((32,48),'f4'))
    fc,fr,fs=backend.fine_candidates(inp,fit,label,z,anchor,False,class_count=8,budget=2)
    assert fc.shape==fr.shape==fs.shape==(32,2)
    c=backend.rescue_candidates(rng.normal(size=(32,8)),labels,budget=4)
    assert c.shape==(32,4) and np.all(np.any(c==labels[:,None],axis=1))


def test_tiny_engine_end_to_end_cpu(tmp_path):
    rng=np.random.default_rng(1)
    particles=rng.normal(size=(6,16,16)).astype('f4')
    stack=tmp_path/'input.mrcs';write_stack(stack,particles,1.)
    inp=tmp_path/'meta.npz'
    backend.metadata(inp,particles[:2],np.tile(np.arange(2),(6,1)),np.zeros((6,2)))
    out=tmp_path/'fit'
    command=[sys.executable,'-m','cryorisc.engine','--input',str(inp),
        '--images-mrcs',str(stack),'--output-dir',str(out),'--iterations','1',
        '--batch-size','2','--max-shift','2','--residual-angle','0',
        '--mask-radius','5','--mask-edge','1','--cpu-images']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    done=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
    assert done.returncode==0,done.stdout+done.stderr
    with np.load(out/'soft_refine_result.npz') as fit:
        assert fit['assignment'].shape==(6,)
        assert np.isfinite(fit['templates']).all()


def test_iac_cli(tmp_path):
    np.save(tmp_path/'v.npy',np.array([[1.,0,0],[1.,0,0]]))
    np.save(tmp_path/'y.npy',np.array([0,0]))
    out=tmp_path/'iac.json'
    done=subprocess.run([sys.executable,'-m','cryorisc','iac','--directions',str(tmp_path/'v.npy'),
        '--labels',str(tmp_path/'y.npy'),'--output',str(out)],capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    assert json.loads(out.read_text())['iac_percent']['8']==100
