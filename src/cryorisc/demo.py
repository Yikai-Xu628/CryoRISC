import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import backend
from .features import spherical_kmeans
from .image_search import pixel_scores
from .io import read_mrc_memmap,write_stack
from .pipeline import digest,extract_features,load_pair,run_engine,save
from .profiles import get_profile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--example-dir',type=Path,default=Path('examples/tiny65045'))
    p.add_argument('--encoder',type=Path,default=Path('checkpoints/best_encoder.pt'))
    p.add_argument('--adapter',type=Path,default=Path('checkpoints/best_adapter.pt'))
    p.add_argument('--output-dir',type=Path,default=Path('runs/tiny65045'))
    a=p.parse_args()
    started=time.perf_counter();torch.set_num_threads(4)
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    stack=(a.example_dir/'particles.mrcs').resolve()
    x=read_mrc_memmap(stack);n=len(x);k=8
    if x.shape!=(32,128,128):raise ValueError('This demo expects 32 native 128-pixel particles')
    manifest=dict(particles=n,classes=k,coarse_candidates=4,fine_candidates=2,
        rescue_candidates=4,stack_sha256=digest(stack),encoder_sha256=digest(a.encoder),
        adapter_sha256=digest(a.adapter),code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    if (out/'demo_manifest.json').exists() and json.loads((out/'demo_manifest.json').read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Demo inputs/code changed; choose a new output directory')
    save(out/'demo_manifest.json',manifest)
    profile=get_profile('65045');device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,adapter,epoch=load_pair(a.encoder,a.adapter,device)
    z=extract_features(x,model,adapter,device,profile,16)
    labels,centres=spherical_kmeans(z,k,device,20260809)
    sim=z@centres.T
    anchors=np.array([np.flatnonzero(labels==c)[np.argmax(sim[labels==c,c])] for c in range(k)])
    source=out/'features_and_groups.npz'
    np.savez_compressed(source,embeddings=z,initial_kmeans_labels=labels,
                        selected_anchor_ids=anchors,cluster_centres=centres)
    config=dict(profile,stack=stack,source=source,particles=n,shift=8)
    labels,anchors,z,cc,cs,templates=backend.prepare(config,'A_full',20261012,x,classes=k,candidate_budget=4)
    label_path=out/'initial_labels.npy';np.save(label_path,labels)
    initial=out/'bootstrap_input.npz';backend.full_angles(initial,templates,labels[:,None],6)
    basic={'iterations':1,'batch-size':2,'max-shift':8,'residual-angle':0,
        'hard-core-template-updates':True,'embedding-weight':0,'class-prior-power':0,
        'template-momentum':0,'max-cutoff':.12}
    boot=run_engine(config,out,initial,'bootstrap',basic,label_path)
    coarse_input=out/'coarse_input.npz';backend.full_angles(coarse_input,boot['templates'],cc,30,cs)
    common={'iterations':1,'batch-size':2,'max-shift':8,'residual-angle':0,'angle-step':1,
        'marginalize-angles':True,'min-match-frequency':.015,'temperature':.025,
        'embedding-weight':.5,'class-prior-power':.25,'template-momentum':.15,'max-cutoff':.20}
    coarse=run_engine(config,out,coarse_input,'coarse',common)
    fc,fr,fs=backend.fine_candidates(coarse_input,coarse,labels,z,anchors,False,class_count=k,budget=2)
    fine_input=out/'fine_input.npz';backend.metadata(fine_input,coarse['templates'],fc,fr,fs)
    fine=run_engine(config,out,fine_input,'fine',dict(common,**{'residual-angle':6,'max-cutoff':.22}))
    search=out/'image_allclass_search';search.mkdir(exist_ok=True)
    scores=pixel_scores(fine['templates'],np.arange(n),np.bincount(fine['assignment'],minlength=k),
        search,lambda message:print(message,flush=True),stack)
    candidates=backend.rescue_candidates(scores,fine['assignment'],budget=4)
    rescue_input=out/'rescue_input.npz';backend.full_angles(rescue_input,fine['templates'],candidates,6)
    rescue=run_engine(config,out,rescue_input,'rescue',dict(basic,**{'batch-size':1}))
    union,prior=backend.unique_candidates(rescue['assignment'],
        np.argsort(-scores,axis=1,kind='stable'),fc,fs,budget=4)
    union_input=out/'union_input.npz';backend.full_angles(union_input,rescue['templates'],union,6,prior)
    final=run_engine(config,out,union_input,'cautious_union',dict(basic,**{
        'batch-size':1,'embedding-weight':.10,'template-momentum':.35,
        'min-match-frequency':.015,'temperature':.025,'max-cutoff':.16}))
    averages,mass,dominant=backend.final_average(config,'10028',union_input,final,x,None,out,class_count=k)
    write_stack(out/'class_averages.mrcs',averages,2.8125)
    np.save(out/'final_dominant_labels.npy',dominant.astype('i4'))
    assert averages.shape==(k,128,128) and np.isfinite(averages).all()
    with np.load(out/'final_membership_and_poses.npz') as membership:
        np.testing.assert_allclose(membership['weights'].sum(1),1,atol=2e-6)
        np.testing.assert_allclose(membership['class_mass'].sum(),n,atol=.01)
    record=dict(status='passed',particles=n,classes=k,device=str(device),
        seconds=time.perf_counter()-started,benchmark_result=False,
        output='class_averages.mrcs',model_epoch=epoch)
    save(out/'demo_result.json',record);print(json.dumps(record,indent=2))


if __name__=='__main__':main()
