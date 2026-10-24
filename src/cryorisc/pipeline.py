from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import torch.nn.functional as F
from . import backend
from .features import background_normalise,gaussian_blur,rescale_on_canvas,spherical_kmeans
from .image_search import pixel_scores
from .io import read_mrc_memmap,write_stack


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,default=str,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def load_pair(encoder_path,adapter_path,device):
    from .models import CryoC8Encoder,ResidualMetricAdapter
    enc=torch.load(encoder_path,map_location='cpu',weights_only=False)
    ada=torch.load(adapter_path,map_location='cpu',weights_only=False)
    epoch=enc['metrics']['best_epoch']
    if epoch != ada['result']['best_epoch']:
        raise ValueError('Encoder and adapter must be from the same joint epoch')
    model=CryoC8Encoder(**enc['args']).to(device)
    model.train()
    if any(k.endswith('.filter') for k in enc['model_state']):model.eval()
    model.load_state_dict(enc['model_state']);model.eval()
    adapter=ResidualMetricAdapter(enc['args']['embedding_dim'],48).to(device)
    adapter.load_state_dict(ada['adapter_state']);adapter.eval()
    return model,adapter,epoch


@torch.no_grad()
def extract_features(particles,model,adapter,device,profile,batch_size=16):
    output=[]
    for start in range(0,len(particles),batch_size):
        stop=min(start+batch_size,len(particles))
        x=torch.from_numpy(np.array(particles[start:stop]))[:,None].to(device)
        x=background_normalise(x,profile['background_radius'])
        if x.shape[-1]!=128:x=F.interpolate(x,size=(128,128),mode='area')
        views=[]
        for scale in profile['scales']:
            view=gaussian_blur(rescale_on_canvas(x,scale),3.)
            view=(view-view.mean((-2,-1),keepdim=True))/view.std((-2,-1),keepdim=True).clamp_min(1e-6)
            views.append(adapter(model(view)))
        output.append(F.normalize(torch.stack(views).mean(0),dim=1).cpu())
        if start==0 or stop==len(particles) or stop%1000==0:
            print(f'Feature extraction {stop}/{len(particles)}',flush=True)
    return torch.cat(output).numpy().astype('f4')


def run_engine(config,folder,inp,stage,settings,fixed_labels=None):
    output=folder/stage
    output.mkdir(exist_ok=True)
    result=output/'soft_refine_result.npz'
    cmd=[sys.executable,'-u','-m','cryorisc.engine','--input',str(inp),
         '--images-mrcs',str(config['stack']),'--output-dir',str(output),
         '--cpu-images','--subpixel','--aggregate-duplicate-candidates',
         '--mask-radius',str(config['radius']),'--mask-edge',str(config['edge'])]
    for key,value in settings.items():
        if isinstance(value,bool):
            if value:cmd.append('--'+key)
        else:cmd.extend(['--'+key,str(value)])
    if fixed_labels is not None:cmd.extend(['--fixed-labels-npy',str(fixed_labels)])
    manifest=dict(command=cmd,input_sha256=digest(inp),
                  fixed_labels_sha256=digest(fixed_labels) if fixed_labels is not None else None)
    provenance=output/'manifest.json'
    if provenance.exists() and json.loads(provenance.read_text(encoding='utf-8'))!=manifest:
        raise ValueError(f'Stage inputs/settings changed: {stage}; use a fresh output directory')
    save(provenance,manifest)
    save(output/'command.json',cmd)
    if not result.exists():
        print('START '+stage,flush=True)
        save(folder/'status.json',dict(status='running',stage=stage))
        with (output/'run.log').open('a',encoding='utf-8') as log:
            with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,encoding='utf-8',errors='replace') as proc:
                for line in proc.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                if proc.wait():raise RuntimeError(f'{stage} failed; see {output / "run.log"}')
    with np.load(result) as data:fit={k:data[k] for k in data.files}
    if fixed_labels is not None:
        np.testing.assert_array_equal(fit['assignment'],np.load(fixed_labels))
    return fit


def initial_candidates(embeddings,anchors):

    similarity=embeddings@embeddings[anchors].T
    chosen=np.argpartition(-similarity,31,axis=1)[:,:32]
    order=np.argsort(-np.take_along_axis(similarity,chosen,1),axis=1)
    chosen=np.take_along_axis(chosen,order,1).astype('i4')
    return chosen,np.take_along_axis(similarity,chosen,1).astype('f4')


def refine(config,particles,folder,arm,seed):
    labels,anchors,embeddings,coarse_classes,coarse_sims,templates=backend.prepare(
        config,'B_no_network' if arm=='no-network' else 'A_full',seed,particles)
    labels_path=folder/'initial_labels.npy';np.save(labels_path,labels)
    inp=folder/'bootstrap_input.npz'
    backend.full_angles(inp,templates,labels[:,None],6)
    basic={'iterations':1,'batch-size':4,'max-shift':config['shift'],'residual-angle':0,
           'hard-core-template-updates':True,'embedding-weight':0,'class-prior-power':0,
           'template-momentum':0,'max-cutoff':.12}
    boot=run_engine(config,folder,inp,'bootstrap',basic,labels_path)
    coarse_input=folder/'coarse_input.npz'
    if config['historical']:
        if arm!='no-network':coarse_classes,coarse_sims=initial_candidates(embeddings,anchors)
        angles=np.tile(-np.arange(-180,180,30,dtype='f4'),32)[None]
        backend.metadata(coarse_input,boot['templates'],np.repeat(coarse_classes,12,1),
                         angles,np.repeat(coarse_sims,12,1))
    else:backend.full_angles(coarse_input,boot['templates'],coarse_classes,30,coarse_sims)
    common={'iterations':1,'batch-size':config['coarse_batch'],
        'max-shift':min(config['shift'],12),'residual-angle':0,'angle-step':1,
        'marginalize-angles':True,'min-match-frequency':.015,'temperature':.025,
        'embedding-weight':.5,'class-prior-power':.25,'template-momentum':.15,
        'max-cutoff':config['coarse_cutoff']}
    if config['historical']:common['use-precomputed-reflection']=True
    coarse=run_engine(config,folder,coarse_input,'coarse',common)
    if config['historical']:
        with np.load(coarse_input) as data:classes,rotations=data['candidate_classes'],data['pair_rotations_deg']
        similarity=np.zeros((len(labels),200),'f4') if embeddings is None else embeddings@embeddings[anchors].T
        fc,fr,fs=backend.historical_fine_candidates(classes,coarse['pair_posterior'],rotations,similarity)
    else:fc,fr,fs=backend.fine_candidates(coarse_input,coarse,labels,embeddings,anchors,False)
    fine_input=folder/'fine_input.npz'
    backend.metadata(fine_input,coarse['templates'],fc,fr,fs)
    fine=run_engine(config,folder,fine_input,'fine',dict(common,**{
        'iterations':config['fine_iterations'],'batch-size':16,'residual-angle':6,
        'max-cutoff':config['fine_cutoff']}))
    if arm=='no-candidate-rescue':

        ranking=np.argsort(-(embeddings@embeddings[anchors].T),axis=1,kind='stable')
        candidates=ranking[:,:16].copy()
        missing=~np.any(candidates==fine['assignment'][:,None],axis=1)
        candidates[missing,-1]=fine['assignment'][missing]
    else:
        search=folder/'image_allclass_search';search.mkdir(exist_ok=True)
        scores=pixel_scores(fine['templates'],np.arange(len(labels)),
            np.bincount(fine['assignment'],minlength=200),search,
            lambda message:print(message,flush=True),config['stack'])
        ranking=np.argsort(-scores,axis=1,kind='stable')
        candidates=backend.rescue_candidates(scores,fine['assignment'])
    rescue_input=folder/'rescue_input.npz'
    backend.full_angles(rescue_input,fine['templates'],candidates,6)
    fit=run_engine(config,folder,rescue_input,'rescue',dict(basic,**{'batch-size':1}))
    final_input=rescue_input
    if config['union']:
        union,prior=backend.unique_candidates(fit['assignment'],ranking,fc,fs)
        final_input=folder/'cautious_union_input.npz'
        backend.full_angles(final_input,fit['templates'],union,6,prior)
        fit=run_engine(config,folder,final_input,'cautious_union',dict(basic,**{
            'batch-size':1,'embedding-weight':.10,'template-momentum':.35,
            'min-match-frequency':.015,'temperature':.025,'max-cutoff':.16}))
    averages,counts,dominant=backend.final_average(config,config['final_band_profile'],
        final_input,fit,particles,None,folder)
    write_stack(folder/'class_averages.mrcs',averages,config['apix'])
    np.save(folder/'final_dominant_labels.npy',dominant.astype('i4'))
    np.save(folder/'effective_class_mass.npy',counts)
    save(folder/'status.json',dict(status='complete',particles=len(particles),classes=200,
        arm=arm,output='class_averages.mrcs',centered=False,reconstructed=False))


def classify(args,profile):
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu' and not args.allow_cpu:
        raise RuntimeError('CUDA recommended/required by default; --allow-cpu enables slow diagnostic runs')
    particles=read_mrc_memmap(args.input)
    if particles.ndim!=3 or particles.shape[1:]!=(profile['size'],profile['size']):
        raise ValueError(f"Expected native square box {profile['size']}, got {particles.shape}")
    if len(particles)<200:raise ValueError('This release uses 200 classes and requires at least 200 particles')
    if len(particles)!=profile['expected_particles']:
        print('NOTE: non-paper particle count; report it as a new experiment',flush=True)
    if args.arm!='no-network' and (not args.encoder or not args.adapter):
        raise ValueError('--encoder and --adapter are required except for --arm no-network')
    if args.preflight:
        if args.arm=='no-network':
            print(f'Preflight passed: {particles.shape}, no network is loaded');return
        model,adapter,epoch=load_pair(args.encoder,args.adapter,device)
        z=extract_features(particles[:16],model,adapter,device,profile,args.feature_batch)
        np.testing.assert_allclose(np.linalg.norm(z,axis=1),1.,atol=2e-5)
        print(json.dumps(dict(preflight='passed',embedding_shape=list(z.shape),epoch=epoch)));return
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    config=dict(profile,stack=args.input.resolve(),source=out/'features_and_groups.npz',particles=len(particles))
    if args.apix is not None:config['apix']=args.apix
    if config['apix']<=0:raise ValueError('Pixel size must be positive')

    manifest=dict(profile=profile,apix=config['apix'],input_sha256=digest(args.input),
        input_name=args.input.name,arm=args.arm,seed=args.seed,feature_batch=args.feature_batch,
        encoder_sha256=digest(args.encoder) if args.arm!='no-network' else None,
        adapter_sha256=digest(args.adapter) if args.arm!='no-network' else None,
        code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Different inputs/code/config in output directory; choose a fresh directory')
    save(path,manifest)
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text(encoding='utf-8')).get('status')=='complete':
        print('Already complete: '+str(out));return
    if args.arm!='no-network' and not config['source'].exists():
        model,adapter,epoch=load_pair(args.encoder,args.adapter,device)
        z=extract_features(particles,model,adapter,device,profile,args.feature_batch)
        labels,centres=spherical_kmeans(z,200,device,profile['cluster_seed'])
        sim=z@centres.T
        anchors=np.array([np.flatnonzero(labels==c)[np.argmax(sim[labels==c,c])] for c in range(200)])
        np.savez_compressed(config['source'],embeddings=z,initial_kmeans_labels=labels,
            selected_anchor_ids=anchors,cluster_centres=centres,model_epoch=epoch)
        del model,adapter,z,sim
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    if args.features_only:
        print('Features complete: '+str(config['source']));return
    refine(config,particles,out,args.arm,args.seed)
