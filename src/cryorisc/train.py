from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from .models import CryoC8Encoder
from .data import ProjectionDataset
from .losses import supervised_direction_contrastive_loss
from .models import ResidualMetricAdapter

ROOT = Path(__file__).resolve().parent
DATA = Path('data/simulated')
OUT = Path('runs/joint_training')
STRUCTURES = {0:'EMD33026',1:'EMD70734',2:'EMD22689',3:'EMD20841',4:'EMD3225'}
ARCH = dict(embedding_dim=96, rotations=16, full_c16=True, field_multiplier=1.5,
            residual_blocks=1, multiscale_radial=False, radial_bins=4, preserve_regular_fields=False)
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def emit(value):
    print(json.dumps(value,ensure_ascii=False,default=str),flush=True)


def save_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    os.replace(temp,path)


def save_pt(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temp);os.replace(temp,path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=CryoC8Encoder(**ARCH)
        self.adapter=ResidualMetricAdapter(96,48)

    def forward(self,x):

        return F.normalize(self.adapter(self.encoder(x).float()).float(),dim=1)


class BalancedSampler:

    def __init__(self,groups,structures,train_ids,batch_size,seed,epoch=0):
        self.by={}
        for ix in train_ids:
            self.by.setdefault(int(structures[ix]),{}).setdefault(int(groups[ix]),[]).append(int(ix))
        self.structures=sorted(self.by)
        self.groups_each=batch_size//8;self.samples_each=4
        if batch_size % 8 or self.groups_each<2:raise ValueError('Invalid contrastive batch')
        for s,mapping in self.by.items():
            if len(mapping)<self.groups_each or min(map(len,mapping.values()))<4:
                raise ValueError(f'Insufficient unique groups/particles for structure {s}')
        self.steps=math.ceil(len(train_ids)/batch_size)
        self.seed=seed;self.epoch=epoch

    def __len__(self):return self.steps

    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)


        order=np.array(self.structures);rng.shuffle(order)
        for step in range(self.steps):
            chosen=order[(2*step+np.arange(2))%len(order)]
            batch=[]
            for s in chosen:
                mapping=self.by[int(s)]
                selected=rng.choice(list(mapping),self.groups_each,replace=False)
                for g in selected:batch.extend(rng.choice(mapping[int(g)],4,replace=False).tolist())
            rng.shuffle(batch);yield batch


def loss_with_diagnostics(z,directions,groups,structures):
    loss=supervised_direction_contrastive_loss(z,directions,groups,structure_ids=structures)
    with torch.no_grad():
        eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
        same=structures[:,None].eq(structures[None,:])
        positive=groups[:,None].eq(groups[None,:])&same&~eye
        angle=torch.rad2deg(torch.acos((directions@directions.T).abs().clamp(0,1)))
        negative=(angle>=15)&same&~eye
        if not positive.any(1).all() or not negative.any(1).all():
            raise RuntimeError('Batch lacks positive or valid same-structure negative samples')
        sim=z@z.T
        diag=dict(positive_similarity=float(sim[positive].mean()),negative_similarity=float(sim[negative].mean()),
                  positives_per_anchor=float(positive.sum(1).float().mean()),
                  negatives_per_anchor=float(negative.sum(1).float().mean()),
                  min_negatives=int(negative.sum(1).min()))
    return loss,diag


def step(model,optimizer,scaler,images,dirs,groups,structures):
    optimizer.zero_grad(set_to_none=True)


    count=min(16,len(images));subset=torch.randperm(len(images),device=images.device)[:count]
    quarter=torch.randint(1,4,(count,),device=images.device)
    rotated=torch.stack([torch.rot90(x,int(k),(-2,-1)) for x,k in zip(images[subset],quarter)])
    with torch.cuda.amp.autocast(enabled=images.is_cuda,dtype=AMP_DTYPE):
        z=model(images);zr=model(rotated)
    structure,diag=loss_with_diagnostics(z,dirs,groups,structures)
    rotation=(1-(z[subset]*zr).sum(1)).mean()
    loss=structure+.5*rotation
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
    encoder_grad=sum(float(p.grad.detach().float().norm()) for p in model.encoder.parameters() if p.grad is not None)
    adapter_grad=sum(float(p.grad.detach().float().norm()) for p in model.adapter.parameters() if p.grad is not None)
    if encoder_grad<=0 or adapter_grad<=0:raise RuntimeError('A trainable module received no gradient')
    scaler.step(optimizer);scaler.update()
    return dict(loss=float(loss.detach()),contrastive=float(structure.detach()),rotation=float(rotation.detach()),
                grad_norm=float(norm),encoder_grad=encoder_grad,adapter_grad=adapter_grad,**diag)


def probe(out,seed):
    torch.set_num_threads(4)
    results=[]
    for batch in (128,96,64,48,32):
        seed_all(seed);model=JointEncoder().cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
        scaler=torch.cuda.amp.GradScaler(enabled=AMP_DTYPE==torch.float16)
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        try:
            images=torch.randn(batch,1,128,128,device='cuda')
            groups=torch.arange(batch,device='cuda')//4
            structures=torch.arange(batch,device='cuda')//(batch//2)
            vectors=F.normalize(torch.randn(batch//4,3,device='cuda'),dim=1)
            dirs=vectors[groups]
            records=[step(model,optimizer,scaler,images,dirs,groups,structures) for _ in range(3)]
            torch.cuda.synchronize()
            result=dict(batch_size=batch,structures_per_batch=2,groups_per_structure=batch//8,
                        samples_per_group=4,amp=True,amp_dtype=str(AMP_DTYPE),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                        seconds_per_step=(time.perf_counter()-start)/3,last_step=records[-1],success=True)
            results.append(result)
            if result['peak_gpu_gib']<5.7:
                save_json(out/'gpu_probe.json',dict(selected_batch=batch,probes=results))
                emit({'GPU_PROBE':result});return batch
        except torch.cuda.OutOfMemoryError:
            results.append(dict(batch_size=batch,success=False,error='CUDA OOM'))
        finally:
            del model,optimizer,scaler
            for name in ('images','dirs','groups','structures','vectors','records'):
                if name in locals():del locals()[name]
            gc.collect();torch.cuda.empty_cache()
    raise RuntimeError('No batch size passed the GPU memory probe')


def split_groups(dataset,seed):
    rng=np.random.default_rng(seed)
    held=[]
    for s in sorted(np.unique(dataset.structure_ids)):
        ids=np.flatnonzero(dataset.structure_ids==s)
        groups=np.unique(dataset.group_ids[ids]);rng.shuffle(groups)
        selected=groups[:max(2,round(.10*len(groups)))]
        held.extend(ids[np.isin(dataset.group_ids[ids],selected)].tolist())
    held=np.sort(np.asarray(held,dtype=np.int64))
    train=np.setdiff1d(np.arange(len(dataset)),held)
    for s in np.unique(dataset.structure_ids):
        a=set(dataset.group_ids[train[dataset.structure_ids[train]==s]])
        b=set(dataset.group_ids[held[dataset.structure_ids[held]==s]])
        assert not a&b
    return train,held


def validation_score(results):
    return float(np.mean([result['macro_precision10_at8deg'] for result in results]))


@torch.no_grad()
def evaluate(model,dataset,ids,snr,noise_seed):
    model.eval()
    previous=(dataset.noise_snr_min,dataset.noise_snr_max,dataset.deterministic_noise,dataset.noise_seed,dataset.scale_min,dataset.scale_max)
    dataset.noise_snr_min=dataset.noise_snr_max=snr
    dataset.deterministic_noise=True;dataset.noise_seed=noise_seed
    dataset.scale_min=dataset.scale_max=1.0
    try:
        all_z=[];all_d=[];all_s=[]
        for images,dirs,_,structures in DataLoader(Subset(dataset,ids),batch_size=16,shuffle=False,num_workers=0):
            with torch.cuda.amp.autocast(enabled=True,dtype=AMP_DTYPE):z=model(images.cuda())
            all_z.append(z.cpu());all_d.append(dirs);all_s.append(structures)
        z=F.normalize(torch.cat(all_z),dim=1);d=torch.cat(all_d);s=torch.cat(all_s)
        result={}
        for sid in torch.unique(s):
            keep=s==sid;e=z[keep];v=d[keep]
            sim=e@e.T;sim.fill_diagonal_(-float('inf'))
            near=sim.topk(min(10,len(e)-1),dim=1).indices
            angles=torch.rad2deg(torch.acos((v[:,None]*v[near]).sum(2).abs().clamp(0,1)))
            result[STRUCTURES[int(sid)]]=dict(n=len(e),precision10_at8deg=float((angles<=8).float().mean()),
                                            median_angle_deg=float(angles.median()),
                                            feature_std=float(e.std(0).mean()))
        return dict(snr=snr,by_structure=result,macro_precision10_at8deg=float(np.mean([v['precision10_at8deg'] for v in result.values()])),
                    macro_median_angle_deg=float(np.mean([v['median_angle_deg'] for v in result.values()])))
    finally:
        (dataset.noise_snr_min,dataset.noise_snr_max,dataset.deterministic_noise,dataset.noise_seed,dataset.scale_min,dataset.scale_max)=previous


def main():
    global DATA
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output-dir',type=Path,default=OUT)
    p.add_argument('--data-dir',type=Path,default=DATA)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--seed',type=int,default=20261014)
    p.add_argument('--profile-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();DATA=args.data_dir
    if not torch.cuda.is_available():raise RuntimeError('Joint training requires CUDA; use CPU for unit tests only')
    if args.batch_size < 16 or args.batch_size % 8:raise ValueError('Batch must be a multiple of 8, at least 16')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    try:
        if args.profile_only:probe(out,args.seed);return
        if (out/'last.pt').exists() and not args.resume:raise FileExistsError('Existing run: use --resume or a new output directory')
        batch=args.batch_size
        seed_all(args.seed)
        dataset=ProjectionDataset(DATA/'multistructure_v2_train.npz',noise_snr_min=.07,noise_snr_max=.20,
                                  blur_sigma=3,max_shift=4,scale_min=.90,scale_max=1.10,return_structure=True)
        if set(dataset.structure_ids)!=set(STRUCTURES):raise ValueError('Unexpected training structures')
        train_ids,val_ids=split_groups(dataset,args.seed)
        np.savez_compressed(out/'split_indices.npz',train_ids=train_ids,validation_ids=val_ids)
        config=dict(seed=args.seed,architecture=ARCH,from_random_initialization=True,pretrained_checkpoint=None,
                    encoder_and_adapter_trainable=True,old_preservation_loss=False,train_structures=[STRUCTURES[i] for i in range(5)],
                    train_samples=len(train_ids),seen_validation_samples=len(val_ids),
                    validation_split='held-out view groups within the five training structures',validation_fraction=.10,
                    test_structure='EMD65045',test_used_for_selection=False,batch_size=batch,structures_per_batch=2,
                    groups_per_structure=batch//8,samples_per_group=4,train_snr=[.07,.20],train_scale=[.90,1.10],max_shift=4,
                    blur_sigma=3,rotation_consistency_weight=.5,rotation_subset=16,rotation_angles=[90,180,270],
                    optimizer='AdamW',peak_lr=.0008,weight_decay=.0001,gradient_clip=2,amp=True,amp_dtype=str(AMP_DTYPE),

                    epochs=args.epochs,warmup_epochs=5,lr_schedule='cosine',minimum_lr=.00004,
                    validation_snrs=[.03,.10],validation_every_epochs=2,
                    selection='mean held-out view-group macro precision10_at8deg over validation SNRs',
                    early_stop_minimum_epoch=30,early_stop_patience_epochs=16,
                    source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                   [Path(__file__).name,'models.py','data.py','losses.py']},
                    input_signatures={str(path):dict(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns) for path in
                                      [DATA/'multistructure_v2_train.npz']})
        if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:raise ValueError('Run configuration changed')
        save_json(out/'config.json',config)
        model=JointEncoder().cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.0001)
        scaler=torch.cuda.amp.GradScaler(enabled=AMP_DTYPE==torch.float16);history=[];best=-1.;best_epoch=0;start_epoch=1
        if args.resume:
            state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)


            model.train();model.load_state_dict(state['joint_state'])
            optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
            history=state['history'];best=state['best'];best_epoch=state['best_epoch'];start_epoch=state['epoch']+1
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        emit({'START':config})
        total_start=time.perf_counter()
        for epoch in range(start_epoch,args.epochs+1):
            if epoch<=5:lr=.0008*epoch/5
            else:lr=.00004+.5*(.0008-.00004)*(1+math.cos(math.pi*(epoch-5)/max(1,args.epochs-5)))
            for group in optimizer.param_groups:group['lr']=lr
            model.train()
            sampler=BalancedSampler(dataset.group_ids,dataset.structure_ids,train_ids,batch,args.seed,epoch)
            loader=DataLoader(dataset,batch_sampler=sampler,num_workers=0)
            metrics=[];started=time.perf_counter()
            for ix,(images,dirs,groups,structures) in enumerate(loader,1):
                m=step(model,optimizer,scaler,images.cuda(),dirs.cuda(),groups.cuda(),structures.cuda());metrics.append(m)
                if ix==1 or ix%25==0:
                    seconds=time.perf_counter()-started
                    progress=dict(status='training',pid=os.getpid(),epoch=epoch,max_epochs=args.epochs,step=ix,steps=len(sampler),
                                  learning_rate=lr,seconds_per_step=seconds/ix,epoch_elapsed_seconds=seconds,**m)
                    save_json(out/'status.json',progress);emit(progress)
            record=dict(epoch=epoch,learning_rate=lr,seconds=time.perf_counter()-started,
                        **{key:float(np.mean([v[key] for v in metrics])) for key in metrics[0]})
            if epoch==1 or epoch%2==0 or epoch==args.epochs:
                save_json(out/'status.json',dict(status='validating',epoch=epoch,pid=os.getpid()))
                seen_metrics=[]
                for snr in (.03,.10):
                    seen_metrics.append(evaluate(model,dataset,val_ids,snr,args.seed+101))
                score=validation_score(seen_metrics)
                record.update(validation_score=score,seen_validation=seen_metrics)
                noncollapsed=all(v['feature_std']>.001 for result in seen_metrics for v in result['by_structure'].values())
                if score>best and noncollapsed:
                    best=float(score);best_epoch=epoch
                    summary=dict(best_epoch=epoch,best_validation_score=best,protocol='from-scratch joint multi-structure encoder + residual head',validation=record)
                    save_pt(out/'best_encoder.pt',dict(model_state=model.encoder.state_dict(),args=ARCH,metrics=summary))
                    save_pt(out/'best_adapter.pt',dict(adapter_state=model.adapter.state_dict(),result=summary))
                    save_json(out/'best_metrics.json',summary)
            history.append(record);save_json(out/'history.json',history)
            save_pt(out/'last.pt',dict(joint_state=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                                     epoch=epoch,history=history,best=best,best_epoch=best_epoch,config=config,
                                     torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                                     numpy_rng=np.random.get_state(),python_rng=random.getstate()))
            save_json(out/'status.json',dict(status='epoch_complete',pid=os.getpid(),epoch=epoch,best_epoch=best_epoch,
                                           best_validation_score=best,total_seconds=time.perf_counter()-total_start))
            emit({'EPOCH_COMPLETE':record,'best_epoch':best_epoch,'best_score':best})
            if epoch>=30 and epoch-best_epoch>=16:break
        save_json(out/'status.json',dict(status='complete',epochs_completed=epoch,best_epoch=best_epoch,best_validation_score=best,
                                       encoder=str(out/'best_encoder.pt'),adapter=str(out/'best_adapter.pt'),
                                       test_evaluated=False,total_seconds=time.perf_counter()-total_start))
        emit({'COMPLETE':str(out),'best_epoch':best_epoch,'test_not_used':True})
    except Exception as error:
        save_json(out/'status.json',dict(status='failed',pid=os.getpid(),error=repr(error)))
        raise


if __name__=='__main__':main()
