import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import ProjectionDataset
from .train import ARCH,AMP_DTYPE,BalancedSampler,JointEncoder,save_json,save_pt,seed_all,step


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('examples/tiny65045/train.npz'))
    p.add_argument('--output-dir',type=Path,default=Path('runs/tiny_training'))
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--seed',type=int,default=20261019)
    a=p.parse_args()
    if a.epochs<1:p.error('--epochs must be positive')
    out=a.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory for training from scratch')
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    seed_all(a.seed);torch.set_num_threads(4)
    dataset=ProjectionDataset(a.data,noise_snr_min=.07,noise_snr_max=.20,
        blur_sigma=3,max_shift=4,scale_min=.90,scale_max=1.10,return_structure=True)
    if len(np.unique(dataset.structure_ids))<2:
        raise ValueError('The example requires at least two structures')
    ids=np.arange(len(dataset))
    BalancedSampler(dataset.group_ids,dataset.structure_ids,ids,a.batch_size,a.seed)
    model=JointEncoder().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.0001)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda' and AMP_DTYPE==torch.float16)
    history=[];started=time.perf_counter()
    for epoch in range(1,a.epochs+1):
        model.train()
        sampler=BalancedSampler(dataset.group_ids,dataset.structure_ids,ids,a.batch_size,a.seed,epoch)
        for index,batch in enumerate(DataLoader(dataset,batch_sampler=sampler,num_workers=0),1):
            record=step(model,optimizer,scaler,*(value.to(device) for value in batch))
            record.update(epoch=epoch,step=index,steps=len(sampler))
            history.append(record)
            print(json.dumps(record),flush=True)
    summary=dict(best_epoch=a.epochs,checkpoint_kind='last epoch',benchmark_result=False)
    save_pt(out/'encoder.pt',dict(model_state=model.encoder.state_dict(),args=ARCH,metrics=summary))
    save_pt(out/'adapter.pt',dict(adapter_state=model.adapter.state_dict(),result=summary))
    result=dict(status='passed',purpose='workflow smoke test',benchmark_result=False,
        particles=len(dataset),structures=len(np.unique(dataset.structure_ids)),batch_size=a.batch_size,
        groups_per_structure=a.batch_size//8,particles_per_group=4,epochs=a.epochs,
        optimization_steps=len(history),device=str(device),seconds=time.perf_counter()-started,
        encoder_and_adapter_trainable=True,loss='L_con + 0.5 L_rot',history=history)
    save_json(out/'training_result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))


if __name__=='__main__':main()
