import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def select_groups(vectors,groups):
    unique=np.unique(groups)
    selected=[unique[0]]
    centres=np.stack([vectors[np.flatnonzero(groups==g)[0]] for g in unique])
    centres=centres/np.linalg.norm(centres,axis=1,keepdims=True)
    while len(selected)<4:
        positions=np.searchsorted(unique,selected)
        separation=np.rad2deg(np.arccos(np.abs(centres@centres[positions].T).clip(0,1))).min(1)
        separation[positions]=-1
        index=int(separation.argmax())
        if separation[index]<30:raise ValueError('Four sufficiently separated view groups are required')
        selected.append(unique[index])
    return selected


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',action='append',required=True,help='STRUCTURE=grouped.mat; supply two sources')
    p.add_argument('--output-dir',type=Path,default=Path('examples/tiny65045'))
    a=p.parse_args()
    if len(a.source)!=2:p.error('Supply two structures')
    paths=[a.output_dir/'train.npz',a.output_dir/'training_manifest.json']
    if any(path.exists() for path in paths):raise FileExistsError('Training example already exists')
    arrays={key:[] for key in ('images','viewdirs','group_ids','structure_ids','source_indices','source_group_ids')}
    sources=[]
    for sid,spec in enumerate(a.source):
        name,path=spec.split('=',1)
        with h5py.File(path,'r') as data:
            groups=np.asarray(data['group_ids']).reshape(-1).astype('i8')
            vectors=np.asarray(data['viewdirs'],dtype='f4')
            if vectors.shape!=(len(groups),3):vectors=vectors.T
            chosen=select_groups(vectors,groups)
            selected=[];labels=[]
            for gidx,group in enumerate(chosen):
                candidates=np.flatnonzero(groups==group)
                if len(candidates)<4:raise ValueError('At least four projections per group are required')
                ids=candidates[np.linspace(0,len(candidates)-1,4,dtype=int)]
                selected.extend(ids.tolist());labels.extend([sid*4+gidx]*4)
            selected=np.asarray(selected,dtype='i8')
            projection=data['projs']
            images=np.stack([np.asarray(projection[int(i)] if projection.shape[0]==len(groups)
                else projection[:,:,int(i)],dtype='f4') for i in selected])
            if images.shape!=(16,128,128):raise ValueError('Expected 128-pixel projections')
            values=dict(images=images,viewdirs=vectors[selected],group_ids=np.asarray(labels,dtype='i8'),
                structure_ids=np.full(16,sid,dtype='i8'),source_indices=selected,source_group_ids=groups[selected])
            for key,value in values.items():arrays[key].append(value)
            sources.append(dict(structure_id=sid,structure=name,source_file=Path(path).name,
                source_indices=selected.tolist(),source_group_ids=groups[selected].tolist()))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(paths[0],**{key:np.concatenate(value) for key,value in arrays.items()})
    record=dict(particles=32,structures=sources,view_groups_per_structure=4,particles_per_group=4,
        clean_projections=True,training_noise_snr=[.07,.20],training_blur_sigma=3,
        test_structure='EMD65045',test_particles=32,purpose='workflow smoke test, not a benchmark')
    paths[1].write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
