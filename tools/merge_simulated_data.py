import argparse
import json
from pathlib import Path
import numpy as np


def merge_grouped(root,output):
    import h5py
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/joint_training_record.json').read_text(encoding='utf-8'))
    structures=config['train_structures']
    target=output/'multistructure_v2_train.npz'
    if target.exists():raise FileExistsError('Choose a new output directory')
    sources=[(sid,name,root/name.lower()/'emd33026_grouped_train.mat') for sid,name in enumerate(structures)]
    for _,_,path in sources:
        if not path.is_file():raise FileNotFoundError(path)
    arrays={key:[] for key in ('images','viewdirs','group_ids','structure_ids','structure_names')}
    offset=0
    for sid,name,path in sources:
        with h5py.File(path,'r') as d:
            groups=np.asarray(d['group_ids'],dtype='i8').reshape(-1)
            vectors=np.asarray(d['viewdirs'],dtype='f4')
            if vectors.shape!=(len(groups),3):vectors=vectors.T
            images=np.asarray(d['projs'],dtype='f4')
            if images.shape[0]!=len(groups):images=np.moveaxis(images,-1,0)
        if images.shape!=(len(groups),128,128) or vectors.shape!=(len(groups),3):
            raise ValueError(f'Unexpected projection dimensions: {path}')
        if len(np.unique(groups))<18 or min(np.unique(groups,return_counts=True)[1])<4:
            raise ValueError(f'Generate full grouped training projections for {name}: {path}')
        if not np.isfinite(images).all() or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors,axis=1)==0):
            raise ValueError(f'Invalid projections or directions: {path}')
        groups=groups-groups.min()+offset;offset=int(groups.max())+1
        values=dict(images=images,viewdirs=vectors,group_ids=groups,
            structure_ids=np.full(len(groups),sid,'i8'),structure_names=np.full(len(groups),name))
        for key,value in values.items():arrays[key].append(value)
        print(f'{name}: {len(images)} particles from {path}',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**{key:np.concatenate(value) for key,value in arrays.items()})
    print(target,flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--inputs',type=Path,nargs='+')
    p.add_argument('--structure-ids',type=int,nargs='+')
    p.add_argument('--output',type=Path)
    p.add_argument('--grouped-root',type=Path)
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args()
    if a.grouped_root is not None:
        if a.output_dir is None:p.error('--grouped-root requires --output-dir')
        if a.inputs is not None or a.structure_ids is not None or a.output is not None:
            p.error('Use either --grouped-root/--output-dir or --inputs/--structure-ids/--output')
        return merge_grouped(a.grouped_root,a.output_dir)
    if a.inputs is None or a.structure_ids is None or a.output is None:
        p.error('Provide --inputs, --structure-ids and --output')
    if len(a.inputs)!=len(a.structure_ids):p.error('Provide one structure ID per input')
    if len(set(a.structure_ids))!=len(a.structure_ids):p.error('Structure IDs must be distinct')
    if a.output.exists():raise FileExistsError(a.output)
    images=[];dirs=[];groups=[];structures=[];offset=0
    for path,sid in zip(a.inputs,a.structure_ids):
        with np.load(path,allow_pickle=False) as d:
            x=d['images'].astype('f4');v=d['viewdirs'].astype('f4');g=d['group_ids'].astype('i8')
        if x.ndim!=3 or x.shape[1:]!=(128,128) or v.shape!=(len(x),3) or g.shape!=(len(x),):
            raise ValueError(f'Invalid grouped projection arrays: {path}')
        if not np.isfinite(x).all() or not np.isfinite(v).all() or np.any(np.linalg.norm(v,axis=1)==0):
            raise ValueError(f'Nonfinite images/directions or zero directions: {path}')
        g=g-g.min()+offset;offset=int(g.max())+1
        images.append(x);dirs.append(v);groups.append(g);structures.append(np.full(len(x),sid,'i8'))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,images=np.concatenate(images),viewdirs=np.concatenate(dirs),
        group_ids=np.concatenate(groups),structure_ids=np.concatenate(structures))
    print(a.output)

if __name__=='__main__':main()
