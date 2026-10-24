import argparse
from pathlib import Path
import h5py
import numpy as np

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--structure-id',type=int,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with h5py.File(a.input,'r') as f:
        images=np.asarray(f['projs'],dtype='f4')
        directions=np.asarray(f['viewdirs'],dtype='f4')
        groups=np.asarray(f['group_ids'],dtype='i8').reshape(-1)
    if images.shape[0]!=len(groups):images=np.moveaxis(images,-1,0)
    if directions.shape!=(len(groups),3):directions=directions.T
    if images.shape!=(len(groups),128,128) or directions.shape!=(len(groups),3):
        raise ValueError('Unexpected MATLAB arrays; inspect dimensions/order before conversion')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,images=images,viewdirs=directions,group_ids=groups,
        structure_ids=np.full(len(groups),a.structure_id,dtype='i8'))
    print(a.output)

if __name__=='__main__':main()
