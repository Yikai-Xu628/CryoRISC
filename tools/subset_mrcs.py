import argparse
from pathlib import Path
import numpy as np
from cryorisc.io import read_mrc_memmap,write_stack

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,required=True)
    p.add_argument('--apix',type=float,required=True)
    a=p.parse_args()
    x=read_mrc_memmap(a.input)
    if not 0<a.count<=len(x):p.error('count must be between 1 and the source size')
    if a.apix<=0:p.error('apix must be positive')
    indices=np.floor((np.arange(a.count)+.5)*len(x)/a.count).astype('i8')
    ids_path=a.output.with_suffix('.indices.npy')
    if a.output.exists() or ids_path.exists():raise FileExistsError('Choose new output paths')
    write_stack(a.output,np.asarray(x[indices]),a.apix)
    np.save(ids_path,indices)
    print(a.output,ids_path)

if __name__=='__main__':main()
