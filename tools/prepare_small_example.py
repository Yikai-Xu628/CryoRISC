import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cryorisc.io import read_mrc_memmap,write_stack


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--directions',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--count',type=int,default=32)
    a=p.parse_args()
    x=read_mrc_memmap(a.input)
    if x.shape[1:]!=(128,128) or not 8<=a.count<=len(x):
        raise ValueError('Expected at least 8 selected particles from a 128-pixel stack')
    data=np.load(a.directions,allow_pickle=False)
    if isinstance(data,np.lib.npyio.NpzFile):
        with data:vectors=data['viewdirs']
    else:vectors=data
    if vectors.shape!=(len(x),3) or not np.isfinite(vectors).all():
        raise ValueError('Directions must match the original particle order')
    out=a.root/'examples/tiny65045'
    out.mkdir(parents=True,exist_ok=True)
    targets=[out/'particles.mrcs',out/'viewdirs.npy',out/'source_indices.npy',
             out/'manifest.json']
    if any(path.exists() for path in [*targets[:3],targets[-1]]):
        raise FileExistsError('Example files already exist; choose a fresh output root')
    ids=np.floor((np.arange(a.count)+.5)*len(x)/a.count).astype('i8')
    write_stack(targets[0],np.asarray(x[ids]),2.8125)
    np.testing.assert_array_equal(read_mrc_memmap(targets[0]),x[ids])
    np.save(targets[1],vectors[ids]);np.save(targets[2],ids)
    record=dict(structure='EMD-65045',purpose='workflow smoke test, not a benchmark',
        particles=a.count,source_particles=len(x),size=128,apix=2.8125,
        source_stack_name=a.input.name,source_stack_sha256=sha256(a.input),
        selection='floor((arange(count)+0.5)*source_particles/count), zero-based',
        source_noise_snr=.1,noise_added=False,particle_values_unchanged=True,
        directions='actual per-particle simulation directions; evaluation only',
        classification_profile='tiny demonstration',classes=8,
        files={path.relative_to(a.root).as_posix():dict(bytes=path.stat().st_size,
               sha256=sha256(path)) for path in targets[:-1]})
    targets[-1].write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
