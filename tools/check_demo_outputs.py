import argparse
import json
from pathlib import Path
import numpy as np
from cryorisc.io import read_mrc_memmap
from cryorisc.metrics import intra_class_angular_consistency


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,default=Path('runs/tiny65045'))
    p.add_argument('--example-dir',type=Path,default=Path('examples/tiny65045'))
    a=p.parse_args()
    meta=json.loads((a.example_dir/'manifest.json').read_text(encoding='utf-8'))
    n,k,size=meta['particles'],meta['classes'],meta['size']
    avg=read_mrc_memmap(a.output_dir/'class_averages.mrcs')
    labels=np.load(a.output_dir/'final_dominant_labels.npy')
    assert avg.shape==(k,size,size) and np.isfinite(avg).all()
    assert labels.shape==(n,) and np.all((labels>=0)&(labels<k))
    with np.load(a.output_dir/'final_membership_and_poses.npz') as d:
        assert d['classes'].shape==(n,4)
        assert d['weights'].shape==(n,4) and np.isfinite(d['weights']).all()
        assert np.all(d['weights']>=0)
        np.testing.assert_allclose(d['weights'].sum(1),1,atol=2e-6)
        np.testing.assert_array_equal(labels,d['classes'][np.arange(n),d['weights'].argmax(1)])
        np.testing.assert_allclose(d['class_mass'].sum(),n,atol=.03,rtol=0)
        for name in ('angles','dy','dx'):assert np.isfinite(d[name]).all()
    scores=intra_class_angular_consistency(np.load(a.example_dir/'viewdirs.npy'),labels)
    scores.pop('per_particle')
    record=dict(status='passed',particles=n,classes=k,average_shape=list(avg.shape),
                iac=scores,benchmark_result=False)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
