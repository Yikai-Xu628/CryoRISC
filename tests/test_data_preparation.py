import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import numpy as np


ROOT=Path(__file__).resolve().parents[1]


def test_matlab_generator_uses_only_the_five_training_structures():
    config=json.loads((ROOT/'configs/joint_training_record.json').read_text(encoding='utf-8'))
    source=(ROOT/'matlab/generate_training_projections.m').read_text(encoding='utf-8')
    names=re.findall(r"'(emd[0-9]+)'",source)
    assert names==[name.lower() for name in config['train_structures']]
    assert 'groups=[300 0 0]' in source
    assert 'groups=[200 0 0]' in source


def test_grouped_merger_requires_only_training_files(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('merge',ROOT/'tools/merge_simulated_data.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    names=json.loads((ROOT/'configs/joint_training_record.json').read_text(encoding='utf-8'))['train_structures']
    inputs={};opened=[]
    for sid,name in enumerate(names):
        path=tmp_path/'projections'/name.lower()/'emd33026_grouped_train.mat'
        path.parent.mkdir(parents=True);path.touch()
        inputs[path]=dict(projs=np.full((72,128,128),sid+1,dtype='f4'),
            group_ids=np.repeat(np.arange(18),4),viewdirs=np.tile([1.,0,0],(72,1)))
    class Source:
        def __init__(self,path,mode):
            self.path=path;opened.append(path)
            assert mode=='r'
        def __enter__(self):return inputs[self.path]
        def __exit__(self,*args):return False
    monkeypatch.setitem(sys.modules,'h5py',SimpleNamespace(File=Source))
    output=tmp_path/'merged'
    module.merge_grouped(tmp_path/'projections',output)
    assert opened==list(inputs)
    assert [path.name for path in output.iterdir()]==['multistructure_v2_train.npz']
    with np.load(output/'multistructure_v2_train.npz') as data:
        assert data['images'].shape==(360,128,128)
        assert set(data['structure_names'])==set(names)
        for sid in range(5):
            np.testing.assert_array_equal(np.unique(data['images'][data['structure_ids']==sid]),[sid+1])
