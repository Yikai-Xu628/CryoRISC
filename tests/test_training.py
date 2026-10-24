import numpy as np
import pytest
import torch
import json
from pathlib import Path
pytest.importorskip('escnn')
from cryorisc.train import BalancedSampler,JointEncoder,split_groups
from cryorisc.losses import supervised_direction_contrastive_loss
from cryorisc.pipeline import load_pair
from cryorisc.train import ARCH,STRUCTURES,validation_score


def test_actual_128_particle_batch_contract():
    structures=np.repeat(np.arange(5),16*4)
    groups=np.repeat(np.arange(5*16),4)
    sampler=BalancedSampler(groups,structures,np.arange(len(groups)),128,9)
    for batch in sampler:
        batch=np.array(batch)
        assert len(batch)==len(set(batch))==128
        s,counts=np.unique(structures[batch],return_counts=True)
        assert len(s)==2 and np.all(counts==64)
        for structure in s:
            g,n=np.unique(groups[batch[structures[batch]==structure]],return_counts=True)
            assert len(g)==16 and np.all(n==4)


def test_group_disjoint_split():
    class Dataset:
        group_ids=np.repeat(np.arange(100),4)
        structure_ids=np.repeat(np.arange(5),80)
        def __len__(self):return len(self.group_ids)
    d=Dataset();train,val=split_groups(d,8)
    assert not set(d.group_ids[train]) & set(d.group_ids[val])


def test_encoder_and_adapter_both_receive_gradients():
    torch.set_num_threads(2);torch.manual_seed(9)
    model=JointEncoder().train()
    x=torch.randn(8,1,32,32)
    z=model(x)
    assert z.shape==(8,96)
    torch.testing.assert_close(z.norm(dim=1),torch.ones(8))
    v=torch.tensor([[1.,0,0],[1.,0,0],[0.,1,0],[0.,1,0]]*2)
    groups=torch.arange(8)//2;structures=torch.arange(8)//4
    loss=supervised_direction_contrastive_loss(z,v,groups,structure_ids=structures)
    zr=model(torch.rot90(x,1,(-2,-1)))
    (loss+.5*(1-(z*zr).sum(1)).mean()).backward()
    for module in (model.encoder,model.adapter):
        norms=[p.grad.norm().item() for p in module.parameters() if p.grad is not None]
        assert sum(norms)>0 and np.isfinite(norms).all()


@pytest.mark.parametrize('evaluation_state',[False,True])
def test_joint_checkpoints_load_in_training_and_evaluation_modes(tmp_path,evaluation_state):
    torch.set_num_threads(2)
    model=JointEncoder()
    model.eval() if evaluation_state else model.train()
    summary={'best_epoch':2}
    encoder=tmp_path/'encoder.pt';adapter=tmp_path/'adapter.pt'
    with encoder.open('wb') as f:torch.save(dict(model_state=model.encoder.state_dict(),args=ARCH,metrics=summary),f)
    with adapter.open('wb') as f:torch.save(dict(adapter_state=model.adapter.state_dict(),result=summary),f)
    expected=model.eval()(torch.ones(2,1,32,32)).detach()
    loaded,head,epoch=load_pair(encoder,adapter,torch.device('cpu'))
    with torch.no_grad():actual=head(loaded(torch.ones(2,1,32,32)))
    assert epoch==2
    torch.testing.assert_close(actual,expected)


def test_tiny_training_batch_contract():
    structures=np.repeat([0,1],16)
    groups=np.repeat(np.arange(8),4)
    sampler=BalancedSampler(groups,structures,np.arange(32),16,20261019)
    assert len(sampler)==2
    for batch in sampler:
        batch=np.asarray(batch)
        assert len(np.unique(batch))==16
        for sid in (0,1):
            chosen=groups[batch[structures[batch]==sid]]
            _,counts=np.unique(chosen,return_counts=True)
            np.testing.assert_array_equal(counts,[4,4])


def test_training_structures_and_group_validation_match_config():
    expected=['EMD33026','EMD70734','EMD22689','EMD20841','EMD3225']
    assert list(STRUCTURES.values())==expected
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/joint_training_record.json').read_text(encoding='utf-8'))
    assert config['train_structures']==expected
    assert config['validation_fraction']==.10
    assert config['input_files']==['multistructure_v2_train.npz']
    assert 'unseen_validation_structure' not in config


def test_validation_score_averages_only_held_out_group_metrics():
    metrics=[dict(macro_precision10_at8deg=.3),dict(macro_precision10_at8deg=.7)]
    assert validation_score(metrics)==pytest.approx(.5)


def test_each_structure_holds_out_entire_view_groups():
    class Dataset:
        structure_ids=np.repeat(np.arange(5),400)
        group_ids=np.tile(np.repeat(np.arange(100),4),5)
        def __len__(self):return len(self.structure_ids)
    data=Dataset();train,validation=split_groups(data,20261014)
    assert len(train)==1800 and len(validation)==200
    for sid in range(5):
        a=data.group_ids[train[data.structure_ids[train]==sid]]
        b=data.group_ids[validation[data.structure_ids[validation]==sid]]
        assert len(np.unique(b))==10
        assert not set(a)&set(b)
    np.testing.assert_array_equal(np.sort(np.concatenate([train,validation])),np.arange(len(data)))


