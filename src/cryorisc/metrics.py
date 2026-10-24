from __future__ import annotations
import numpy as np


def intra_class_angular_consistency(viewdirs,labels,thresholds=(5.,8.,10.,15.),block_size=512):
    vectors=np.asarray(viewdirs,dtype=np.float64)
    labels=np.asarray(labels)
    if vectors.ndim!=2 or vectors.shape[1]!=3 or labels.shape!=(len(vectors),) or not len(vectors):
        raise ValueError('Expected nonempty (N,3) directions and (N,) labels')
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors,axis=1)==0):
        raise ValueError('Directions must be finite and nonzero')
    if not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0):
        raise ValueError('Every particle must have one nonnegative integer final label')
    angles=np.asarray(thresholds,dtype=np.float64)
    if angles.ndim!=1 or np.any((angles<0)|(angles>90)) or not np.isfinite(angles).all():
        raise ValueError('Axial angular thresholds must lie in [0,90] degrees')
    if block_size<1:raise ValueError('block_size must be positive')
    vectors=vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
    per_particle=np.zeros((len(labels),len(angles)),dtype=np.float64)
    singleton_particles=0
    for label in np.unique(labels):
        ids=np.flatnonzero(labels==label);n=len(ids)
        if n==1:
            singleton_particles+=1;continue
        for start in range(0,n,block_size):
            take=ids[start:start+block_size]
            distance=np.rad2deg(np.arccos(np.clip(np.abs(vectors[take]@vectors[ids].T),0,1)))
            for t,threshold in enumerate(angles):
                accepted=distance<=threshold
                accepted[np.arange(len(take)),np.arange(start,start+len(take))]=False
                per_particle[take,t]=accepted.sum(1)/(n-1)
    return dict(particles=len(labels),active_classes=len(np.unique(labels)),
        singleton_particles=singleton_particles,denominator=len(labels),
        singleton_contribution=0,axial_distance=True,
        iac_percent={f'{t:g}':float(v) for t,v in zip(angles,per_particle.mean(0)*100)},
        per_particle=per_particle)
