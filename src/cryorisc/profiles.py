from copy import deepcopy

PROFILES = {
    '10028': dict(size=180,apix=2.68,expected_particles=10000,
        scales=[1.,1.2,1.4],background_radius=75.,cluster_seed=20260809,
        radius=65.,edge=10.,shift=24,historical=True,fine_iterations=3,
        coarse_cutoff=.20,fine_cutoff=.22,coarse_batch=8,union=True,
        final_band_profile='10028'),
    '10219': dict(size=160,apix=1.71,expected_particles=20000,
        scales=[.9,1.,1.1],background_radius=75.,cluster_seed=20260912,
        radius=58.,edge=9.,shift=21,historical=False,fine_iterations=3,
        coarse_cutoff=.16,fine_cutoff=.18,coarse_batch=2,union=True,
        final_band_profile='10219'),
    '65045': dict(size=128,apix=2.8125,expected_particles=5000,
        scales=[1.,1.2,1.4],background_radius=75.*128/180,cluster_seed=20260809,
        radius=65.*128/180,edge=10.*128/180,shift=round(24*128/180),
        historical=False,fine_iterations=3,coarse_cutoff=.20,fine_cutoff=.22,
        coarse_batch=2,union=True,final_band_profile='10028'),
}

def get_profile(name):
    return deepcopy(PROFILES[name])
