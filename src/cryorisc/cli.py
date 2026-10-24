import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(prog='cryorisc')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('train',add_help=False,help='Joint encoder/adapter training; use train --help')
    sub.add_parser('train-demo',add_help=False,help='Tiny supervised training example; use train-demo --help')
    sub.add_parser('demo',add_help=False,help='32-particle tiny workflow; use demo --help')
    p=sub.add_parser('classify',help='Native-particle 2D classification')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--profile',choices=['10028','10219','65045'],required=True)
    p.add_argument('--encoder',type=Path)
    p.add_argument('--adapter',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--arm',choices=['full','no-network','no-candidate-rescue'],default='full')
    p.add_argument('--feature-batch',type=int,default=16)
    p.add_argument('--seed',type=int,default=20261012)
    p.add_argument('--apix',type=float)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--features-only',action='store_true')
    p.add_argument('--allow-cpu',action='store_true')
    p=sub.add_parser('iac',help='Evaluate final hard labels against known simulation directions')
    p.add_argument('--directions',type=Path,required=True,help='NPY (N,3) or NPZ key viewdirs')
    p.add_argument('--labels',type=Path,required=True,help='NPY (N,), or final_membership_and_poses.npz')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--thresholds',type=float,nargs='+',default=[5,8,10,15])
    p=sub.add_parser('inspect',help='Print an MRCS stack shape without loading the stack')
    p.add_argument('path',type=Path)

    if len(sys.argv)>1 and sys.argv[1] in ('train','train-demo','demo'):
        command=sys.argv[1]
        sys.argv=[sys.argv[0]]+sys.argv[2:]
        if command=='train':
            from .train import main as command_main
        elif command=='train-demo':
            from .train_demo import main as command_main
        else:
            from .demo import main as command_main
        return command_main()
    args=parser.parse_args()
    if args.command=='classify':
        from .pipeline import classify
        from .profiles import get_profile
        if args.feature_batch<1:parser.error('--feature-batch must be positive')
        return classify(args,get_profile(args.profile))
    if args.command=='inspect':
        from .io import read_mrc_memmap
        print(json.dumps(dict(path=str(args.path),shape=list(read_mrc_memmap(args.path).shape))))
    elif args.command=='iac':
        import numpy as np
        from .metrics import intra_class_angular_consistency
        v=np.load(args.directions,allow_pickle=False)
        if isinstance(v,np.lib.npyio.NpzFile):
            with v:directions=v['viewdirs']
        else:directions=v
        y=np.load(args.labels,allow_pickle=False)
        if isinstance(y,np.lib.npyio.NpzFile):
            with y:labels=y['classes'][np.arange(len(y['classes'])),y['weights'].argmax(1)]
        else:labels=y
        result=intra_class_angular_consistency(directions,labels,args.thresholds)
        result.pop('per_particle')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
