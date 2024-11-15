"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        opts=args.opts,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning,
        save=not args.run_type,
        eval_data=args.eval_data
    )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.run_type == 'eval':
        solver.val()
    elif args.run_type == 'train':
        solver.fit()
    elif args.run_type == 'track':
        solver.track()
    else:
        raise NotImplementedError('Unknown type of run type: {}'.format(args.run_type))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--run_type', default='train', choices=['train', 'eval', 'track'])
    parser.add_argument('--eval_data', default='val', choices=['val', 'test', 'real', 'eval_train', 'train'])
    parser.add_argument('--amp', action='store_true', default=False,)

    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    main(args)
