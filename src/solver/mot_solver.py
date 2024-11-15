'''
by lyuwenyu
'''
import time 
import json
import datetime
import os

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset
from pathlib import Path 
from .solver import BaseSolver
from .mot_engine import train_one_epoch, evaluate_det


class MOTSolver(BaseSolver):
    
    def setup(self, load_tuning_last_epoch=False):
        '''Avoid instantiating unnecessary classes 
        '''
        cfg = self.cfg
        device = cfg.device
        self.device = device
        self.last_epoch = cfg.last_epoch
        # import pdb; pdb.set_trace()
        self.model = dist.warp_model(cfg.model.to(device), cfg.find_unused_parameters, cfg.sync_bn)
        
        if dist.is_parallel(self.model):
            self.criterion = self.model.module.criterion
        else:
            self.criterion = self.model.criterion
        
        self.postprocessor = cfg.postprocessor

        # NOTE (lvwenyu): should load_tuning_state before ema instance building
        if self.cfg.tuning:
            print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning, load_last_epoch=load_tuning_last_epoch)

        self.scaler = cfg.scaler
        self.ema = cfg.ema.to(device) if cfg.ema is not None else None 

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):

            self.train_dataloader.dataset.set_epoch(epoch, overlap=True)
            self.val_dataloader.dataset.set_epoch(epoch, overlap=False)

            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
                self.val_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate_det(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # TODO 
            for k in test_stats.keys():
                if k in ['result_dict']:
                    continue
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

            print('best_stat: ', best_stat)


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()
        eval_data = self.cfg.yaml_cfg['eval_data']
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        self.val_dataloader.dataset.set_epoch('evaluation', overlap=False)
        if dist.is_dist_available_and_initialized():
            self.val_dataloader.sampler.set_epoch(0)

        test_stats, coco_evaluator = evaluate_det(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            # dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval_{}_bbox_e{}.pth".format(eval_data, self.last_epoch))
            txt_path = self.output_dir / "eval_{}_e{}.txt".format(eval_data, self.last_epoch)
            if os.path.exists(txt_path):
                os.remove(txt_path)
            for k, res in test_stats['result_dict'].items():
                with open(txt_path, 'a') as f:
                    f.write('\n'+k+'\n')
                    f.write(res)
            print('saved to {}'.format(txt_path))
        return

    
    def track(self, ):
        self.eval_track()
        tracker = self.cfg.tracker 
        kargs = {
            'model':  self.ema.module if self.ema else self.model,
            'dataset': self.track_dataset,
            'epoch': self.last_epoch,
            'output_dir': self.output_dir,
            'cfg': self.cfg.yaml_cfg # used to save the runtime confg
        }
        tracker.prepare(**kargs)
        tracker.track()
        
