"""by lyuwenyu
"""

import torch 
import torch.nn as nn

import re 
import copy
import os
from src.misc import dist
from .config import BaseConfig
from .yaml_utils import load_config, merge_config, create, merge_dict, save_config, merge_opts_to_config, write_args


class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, opts=None, run_type='train', auto_resume=False, **kwargs) -> None:
        super().__init__()

        cfg = load_config(cfg_path)
        #import pdb;pdb.set_trace()  #debug
        merge_dict(cfg, kwargs)
        merge_opts_to_config(cfg, opts)
        
        # pprint(cfg)
        self.cfg_path = cfg_path
        self.yaml_cfg = cfg 

        self.log_step = cfg.get('log_step', 100)
        self.checkpoint_step = cfg.get('checkpoint_step', 1)
        self.epoches = cfg.get('epoches', -1)
        self.resume = cfg.get('resume', '')
        self.tuning = cfg.get('tuning', '')
        # assert not all([self.tuning, self.resume]), 'Only support from_scrach or resume or tuning at one time'

        self.sync_bn = cfg.get('sync_bn', False)
        self.output_dir = cfg.get('output_dir', None)
        # import pdb; pdb.set_trace()
        if self.output_dir is None:
            if cfg_path.endswith('config.yaml') and 'outputs' in cfg_path:
                self.output_dir = os.path.dirname(cfg_path)
            else:
                if run_type in ['train']:
                    start_index = cfg_path.find('configs'+os.path.sep) + len('configs'+os.path.sep)
                    self.output_dir = os.path.join('output', cfg_path[start_index:].replace('.yml', '_g{}'.format(dist.get_world_size())).replace('.yaml', '_g{}'.format(dist.get_world_size())))
                elif run_type in ['eval', 'track']:
                    ckpt_path = self.resume if self.resume else self.tuning
                    self.output_dir = os.path.dirname(ckpt_path)
                else:
                    raise RuntimeError

        if auto_resume:
            ckpt_path = os.path.join(self.output_dir, 'checkpoint.pth')
            if os.path.exists(ckpt_path):
                self.resume = ckpt_path

        self.use_ema = cfg.get('use_ema', False)
        self.use_amp = cfg.get('use_amp', False)
        self.autocast = cfg.get('autocast', dict())
        self.find_unused_parameters = cfg.get('find_unused_parameters', None)
        self.clip_max_norm = cfg.get('clip_max_norm', 0.)
        
        # save configs
        if dist.is_main_process() and run_type in ['train']:
            cfg_save_path = os.path.join(self.output_dir, 'config.yml')
            os.makedirs(os.path.dirname(cfg_save_path), exist_ok=True)
            save_config(cfg, save_path=cfg_save_path, verbose=True)

            arg_save_path = os.path.join(self.output_dir, 'args.txt')
            write_args(cfg, save_path=arg_save_path)

            # save args
            

    @property
    def model(self, ) -> torch.nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            #import pdb; pdb.set_trace()
            merge_config(self.yaml_cfg)
            # import pdb; pdb.set_trace()
            self._model = create(self.yaml_cfg['model'])
        return self._model 

    @property
    def postprocessor(self, ) -> torch.nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._postprocessor = create(self.yaml_cfg['postprocessor'])
        return self._postprocessor

    @property
    def criterion(self, ):
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._criterion = create(self.yaml_cfg['criterion'])
        return self._criterion

    
    @property
    def optimizer(self, ):
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            params = self.get_optim_params(self.yaml_cfg['optimizer'], self.model)
            #import pdb;pdb.set_trace()
            self._optimizer = create('optimizer', params=params)

        return self._optimizer
    
    @property
    def lr_scheduler(self, ):
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._lr_scheduler = create('lr_scheduler', optimizer=self.optimizer)
            print('Initial lr: ', self._lr_scheduler.get_last_lr())

        return self._lr_scheduler
    
    @property
    def train_dataloader(self, ):
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._train_dataloader = create('train_dataloader')
            self._train_dataloader.shuffle = self.yaml_cfg['train_dataloader'].get('shuffle', False)

        return self._train_dataloader
    
    @property
    def val_dataloader(self, ):
        # import pdb; pdb.set_trace()
        if self.yaml_cfg.get('eval_data', 'val') == 'val':
            if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:
                merge_config(self.yaml_cfg)
                self._val_dataloader = create('val_dataloader')
                self._val_dataloader.shuffle = self.yaml_cfg['val_dataloader'].get('shuffle', False)
        elif self.yaml_cfg.get('eval_data', 'val') == 'test':
            if self._val_dataloader is None and 'test_dataloader' in self.yaml_cfg:
                merge_config(self.yaml_cfg)
                self._val_dataloader = create('test_dataloader')
                self._val_dataloader.shuffle = self.yaml_cfg['test_dataloader'].get('shuffle', False)
        elif self.yaml_cfg.get('eval_data', 'val') == 'real':
            if self._val_dataloader is None and 'real_dataloader' in self.yaml_cfg:
                merge_config(self.yaml_cfg)
                self._val_dataloader = create('real_dataloader')
                self._val_dataloader.shuffle = self.yaml_cfg['real_dataloader'].get('shuffle', False)
        elif self.yaml_cfg.get('eval_data', 'val') == 'eval_train':
            if self._val_dataloader is None and 'eval_train_dataloader' in self.yaml_cfg:
                merge_config(self.yaml_cfg)
                self._val_dataloader = create('eval_train_dataloader')
                self._val_dataloader.shuffle = self.yaml_cfg['eval_train_dataloader'].get('shuffle', False)
        elif self.yaml_cfg.get('eval_data', 'val') == 'train':
            if self._val_dataloader is None and 'train_dataloader' in self.yaml_cfg:
                merge_config(self.yaml_cfg)
                self._val_dataloader = create('train_dataloader')
                self._val_dataloader.shuffle = self.yaml_cfg['train_dataloader'].get('shuffle', False)
        else:
            raise NotImplementedError('Unknonw type of eval data: {}'.format(self.yaml_cfg['eval_data']))
        return self._val_dataloader
    
    @property
    def ema(self, ):
        if self._ema is None and self.yaml_cfg.get('use_ema', False):
            merge_config(self.yaml_cfg)
            self._ema = create('ema', model=self.model)
            
        return self._ema
    

    @property
    def scaler(self, ):
        if self._scaler is None and self.yaml_cfg.get('use_amp', False):
            merge_config(self.yaml_cfg)
            self._scaler = create('scaler')

        return self._scaler

 
    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        '''
        E.g.:
            ^(?=.*a)(?=.*b).*$         means including a and b
            ^((?!b.)*a((?!b).)*$       means including a but not b
            ^((?!b|c).)*a((?!b|c).)*$  means including a but not (b | c)
        '''
        assert 'type' in cfg, ''
        cfg = copy.deepcopy(cfg)

        if 'params' not in cfg:
            return model.parameters() 

        assert isinstance(cfg['params'], list), ''

        param_groups = []
        visited = []
        for pg in cfg['params']:
            pattern = pg['params']
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
            pg['params'] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))

        names = [k for k, v in model.named_parameters() if v.requires_grad]
        #sumparam = sum(p.numel() for p in model.backbone.parameters())
        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({'params': params.values()})
            visited.extend(list(params.keys()))
        import pdb;pdb.set_trace()
        assert len(visited) == len(names), ''

        return param_groups

    @property
    def track_dataset(self, ): # dataset for online tracking, rather than a dataloader
        if self.yaml_cfg.get('eval_data', 'val') == 'val':
            if self._track_dataset is None and 'val_dataset' in self.yaml_cfg:
                merge_config(self.yaml_cfg)
                self._track_dataset = create('val_dataset')
        elif self.yaml_cfg.get('eval_data', 'val') == 'test':
            if self._track_dataset is None and 'test_dataset' in self.yaml_cfg:
                merge_config(self.yaml_cfg)
                self._track_dataset = create('test_dataset')
        elif self.yaml_cfg.get('eval_data', 'val') == 'real':
            if self._track_dataset is None and 'real_dataset' in self.yaml_cfg:
                merge_config(self.yaml_cfg)
                self._track_dataset = create('real_dataset')
        elif self.yaml_cfg.get('eval_data', 'val') == 'eval_train':
            if self._track_dataset is None and 'eval_train_dataset' in self.yaml_cfg:
                merge_config(self.yaml_cfg)
                self._track_dataset = create('eval_train_dataset')
        else:
            raise NotImplementedError('Unknonw type of eval data: {}'.format(self.yaml_cfg['eval_data']))
        return self._track_dataset

    @property
    def tracker(self,):
        if self._tracker is None and 'tracker' in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._tracker = create('tracker')
        return self._tracker 