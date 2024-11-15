import torch 
import torch.utils.data as data

from src.core import register
from src.misc.instances import Instances, BatchInstances

__all__ = ['DataLoader']


@register
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn', 'sampler']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string


@register
def default_collate_fn(items):
    '''default collate_fn
    '''    
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]

@register
def mot_collate_fn(items):
    '''collate_fn for end-to-end mot
    '''    
    # assert len(items) == 1, 'Currently, only support batch size = 1, but find batch size = {}'.format(len(items))
    # images, targets = items[0]
    # images = [img.unsqueeze(dim=0) for img in images]
    images = [x[0] for x in items]
    images = list(zip(*images))
    images = [torch.stack(img) for img in images]
    
    target_instances = [x[1] for x in items]
    target_instances = list(zip(*target_instances))
    target_instances = [BatchInstances.stack(t) for t in target_instances]
    return images, target_instances