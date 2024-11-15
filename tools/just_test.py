import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import register, GLOBAL_CONFIG


name = 'Resize'
wrapper = 'mot_transform_wrap'
args = {
    'size': [640, 640],
    'antialias': True
}
transform_class = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)
wrapper_func = GLOBAL_CONFIG[wrapper]
transform = wrapper_func(transform_class, name, **args)

import pdb; pdb.set_trace()