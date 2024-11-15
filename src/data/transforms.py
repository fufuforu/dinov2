""""by lyuwenyu
"""


import torch 
import torch.nn as nn 

import torchvision
# torchvision.disable_beta_transforms_warning()
from torchvision import datapoints
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2.utils import query_bounding_box, query_spatial_size

from PIL import Image 
from typing import Any, Dict, List, Optional, Callable, Union, cast
from torch.utils._pytree import tree_flatten, tree_unflatten
from src.core import register, GLOBAL_CONFIG
import copy

__all__ = ['Compose', ]


RandomPhotometricDistort = register(T.RandomPhotometricDistort)
# RandomZoomOut = register(T.RandomZoomOut)
# RandomIoUCrop = register(T.RandomIoUCrop)
# RandomHorizontalFlip = register(T.RandomHorizontalFlip)
Resize = register(T.Resize)
ToImageTensor = register(T.ToImageTensor)
ConvertDtype = register(T.ConvertDtype)
# SanitizeBoundingBox = register(T.SanitizeBoundingBox)
# RandomCrop = register(T.RandomCrop)
Normalize = register(T.Normalize)
# RandomPerspective = register(T.RandomPerspective) 
RandomAffine = register(T.RandomAffine)
# RandomChoice = register(T.RandomChoice)

@register
class Compose(T.Compose):
    def __init__(self, ops) -> None:
        def make_transform_list(ops_):
            transforms = []
            if ops_ is not None:
                for op in ops_:
                    op_ = copy.deepcopy(op)
                    if isinstance(op_, dict):
                        name = op_.pop('type')
                        if name == 'RandomChoice':
                            p = op_.get('p', None)
                            transform = T.RandomChoice(make_transform_list(op_['ops']), p=p)
                        else:
                            transform_class = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)
                            if 'wrap_func' in op_:
                                wrap_func_name = op_.pop('wrap_func')
                                wrap_func = GLOBAL_CONFIG[wrap_func_name]
                                transform = wrap_func(transform_class, **op_)
                            else:
                                transform = transform_class(**op_)
                        transforms.append(transform)
                        # op_['type'] = name
                    elif isinstance(op_, nn.Module):
                        transforms.append(op_)
                    else:
                        raise ValueError('')
            else:
                transforms =[EmptyTransform(), ]
            
            return transforms

        transforms = make_transform_list(ops)
 
        super().__init__(transforms=transforms)


@register
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register
class PadToSize(T.Pad):
    _transformed_types = (
        Image.Image,
        datapoints.Image,
        datapoints.Video,
        datapoints.Mask,
        datapoints.BoundingBox,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sz = F.get_spatial_size(flat_inputs[0])
        h, w = self.spatial_size[0] - sz[0], self.spatial_size[1] - sz[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        
        self.spatial_size = spatial_size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    # def __call__(self, *inputs: Any) -> Any:
    def forward(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


# @register
# class RandomIoUCrop(T.RandomIoUCrop):
#     def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
#         super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
#         self.p = p 

#     # def __call__(self, *inputs: Any) -> Any:
#     def forward(self, *inputs: Any) -> Any:
#         if torch.rand(1) >= self.p:
#             return inputs if len(inputs) > 1 else inputs[0]

#         return super().forward(*inputs)


# #########################################################################
# The following transforms are written for MOT
# #########################################################################

@register
class RandomZoomOut(T.RandomZoomOut):

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_h, orig_w = query_spatial_size(flat_inputs)

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)
        padding = [left, top, right, bottom]

        params = {
            'padding': padding,
            'needs_pad': torch.rand(1) >= self.p
        }

        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:

        if not params['needs_pad']:
            return inpt
        else:
            params_ = copy.deepcopy(params)
            params_.pop('needs_pad')
            return super()._transform(inpt, params_)

    def forward(self, *inputs: Any) -> Any:
        # We need to almost duplicate `Transform.forward()` here since we always want to check the inputs, but return
        # early afterwards in case the random check triggers. The same result could be achieved by calling
        # `super().forward()` after the random check, but that would call `self._check_inputs` twice.

        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)

        self._check_inputs(flat_inputs)

        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
        )

        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)


@register
class RandomHorizontalFlip(T.RandomHorizontalFlip):
    
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        params = {}
        params['needs_flip'] = torch.rand(1) >= self.p
        return params
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:

        if not params['needs_flip']:
            return inpt
        else:
            return super()._transform(inpt, params)

    def forward(self, *inputs: Any) -> Any:
        # We need to almost duplicate `Transform.forward()` here since we always want to check the inputs, but return
        # early afterwards in case the random check triggers. The same result could be achieved by calling
        # `super().forward()` after the random check, but that would call `self._check_inputs` twice.

        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)

        self._check_inputs(flat_inputs)

        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
        )

        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)



@register
class RandomPerspective(T.RandomPerspective):
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        params = super()._get_params(flat_inputs)
        params['needs_perspective'] = torch.rand(1) >= self.p
        return params
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:

        if not params['needs_perspective']:
            return inpt
        else:
            params_ = copy.deepcopy(params)
            params_.pop('needs_perspective')
            return super()._transform(inpt, params_)

    def forward(self, *inputs: Any) -> Any:
        # We need to almost duplicate `Transform.forward()` here since we always want to check the inputs, but return
        # early afterwards in case the random check triggers. The same result could be achieved by calling
        # `super().forward()` after the random check, but that would call `self._check_inputs` twice.

        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)

        self._check_inputs(flat_inputs)

        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
        )

        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)


@register
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        params = {}
        params['needs_crop'] = torch.rand(1) >= self.p # RandomCrop also has the key 'needs_crop'
        if params['needs_crop']:
            p = super()._get_params(flat_inputs)
            params.update(p)
            if len(p) == 0: # all trials are failed
                params['needs_crop'] = False

        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:

        if not params['needs_crop']:
            return inpt
        else:
            return super()._transform(inpt, params)


@register
class SanitizeBoundingBox(T.SanitizeBoundingBox):
    __inject__ = ['labels_getter']
    """
    This class is compatible with T.SanitizeBoundingBox. But it support a callable labels_getter,
    which return a list/tuple of multiple tensors. The labels should be returned.

    In fact, this class is a combination of SanitizeBoundingBox in torchvision v1.5 and  SanitizeBoundingBoxes in torchvision v1.8
    """
 
    def __init__(self, **args):
        if 'labels_getter' in args:
            if isinstance(args['labels_getter'], (str, )) and args['labels_getter'] in GLOBAL_CONFIG:
                assert callable(GLOBAL_CONFIG[args['labels_getter']]), 'The given labels_getter: {} is not callable!'.format(args['labels_getter'])
                args['labels_getter'] = GLOBAL_CONFIG[args['labels_getter']]
        super().__init__(**args)


    def forward(self, *inputs: Any) -> Any:
        inputs = inputs if len(inputs) > 1 else inputs[0]

        if self._labels_getter is None:
            labels = None
        else:
            
            labels = self._labels_getter(inputs)
        
        # import pdb; pdb.set_trace()
        if labels is not None:
            msg = "The labels in the input to forward() must be a tensor or None, got {type} instead."
            if isinstance(labels, torch.Tensor):
                labels = (labels,)
            elif isinstance(labels, (tuple, list)):
                for entry in labels:
                    if not isinstance(entry, torch.Tensor):
                        # TODO: we don't need to enforce tensors, just that entries are indexable as t[bool_mask]
                        raise ValueError(msg.format(type=type(entry)))
            else:
                raise ValueError(msg.format(type=type(labels)))

        flat_inputs, spec = tree_flatten(inputs)
        # TODO: this enforces one single BoundingBox entry.
        # Assuming this transform needs to be called at the end of *any* pipeline that has bboxes...
        # should we just enforce it for all transforms?? What are the benefits of *not* enforcing this?
        boxes = query_bounding_box(flat_inputs)

        if boxes.ndim != 2:
            raise ValueError(f"boxes must be of shape (num_boxes, 4), got {boxes.shape}")

        if labels is not None:
            for label in labels:
                if boxes.shape[0] != label.shape[0]:
                    raise ValueError(
                        f"Number of boxes (shape={boxes.shape}) and must match the number of labels."
                        f"Found labels with shape={label.shape})."
                    )

        boxes = cast(
            datapoints.BoundingBox,
            F.convert_format_bounding_box(
                boxes,
                new_format=datapoints.BoundingBoxFormat.XYXY,
            ),
        )
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        valid = (ws >= self.min_size) & (hs >= self.min_size) & (boxes >= 0).all(dim=-1)
        # TODO: Do we really need to check for out of bounds here? All
        # transforms should be clamping anyway, so this should never happen?
        image_h, image_w = boxes.spatial_size
        valid &= (boxes[:, 0] <= image_w) & (boxes[:, 2] <= image_w)
        valid &= (boxes[:, 1] <= image_h) & (boxes[:, 3] <= image_h)

        params = dict(valid=valid, labels=labels)  
        flat_outputs = [
            # Even-though it may look like we're transforming all inputs, we don't:
            # _transform() will only care about BoundingBoxes and the labels
            self._transform(inpt, params)
            for inpt in flat_inputs
        ]
        outputs = tree_unflatten(flat_outputs, spec)

        return outputs

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        is_label = params["labels"] is not None and any(inpt is label for label in params["labels"])
        is_bounding_box_or_mask = isinstance(inpt, (datapoints.BoundingBox, datapoints.Mask))

        if not (is_label or is_bounding_box_or_mask):
            return inpt

        output = inpt[params["valid"]]

        if is_label:
            return output

        return type(inpt).wrap_like(inpt, output)



@register
class ConvertBox(T.Transform):
    _transformed_types = (
        datapoints.BoundingBox,
    )
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': datapoints.BoundingBoxFormat.XYXY,
            'cxcywh': datapoints.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        if self.out_fmt:
            spatial_size = inpt.spatial_size
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            inpt = datapoints.BoundingBox(inpt, format=self.data_fmt[self.out_fmt], spatial_size=spatial_size)
        
        if self.normalize:
            inpt = inpt / torch.tensor(inpt.spatial_size[::-1]).tile(2)[None]

        return inpt


@register
def mot_transform_wrap(transform_class, **args):
    class MOTTransformWrapper(transform_class):
        def __init__(self, **args):
            super().__init__(**args)
            self.__class__.__name__ = 'MOT_{}'.format(transform_class.__name__) #TODO: '{}_{}'.format('MOTTransformWrapper', transform_class.__name__)
            self.parent_class_name = transform_class.__name__
            self._params_ = None
        
        def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
            
            if self._params_ is None:
                self._params_ = super()._get_params(flat_inputs)

            params = copy.deepcopy(self._params_)

            if self.parent_class_name in ['RandomIoUCrop']:
                # import pdb; pdb.set_trace()
                if params['needs_crop']: # we need to check if one box is within the image
                    bboxes = flat_inputs[1]
                    assert isinstance(bboxes, (datapoints.BoundingBox,)), 'Not Boxes!'
                    xyxy_bboxes = F.convert_format_bounding_box(
                                    bboxes.as_subclass(torch.Tensor), bboxes.format, datapoints.BoundingBoxFormat.XYXY
                                )
                    cx = 0.5 * (xyxy_bboxes[..., 0] + xyxy_bboxes[..., 2])
                    cy = 0.5 * (xyxy_bboxes[..., 1] + xyxy_bboxes[..., 3])
                    left, new_h, top, new_w = params['left'], params['height'], params['top'], params['width'], 
                    right = left + new_w
                    bottom = top + new_h
                    is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                    params['is_within_crop_area'] = is_within_crop_area
                    
            return params
        
        def forward(self, *inputs: Any) -> Any:

            # if self.parent_class_name in ['SanitizeBoundingBox']:
                # import pdb; pdb.set_trace()
            self._params_ = None
            inputs = inputs if len(inputs) > 1 else inputs[0]
            outputs = []
            for inp in inputs:
                output = super().forward(inp)
                outputs.append(output)

            return outputs

    return MOTTransformWrapper(**args)
