# ------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import itertools
from typing import Any, Dict, List, Tuple, Union
import torch
import numpy as np


class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = ...
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(
            self, 
            image_size: Tuple[int, int], 
            orig_image_size: Tuple[int, int],
            image_id: int, # image id in coco
            **kwargs: Any):
        """
        Args:
            image_size (width, height): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size # (w, h)
        self._orig_image_size = orig_image_size # (w, h)
        self._image_id = image_id
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size
    
    @property
    def orig_image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._orig_image_size
    
    @property
    def image_id(self) -> int:
        """
        Returns:
            int: image id in coco
        """
        return self._image_id


    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding the field {} of length {} to a Instances of length {}".format(name, data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(image_size=self._image_size, orig_image_size=self._orig_image_size, image_id=self._image_id)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def numpy(self):
        ret = Instances(image_size=self._image_size, orig_image_size=self._orig_image_size, image_id=self._image_id)
        for k, v in self._fields.items():
            if hasattr(v, "numpy"):
                v = v.numpy()
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(image_size=self._image_size, orig_image_size=self._orig_image_size, image_id=self._image_id)

        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            if torch.is_tensor(v): # it may be a string or None
                return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        orig_image_size = instance_lists[0].orig_image_size
        image_id = instance_lists[0].image_id
        for i in instance_lists[1:]:
            assert i.orig_image_size == orig_image_size
            assert i.image_id == image_id
        ret = Instances(image_size=image_size, orig_image_size=orig_image_size, image_id=image_id)

        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "image_id={}, ".format(self._image_id)
        s += "num_instances={}, ".format(len(self))
        s += "image_wh={}, ".format(self._image_size)
        s += "orig_image_wh={}, ".format(self._orig_image_size)
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__


class BatchInstances(Instances):
    def __init__(
            self, 
            image_size: List[Tuple[int, int]], 
            orig_image_size: List[Tuple[int, int]],
            image_id: List[int], # image id in coco
            **kwargs: Any):
        """
        Args:
            image_size (width, height): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size # (w, h)
        self._orig_image_size = orig_image_size # (w, h)
        self._image_id = image_id
        self._fields: Dict[str, Any] = {}
        
        ks = list(kwargs.keys())
        for i in range(len(ks)):
            k = ks[i]
            v = kwargs[k]
            self.set(k, v)

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = BatchInstances(image_size=self._image_size, orig_image_size=self._orig_image_size, image_id=self._image_id)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret


    # def numpy(self):
    #     ret = BatchInstances(image_size=self._image_size, orig_image_size=self._orig_image_size, image_id=self._image_id)
    #     for k, v in self._fields.items():
    #         if hasattr(v, "numpy"):
    #             v = v.numpy()
    #         ret.set(k, v)
    #     return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = BatchInstances(image_size=self._image_size, orig_image_size=self._orig_image_size, image_id=self._image_id)

        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret


    @staticmethod
    def cat(instance_lists: List["BatchInstances"]) -> "BatchInstances":
        """
            cat instances along the 1-st dim. 0-th dim is the batch dim
        Args:
            instance_lists (list[BatchInstances])

        Returns:
            BatchInstances
        """
        assert all(isinstance(i, BatchInstances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        orig_image_size = instance_lists[0].orig_image_size
        image_id = instance_lists[0].image_id
        for i in instance_lists[1:]:
            assert tuple(i.orig_image_size) == tuple(orig_image_size)
            assert tuple(i.image_id) == tuple(image_id)
        ret = BatchInstances(image_size=image_size, orig_image_size=orig_image_size, image_id=image_id)

        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=1)
            # elif isinstance(v0, list):
            #     values = list(itertools.chain(*values))
            # elif hasattr(type(v0), "cat"):
            #     values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret
    

    @staticmethod
    def remove_invalid(instances: "BatchInstances") -> "BatchInstances":
        """
            remove invalid instance
        Args:
            instances: BatchInstances

        Returns:
            BatchInstances
        """
        valid_mask = instances.valid_mask  # b, n
        max_num = valid_mask.sum(-1).max()

        if max_num == valid_mask.shape[-1]:
            return instances

        image_size = instances.image_size
        orig_image_size = instances.orig_image_size
        image_id = instances.image_id
        ret = BatchInstances(image_size=image_size, orig_image_size=orig_image_size, image_id=image_id)
        
        bsz = valid_mask.shape[0]
        index = []
        valid_index = valid_mask.nonzero()
        invalid_index = (~valid_mask).nonzero()
        valid_mask_new = torch.full((bsz,max_num,), False, dtype=torch.bool)
        for b in range(bsz):
            index_ = torch.cat((valid_index[valid_index[:,0]==b], invalid_index[invalid_index[:,0]==b]), dim=0)[:max_num] # max_num, 2
            index.append(index_)
            valid_mask_new[b,:valid_mask[b].sum()] = True
        index = torch.cat(index, dim=0) # b*max_num, 2

        for k in instances._fields.keys():
            v_old = instances.get(k)
            if isinstance(v_old, torch.Tensor):
                # import pdb; pdb.set_trace()
                v_new = v_old[index[:,0], index[:,1]].reshape(bsz, max_num, *list(v_old.shape[2:]))
            else:
                raise NotImplementedError('Unknown type of data: {}'.format(type(v_old)))
            ret.set(k, v_new)
        
        # check if the results are right
        assert (valid_mask_new.to(ret.valid_mask) != ret.valid_mask).sum() == 0, 'The results are wrong, please check this function!'

        return ret


    @staticmethod
    def stack(instance_lists: List["Instances"]) -> "BatchInstances":
        """
            create a batch dim. 
        Args:
            instance_lists (list[Instances]). List of gt Instances

        Returns:
            BatchInstances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0

        image_size = []
        orig_image_size = []
        image_id = []
        valid_length = []
        for i in instance_lists:
            image_size.append(i.image_size)
            orig_image_size.append(i.orig_image_size)
            image_id.append(i.image_id)
            valid_length.append(len(i))

        max_length = max(valid_length)
        device = None
        ret = BatchInstances(image_size=image_size, orig_image_size=orig_image_size, image_id=image_id)

        for k in instance_lists[0]._fields.keys():
            v0 = instance_lists[0].get(k)
            if isinstance(v0, torch.Tensor):
                if device is None:
                    device = v0.device
                shape = list(v0.shape) 
                values = []
                for ins in instance_lists:
                    v = ins.get(k)
                    pad = torch.zeros([max_length-len(v)]+shape[1:]).to(v)
                    values.append(torch.cat((v, pad), dim=0))
                values = torch.stack(values, dim=0)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        
        valid_mask = torch.full((len(instance_lists), max_length,), False, dtype=torch.bool, device=device)
        for b, vl in enumerate(valid_length):
            valid_mask[b, :vl] = True
        ret.valid_mask = valid_mask 
        
        return ret

    