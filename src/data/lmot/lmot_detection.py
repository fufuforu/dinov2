"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import torch
import torch.utils.data
import os
import torchvision
# torchvision.disable_beta_transforms_warning()
from PIL import Image
import numpy as np
from torchvision import datapoints

from pycocotools import mask as coco_mask

from src.core import register
from .lmot_utils import RawImageReader

__all__ = ['LMOTDetection']


@register
class LMOTDetection(torchvision.datasets.CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_category']
    
    def __init__(self, img_folder, ann_file, transforms, return_masks, remap_category=False, clip_box=True, visibility_thr=0.2, image_n_bit=None):
        super(LMOTDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_category, clip_box=clip_box, visibility_thr=visibility_thr)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_category = remap_category
        self.image_n_bit = image_n_bit
        self.raw_image_reader = RawImageReader(
                                    is_dark='dark' in os.path.basename(ann_file) or 'real' in os.path.basename(ann_file),
                                    n_bit=image_n_bit)
    

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        path = os.path.join(self.root, path)
        if path.endswith('tiff'):
            black_level = int(self.coco.loadImgs(id)[0]["black_level"])
            image = self.raw_image_reader(path, black_level=black_level, numpy_or_tensor='tensor')
            image = datapoints.Image(image, dtype=torch.float32, requires_grad=False)
        else:
            image = Image.open(path).convert("RGB")
            image = datapoints.Image(image, dtype=torch.float32, requires_grad=False)
        return image

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))


    def _load_img_and_target(self, index: int):
        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        return image, target


    def __getitem__(self, idx):
        img, target = self._load_img_and_target(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # ['boxes', 'masks', 'labels']:
        spatial_size = img.shape[-2:] if torch.is_tensor(img) else img.size[::-1] # h, w
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'], 
                format=datapoints.BoundingBoxFormat.XYXY, 
                spatial_size=spatial_size) # h w

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])

        # import pdb; pdb.set_trace()
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # import pdb; pdb.set_trace()
        return img, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s 
    
    def __len__(self):
        return len(self.ids) #TODO:
        # return min(50, len(self.ids))


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_category=False, clip_box=True, visibility_thr=0.2):
        self.return_masks = return_masks
        self.remap_category = remap_category
        self.clip_box = clip_box
        self.visibility_thr = visibility_thr

    def __call__(self, image, target):
        if torch.is_tensor(image):
            h, w = image.shape[-2:]
        else:
            w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        
        # filter unvisible instances for detection
        anno = [obj for obj in anno if 'visibility' not in obj or obj['visibility'] > self.visibility_thr]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        if self.clip_box:
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_category:
            from ...data.category_map import CATEGORY2LABEL_DICT
            classes = [CATEGORY2LABEL_DICT[self.remap_category][obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
            
        classes = torch.tensor(classes, dtype=torch.int64)

        obj_ids = [obj["track_id"] for obj in anno]
        obj_ids = torch.tensor(obj_ids, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        obj_ids = obj_ids[keep]
        
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["obj_ids"] = obj_ids
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target
