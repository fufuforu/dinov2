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
from src.misc.instances import Instances
from src.misc.debugger import Debugger
from .lmot_utils import RawImageReader
from .lmot_detection import ConvertCocoPolysToMask
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import copy

# __all__ = ['LMOTTracking', 'labels_getter_func_for_mot_in_SanitizeBoundingBox']


class VideoCOCO(COCO):
    def __init__(self, annotation_file=None):
        super(VideoCOCO, self).__init__(annotation_file)
        
        # create index for videos
        videos = {}
        imgToVideo = {}
        imgToFrameId = {}
        videoToImages = {}
        for video in self.dataset['sequences']:
            videos[video['id']] = video
            videoToImages[video['id']] = video['image_ids']
        
        for img in self.dataset['images']:
            imgToVideo[img['id']] = img['sequence_id']
            imgToFrameId[img['id']] = img['frame_id']
        
        self.videos = videos
        self.imgToVideo = imgToVideo
        self.imgToFrameId = imgToFrameId
        self.videoToImages = videoToImages

@register
class LMOTTracking(Dataset):
    __inject__ = ['transforms']
    __share__ = ['remap_category']
    
    def __init__(
            self, 
            img_folder, 
            ann_file, 
            transforms, 
            return_masks, 
            remap_category=False, 
            clip_box=True, 
            visibility_thr=0.2, 
            image_n_bit=None, # the number of bits to load raw images
            
            # args for video clip
            num_frames=[2, 3, 4, 5], # number of frames in each clip
            num_frame_steps=[5, 9, 15], # the epoch to change number of frames
            frame_interval=10, # the interval between adjacent frames
            interval_mode='random_interval', # the mode to set the frame interval
        ):

        self.coco = VideoCOCO(ann_file)


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

        
        self.num_frames = num_frames
        self.num_frame_steps = num_frame_steps
        self.frame_interval = frame_interval
        self.interval_mode = interval_mode

        # check valid argments
        assert len(self.num_frames) == len(self.num_frame_steps) + 1, 'invlide number of steps!'
        if len(self.num_frame_steps) > 1:
            for i in range(len(self.num_frame_steps)-1):
                assert self.num_frame_steps[i] < self.num_frame_steps[i+1]

        self.current_epoch = -1
        self.current_num_frame = 0
        self.valid_img_ids = []
        self.set_epoch(epoch=0)


    def set_epoch(self, epoch, overlap=True):
        if epoch == 'evaluation':
            epoch = self.num_frame_steps[-1] + 1 # for evaluation, evaluate the longest sequances clip
        self.current_epoch = epoch
        if self.num_frame_steps is None or len(self.num_frame_steps) == 0:
            # fixed number of frames.
            return

        idx = 0
        for i in range(len(self.num_frame_steps)):
            if epoch >= self.num_frame_steps[i]:
                idx = i + 1
        self.current_num_frame = self.num_frames[idx]
        

        valid_img_ids = []
        # import pdb; pdb.set_trace()
        for video_id, video_img_ids in self.coco.videoToImages.items():
            end_idx = len(video_img_ids) - (self.current_num_frame - 1) * self.frame_interval
            if end_idx > 0:
                # valid_img_ids.extend(copy.deepcopy(video_img_ids[:end_idx]))
                for idx in range(end_idx):
                    if not overlap and idx % self.current_num_frame != 0:
                        continue
                    valid_img_ids.append(video_img_ids[idx])
        self.valid_img_ids = valid_img_ids
        print("set epoch: epoch {}, number of frames in clip: {}, number samples: {}".format(epoch, self.current_num_frame, len(self)))


    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    def _get_sample_range(self, start_idx):
        # take default sampling method for normal dataset.
        assert self.interval_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.interval_mode == 'fixed_interval':
            frame_interval = self.frame_interval
        elif self.interval_mode == 'random_interval':
            frame_interval = np.random.randint(1, self.frame_interval + 1)
        default_range = start_idx, start_idx + (self.current_num_frame - 1) * frame_interval + 1, frame_interval
        return default_range


    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        path = os.path.join(self.img_folder, path)
        if path.endswith('tiff'):
            black_level = int(self.coco.loadImgs(id)[0]["black_level"])
            image = self.raw_image_reader(path, black_level=black_level, numpy_or_tensor='tensor')
            image = datapoints.Image(image, dtype=torch.float32, requires_grad=False)
        else:
            image = Image.open(path).convert("RGB")
            image = datapoints.Image(image, dtype=torch.float32, requires_grad=False)
        return image # float tensor, 3 x H x W, in [0, 255]

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def _load_img_and_target(self, img_id):

        image = self._load_image(img_id)
        target = self._load_target(img_id)

        target = {'image_id': img_id, 'annotations': target}
        image, target = self.prepare(image, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'], 
                format=datapoints.BoundingBoxFormat.XYXY, 
                spatial_size=image.shape[-2:]) # h w

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])

        return image, target

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        # print('boxes: {}, labels: {}, obj_ids: {}, area: {}, iscrowd: {} '.format(len(targets['boxes']), len(targets['labels']), len(targets['obj_ids']), len(targets['area']), len(targets['iscrowd'])))
        image_id = targets["image_id"].item()
        orig_image_size = targets["orig_size"].tolist() # (w, h)
        gt_instances = Instances(image_size=tuple(img_shape), orig_image_size=tuple(orig_image_size), image_id=image_id)  # (w, h)
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels'].long()
        gt_instances.obj_ids = targets['obj_ids'].long()
        # gt_instances.area = targets['area']
        return gt_instances


    def __getitem__(self, index):
        # self.current_num_frame = 5 #TODO: 
        img_id = self.valid_img_ids[index]
        img_id_start, img_id_end, interval = self._get_sample_range(img_id)

        samples = []
        for img_id_ in range(img_id_start, img_id_end, interval):
            assert self.coco.imgToVideo[img_id_] == self.coco.imgToVideo[img_id]
            assert self.coco.imgToFrameId[img_id_] - self.coco.imgToFrameId[img_id] == img_id_ - img_id
            img_, target_ = self._load_img_and_target(img_id_)
            samples.append((img_, target_))

        if self._transforms is not None:
            samples = self._transforms(samples)
        
        images, targets = [], []
        for (img_, target_) in samples:
            images.append(img_)
            targets.append(self._targets_to_instances(target_, img_shape=(img_.shape[2], img_.shape[1])))

        # if 1: #TODO:
        #     debugger = Debugger(
        #         pause=True,
        #         save_dir='output/debugger_vis',
        #         show_pad=50)
        #     # import pdb; pdb.set_trace()
        #     for idx in range(len(images)):
        #         # sample = samples[idx]
        #         img_s, targets_s = images[idx], targets[idx]
        #         debugger.add_image_with_bbox(img=img_s, meta_data=targets_s, img_id='frame_{}'.format(idx), box_type='cxcywh_norm')
            
        #     debugger.save_all_imgs()
        #     import pdb; pdb.set_trace()


        return images, targets



    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s 
    
    def __len__(self):
        return len(self.valid_img_ids) #TODO:
        # return min(100, len(self.valid_img_ids))

@register
def labels_getter_func_for_mot_in_SanitizeBoundingBox(inputs):
    targets = inputs[1] # image, targets
    labels = []
    for k in ['labels', 'obj_ids', 'area', 'iscrowd']:
        labels.append(targets[k])
    return tuple(labels)



@register
class LMOTEvalTracking(LMOTTracking):
    __inject__ = ['transforms']
    __share__ = ['remap_category']
    
    def __init__(
            self, 
            img_folder, 
            ann_file, 
            transforms, 
            return_masks, 
            remap_category=False, 
            clip_box=True, 
            visibility_thr=-1, # for evaluation, no gt boxes are filtered
            image_n_bit=None, # the number of bits to load raw images
        ):

        self.coco = VideoCOCO(ann_file)

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

        video_ids = []
        video_names = []
        for vid, video in self.coco.videos.items():
            video_ids.append(vid)
            video_names.append(video['sequence_name'])

        self.video_ids = video_ids
        self.video_names = video_names
        self.num_videos = len(self.video_names)
        self.current_video_index = -1
        self.current_video_name = 'null'
        self.set_for_next_video()

    @property
    def dataset_name(self):
        pre = 'dataset'+os.path.sep
        start = self.img_folder.index(pre)
        dataset_name = self.img_folder[start+len(pre):].split(os.path.sep)[0]
        return dataset_name

    @property
    def data_split(self):
        data_split = os.path.basename(self.ann_file).split('_')[0]
        return data_split

    @property
    def videos_root(self):
        videos_root = os.path.join(self.img_folder, self.data_split) # dataset/LMOT/images/test 
        return videos_root

    def set_for_next_video(self, video=None):
        if video is None:
            self.current_video_index += 1
        elif isinstance(video, (int, )): # video index
            self.current_video_index = video
        elif isinstance(video, (str, )): # video name
            self.current_video_index = self.video_names.index(video)

        video_id = self.video_ids[self.current_video_index]
        self.current_video_name = self.video_names[self.current_video_index]
        self.valid_img_ids = self.coco.videos[video_id]['image_ids']

    def __getitem__(self, index):
        
        img_id = self.valid_img_ids[index]
        image, targets = self._load_img_and_target(img_id)
        # if 'dark' in self.ann_file or 'real' in self.ann_file:
        #     scale = 127.0 / (image.mean() + 1e-8)
        #     orig_image = (image * scale).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8) # H x W x 3
        # else:
        orig_image = image.clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8) # H x W x 3
        orig_targets = self._targets_to_instances(targets, img_shape=(orig_image.shape[1], orig_image.shape[0]))

        if self._transforms is not None:
            image, targets = self._transforms(image, targets)

        targets = self._targets_to_instances(targets, img_shape=(image.shape[2], image.shape[1]))

        # if 1: #TODO:
        #     debugger = Debugger(
        #         pause=True,
        #         save_dir='output/debugger_vis',
        #         show_pad=50)
        #     debugger.add_image_with_bbox(img=image, meta_data=targets, img_id='frame', box_type='xyxy')
            
        #     debugger.save_all_imgs()
        #     import pdb; pdb.set_trace()

        output = {
            'origin_image': orig_image, 
            'origin_image_wh': (orig_image.shape[1], orig_image.shape[0]),
            'origin_targets': orig_targets,
            'image_id': img_id,
            'targets': targets,
            'image': image,
            'frame_id': self.coco.imgToFrameId[img_id],

        }

        return output

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s 
    
    def __len__(self):
        return len(self.valid_img_ids) #TODO:
        # return min(20, len(self.valid_img_ids))