# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import random
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from models import build_model
from datasets.joint import RawImageReader
from util.tool import load_model
from main import get_args_parser
from torch.nn.functional import interpolate
from typing import List
from pycocotools.coco import COCO
from src.data.coco.coco_eval import CocoEvaluator
from util.evaluation import Evaluator
import motmetrics as mm
import shutil
import sys

from models.structures import Instances

np.random.seed(2020)

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None):
    # Plots one bounding box on image img

    # tl = line_thickness or round(
    #     0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [0,0,0],#[225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img, score, (c1[0], c1[1] + 30), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def draw_bboxes(ori_img, bbox, identities=None, offset=(0, 0), cvt_color=False):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
        else:
            score = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{:d}'.format(id)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label, score=score)
    return img


def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255), offset=(0, 0)) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x+offset[0]), int(y+offset[1])), 2, color=color, thickness=2)
    return img


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()



class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, args, ann_file, dataset_name, root='DATASET'):

        self.dataset_name = dataset_name
        self.root = root
        self.coco = COCO(os.path.join(root, ann_file))

        self.seq_info = {}
        for seq_info in self.coco.dataset['sequences']:
            self.seq_info[seq_info['sequence_name']] = seq_info
        self.seq_names = sorted(list(self.seq_info.keys()))#[0:2]#TODO:
    
        self.seq_name = None

        self.raw_image_reader = RawImageReader()
        
        if args.meta_arch == 'motr_rt_detr':
            self.mean = [0., 0., 0.]
            self.std = [1., 1., 1.]
            self.img_size = (640, 640)
        else:
            self.max_img_height = 800
            self.max_img_width = 1536
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.img_size = None


    def prepare_for_seq(self, seq_name):
        assert seq_name in self.seq_names
        self.seq_name = seq_name

    def __len__(self):
        if self.seq_name is None:
            return len(self.seq_names)
        else:
            return len(self.seq_info[self.seq_name]['image_ids'])
    
    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        img_id = self.seq_info[self.seq_name]['image_ids'][idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        img_info = self.coco.loadImgs(img_id)[0] # dict

        anns = self.coco.loadAnns(ann_ids) # list of instances
        targets = {
            'boxes': [],
            'labels': [],
            'areas': [],
            'obj_idxes':[],
        }
        for ann in anns:
            if 'visibility' in ann and ann['visibility'] < 1:
                continue
            # import pdb; pdb.set_trace()
            targets['labels'].append(ann['category_id'])
            targets['areas'].append(ann['area'])
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x+w, y+h
            targets['boxes'].append([x1, y1, x2, y2])
            targets['obj_idxes'].append(ann['track_id'])
        # targets['boxes'] = np.asarray(targets['boxes'])
        # targets['areas'] = np.asarray(targets['areas'])
        # targets['labels'] = np.asarray(targets['labels'])
        targets = Instances((1, 1), (1,1), -1,
                            boxes=torch.tensor(targets['boxes']),
                            areas=torch.tensor(targets['areas']),
                            labels=torch.tensor(targets['labels']),
                            obj_idxes=torch.tensor(targets['obj_idxes']))

        # load Image
        img_path = os.path.join(self.root, img_info['file_name'])
        # import pdb; pdb.set_trace()
        if img_path.endswith('.tiff'):
            img = self.raw_image_reader(img_path, black_level=img_info['black_level'], numpy_or_tensor='numpy')
            is_raw = True
        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            is_raw = False

        ori_img = img.copy()
        seq_h, seq_w = img.shape[:2]
        if self.img_size is None:
            scale = self.max_img_height / min(seq_h, seq_w)
            if max(seq_h, seq_w) * scale > self.max_img_width:
                scale = self.max_img_width / max(seq_h, seq_w)
            target_h = int(self.seq_h * scale)
            target_w = int(self.seq_w * scale)
        else:
            target_h, target_w = self.img_size
        if is_raw:
            ori_img = np.clip(ori_img, 0, 255).astype(np.uint8) # used for visulaization
            img = torch.Tensor(img).permute(2, 0, 1) # [h, w, 3] -> [3, h, w]
            img = F.resize(img, (target_h, target_w)) / 255
        else:
            img = cv2.resize(img, (target_w, target_h))
            img = F.to_tensor(img)
        img = F.normalize(img, self.mean, self.std)
        img = img.unsqueeze(0)

        out = {
            'image': img, 
            'image_id': img_id,
            'origin_image': ori_img,
            'targets': targets,
            'image_hw': (seq_h, seq_w)
        }

        return out


class Track(object):
    track_cnt = 0

    def __init__(self, box):
        self.box = box
        self.time_since_update = 0
        self.id = Track.track_cnt
        Track.track_cnt += 1
        self.miss = 0

    def miss_one_frame(self):
        self.miss += 1

    def clear_miss(self):
        self.miss = 0

    def update(self, box):
        self.box = box
        self.clear_miss()


class MOTR(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        pass

    def foreground_label(self, dataset_name):
        if dataset_name in ['MOT17', 'MOT16', 'MOT15']:
            return [0]
        elif dataset_name in ['LMOT']:
            return [1,2,3,4,5,6]
        else:
            raise NotImplementedError(dataset_name)

    def update(self, dt_instances: Instances, dataset_name='MOT17'):
        ret = []
        for i in range(len(dt_instances)):
            label = dt_instances.labels[i]
            if label in self.foreground_label(dataset_name=dataset_name):
                id = dt_instances.obj_ids[i]
                box_with_score = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i+1], dt_instances.labels[i:i+1]], axis=-1)
                ret.append(np.concatenate((box_with_score, [id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))


class Detector(object):
    def __init__(self, args, model=None, dataset=None, epoch=0, eval_type='track'):

        self.args = args
        self.detr = model
        self.dataset = dataset
        self.epoch = epoch
        self.eval_type = eval_type

        self.tr_tracker = MOTR()

        # save used to save the detection results into coco format
        self.coco_evaluator = CocoEvaluator(self.dataset.coco, iou_types=['bbox'])
        self.track_accs = []
        self.track_seq_names = []

    @staticmethod
    def filter_dt_by_score(track_instances_out: Instances, prob_threshold: float) -> Instances:
        keep = track_instances_out.scores > prob_threshold
        return track_instances_out[keep]

    @staticmethod
    def filter_dt_by_area(track_instances_out: Instances, area_threshold: float) -> Instances:
        wh = track_instances_out.boxes[:, 2:4] - track_instances_out.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return track_instances_out[keep]
    
    @staticmethod
    def keep_top_k(track_instances_out: Instances, topk: int = 100) -> Instances:
        keep = track_instances_out.scores.topk(100)[1]
        return track_instances_out[keep]

    def prepare_for_seq(self, seq_name):
        # self.detr.clear()
        self.dataset.prepare_for_seq(seq_name)
        
        self.save_path = os.path.join(self.args.output_dir, 'results_e{}/{}/{}'.format(self.epoch, self.args.eval_split.replace(os.path.sep, '_')+'_thres_{}'.format(args.prob_threshold), self.dataset.seq_name))
        os.makedirs(self.save_path, exist_ok=True)

        self.predict_path = os.path.join(args.output_dir, 'preds_e{}'.format(self.epoch), args.eval_split.replace(os.path.sep, '_')+'_thres_{}'.format(args.prob_threshold))
        os.makedirs(self.predict_path, exist_ok=True)
        if 'track' in self.eval_type:
            if os.path.exists(os.path.join(self.predict_path, '{}.txt'.format(self.dataset.seq_name))):
                os.remove(os.path.join(self.predict_path, '{}.txt'.format(self.dataset.seq_name)))


    def write_tracking_results(self, txt_path, frame_id, bbox_xyxy, identities, labels, dataset_name='MOT17'):
        if dataset_name in ['MOT17', 'MOT15']:
            save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n' 
        elif dataset_name in ['LMOT']:
            save_format = '{frame},{id},{x1},{y1},{w},{h},1,{label},-1\n'
        else:
            raise NotImplementedError(dataset_name)
        with open(txt_path, 'a') as f:
            for xyxy, track_id, label in zip(bbox_xyxy, identities, labels):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                if dataset_name in ['MOT17', 'MOT15']:
                    line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h)
                elif dataset_name in ['LMOT']:
                    line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h, label=int(label))
                f.write(line)
    
    def eval_frame_detection_results(self, det_instances, image_id):
        if 'det' in self.eval_type:
            det_res = {
                image_id: {
                    'boxes': det_instances.boxes, # x1, y1, x2, y2
                    'labels': det_instances.labels,
                    'scores': det_instances.scores,
                }
            }
            self.coco_evaluator.update(det_res, verbose=False)
            
    def summarize_detection_metrics(self):
        if 'det' in self.eval_type:
            self.coco_evaluator.synchronize_between_processes()
            self.coco_evaluator.accumulate()
            results = self.coco_evaluator.summarize()['bbox']

            bbox_metrics = self.coco_evaluator.coco_eval['bbox'].stats

            save_path = os.path.join(self.predict_path, 'detection_det_results.txt')
            if 'track' in self.eval_type:
                save_path = os.path.join(self.predict_path, 'detection_track_results.txt')
            with open(save_path, 'w') as f:
                f.write(str(bbox_metrics)+'\n')
                f.write(results)
            print('results saved to {}'.format(save_path)) 

    def eval_seq_tracking_results(self):
        if 'track' in self.eval_type:
            data_root = os.path.join(self.args.mot_path, self.args.eval_split)
            result_filename = os.path.join(self.predict_path, "{}.txt".format(self.dataset.seq_name))
            evaluator = Evaluator(data_root, self.dataset.seq_name)
            accs = evaluator.eval_file(result_filename)
            self.track_accs.append(accs)
            self.track_seq_names.append(self.dataset.seq_name)

    def summarize_tracking_metrics(self):
        if 'track' in self.eval_type:
            metrics = mm.metrics.motchallenge_metrics
            namemap=mm.io.motchallenge_metric_names
            metrics.extend(["deta_alpha", "assa_alpha", "hota_alpha"])
            namemap.update({
                "hota_alpha": "HOTA", 
                "assa_alpha": "ASSA", 
                "deta_alpha": "DETA"
            })

            summary = Evaluator.get_summary(self.track_accs, self.track_seq_names, metrics)
            
            strsummary = mm.io.render_summary(
                summary,
                formatters=mm.metrics.create().formatters,
                namemap=namemap
            )
            print(strsummary)
            # with open("eval_log.txt", 'a') as f:
            #     print(strsummary, file=f)

            try:
                save_path = os.path.join(self.predict_path, 'tracking_results.csv')
                Evaluator.save_summary(summary=summary, filename=save_path)
                print('results saved to {}'.format(save_path))
            except:
                save_path = os.path.join(self.predict_path, 'tracking_results.txt')
                with open(save_path, 'w') as f:
                    print(strsummary, file=f)
                print('results saved to {}'.format(save_path)) 

    @staticmethod
    def visualize_img_with_bbox(frame_id, img, object_instances: Instances, ref_pts=None, vis='show_pause', box_type='track'):
        assert box_type in ['track', 'det', 'gt']
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # padding the image to show the box more clear
        h, w, c = img.shape
        padding = min(h, w) // 15
        img_show = np.zeros((h+2*padding, w+2*padding, c), dtype=np.uint8)
        img_show[padding:padding+h, padding:padding+w,:] = img

        if object_instances.has('scores'):
            boxes = np.concatenate([object_instances.boxes, object_instances.scores.reshape(-1, 1)], axis=-1)
        else:
            boxes = object_instances.boxes
        if box_type == 'det':
            identities = object_instances.labels
        else:
            identities = object_instances.obj_ids
        img_show = draw_bboxes(img_show, boxes, identities, offset=(padding, padding))

        if ref_pts is not None:
            img_show = draw_points(img_show, ref_pts, offset=(padding, padding))
        if max(img_show.shape) > 800:
            scale = 800 / max(img_show.shape)
            h = int(img_show.shape[0] * scale)
            w = int(img_show.shape[1] * scale)
            img_show = cv2.resize(img_show, (w,h))
        cv2.putText(img_show, str(frame_id), (20,40), 0, fontScale=0.5, color=[0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
        if vis == 'save':
            img_path = os.path.join(self.save_path, box_type, 'frame_{}.jpg'.format(frame_id))
            cv2.imwrite(img_path, img_show)
        elif 'show' in vis:
            cv2.imshow(box_type,img_show)
            pause = 'pause' in vis
            if cv2.waitKey(0 if pause else 1) == 27: # press esc to quit
                # cv2.destroyAllWindows()
                sys.exit(0)


    def detect_one_seq(self, prob_threshold=0.7, area_threshold=100, vis=None, clip_box=False):
        total_tracks = 0
        track_instances = None
        max_id = 0
        for i in tqdm(range(0, len(self.dataset))):
            data = self.dataset[i]

            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')
            res = self.detr.inference_single_image(data['image'].cuda().float(), data['image_hw'], track_instances)
            
            if 'track' in self.eval_type: # only evaluate the performance of detection
                track_instances = res['track_instances']

                if clip_box:
                    boxes = track_instances.boxes # x1, y1 x2, y2
                    boxes[0::2] = boxes[0::2].clamp(0, data['image_hw'][1])
                    boxes[1::2] = boxes[1:2].clamp(0, data['image_hw'][0])
                    track_instances.boxes = boxes
                
                # import pdb; pdb.set_trace()
                # track_instances = self.filter_dt_by_area(track_instances, area_threshold) #TODO:


                max_id = max(max_id, track_instances.obj_ids.max().item())

                all_ref_pts = tensor_to_numpy(res['ref_pts'][0, :, :2])
                track_instances_out = track_instances.to(torch.device('cpu'))
                
                # filter det instances by score.   
                track_instances_out = self.filter_dt_by_score(track_instances_out, prob_threshold)
                track_instances_out = self.filter_dt_by_area(track_instances_out, area_threshold)
                

                total_tracks += len(track_instances_out)
                if vis:
                    # for visual
                    self.visualize_img_with_bbox(i+1, data['origin_image'], track_instances_out, ref_pts=all_ref_pts, vis=vis,box_type='track')

                tracker_outputs = self.tr_tracker.update(track_instances_out, dataset_name=self.dataset.dataset_name)
                self.write_tracking_results(txt_path=os.path.join(self.predict_path, '{}.txt'.format(self.dataset.seq_name)),
                                frame_id=(i + 1),
                                bbox_xyxy=tracker_outputs[:, :4],
                                identities=tracker_outputs[:, 6],
                                labels=tracker_outputs[:, 5],
                                dataset_name=self.dataset.dataset_name)
            if 'det' in self.eval_type:
                # save detection results
                det_instances_out = res['det_instances']
                det_instances_out = self.keep_top_k(det_instances_out, topk=100)
                if clip_box:
                    boxes = det_instances_out.boxes # x1, y1 x2, y2
                    boxes[0::2] = boxes[0::2].clamp(0, data['image_hw'][1])
                    boxes[1::2] = boxes[1:2].clamp(0, data['image_hw'][0])
                    det_instances_out.boxes = boxes
                self.eval_frame_detection_results(det_instances_out, image_id=data['image_id'])
                if vis:
                    # det_instances_out = self.filter_dt_by_score(det_instances_out, prob_threshold)
                    # det_instances_out = self.filter_dt_by_area(det_instances_out, area_threshold)
                    self.visualize_img_with_bbox(i+1, data['origin_image'], det_instances_out.to(torch.device('cpu')), vis=vis,box_type='det')

                

            
            if vis and 'targets' in data:
                    # for visual
                    self.visualize_img_with_bbox(i+1, data['origin_image'], data['targets'], vis=vis,box_type='gt')

        if 'track' in self.eval_type:
            print("totally {} dts max_id={}".format(total_tracks, max_id))
            self.eval_seq_tracking_results()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )

    args = parser.parse_args()


    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--eval_split', default='LMOT/images/val',
                        help='which split to eval')
    parser.add_argument('--img_split', default='img1', 
                        choices=['img1', 'img_real', 'image_real_ns_isp', 'img_dark', 'img_dark_ns_isp', 'img_light', 'img_light_isp'],
                        help='which type of image to eval')
    
    parser.add_argument('--eval_type', default='track_det', choices=['track', 'track_det', 'det'],
                        help='track_det: evaluation the tracking perforance; '
                             'det: evaluation the detection, without using tracking'
                             'track_det: evaluate the tracking and detection performance, the detection is associated with tracking')
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and weights
    detr, _, _ = build_model(args)
    detr = load_model(detr, args.resume)
    detr = detr.cuda()
    detr.eval()
    detr.track_base.score_thresh = args.prob_threshold #FIXME: hack implementation of the tracking threshold
    checkpoint = torch.load(args.resume, map_location='cpu')
    epoch = checkpoint.get('epoch', 0)

    # load dataset
    dataset_name, _, split = tuple(args.eval_split.split(os.path.sep))
    ann_file = os.path.join(dataset_name, 'annotations', '{}_{}.json'.format(split, args.img_split))
    dataset = SimpleDataset(args=args, ann_file=ann_file, dataset_name=dataset_name, root=args.mot_path)
    detector = Detector(args, model=detr, dataset=dataset, epoch=epoch, eval_type=args.eval_type)
    
    # beging to tracking
    for seq_name in dataset.seq_names:
        print("solve {}".format(seq_name))
        detector.prepare_for_seq(seq_name)
        detector.detect_one_seq(vis=args.vis, prob_threshold=args.prob_threshold, clip_box=args.clip_box)

    detector.summarize_tracking_metrics()
    detector.summarize_detection_metrics()
