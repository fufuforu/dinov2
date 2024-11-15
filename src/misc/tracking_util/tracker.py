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
# from __future__ import print_function

import os
import numpy as np
import random
import torch
import cv2
from tqdm import tqdm
from typing import List
from src.data.coco.coco_eval import CocoEvaluator
from src.data.coco.coco_utils import get_coco_api_from_dataset
from src.misc.instances import Instances
from src.data.category_map import LABEL2CATEGORY_DICT
from src.core import register
from src.misc.tracking_util.evaluation import Evaluator
from src.misc import dist
from src.core.yaml_utils import save_config
import motmetrics as mm
import sys


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


@register
class MOTRTracker(object):
    __share__ = ['remap_category', 'score_thresh']
    def __init__(
            self, 
            # output_dir: str, 
            # model: torch.nn.Module, 
            # dataset: torch.utils.data.Dataset, 
            # epoch: int, 
            mot_or_det='mot',
            remap_category=False,
            score_thresh=0.7,
            area_threshold=100, 
            visualize=None, 
            clip_box=False
        ):

        # self.output_dir = output_dir
        # self.model = model
        # self.dataset = dataset
        # self.epoch = epoch
        self.mot_or_det = mot_or_det
        self.remap_category = remap_category
        self.score_thresh = score_thresh
        self.area_threshold = area_threshold
        self.visualize = visualize
        self.clip_box = clip_box

        self.model = None
        self.epoch = 0
        self.dataset = None
        self.output_dir = None
        self.coco_evaluator = None

        assert self.mot_or_det in ['mot', 'det', 'mot_det', 'det_mot'], 'Unknown mot or det: {}'.format(self.mot_or_det) # 

        self.track_results = {}
        self.track_accs = []
        self.track_video_names = []


    def prepare(self, model, dataset, cfg, output_dir, **kargs):
        self.model = model
        self.dataset = dataset 
        self.cfg = cfg 
        self.output_dir = output_dir
        for k, v in kargs.items():
            setattr(self, k, v)

        self.coco_evaluator = CocoEvaluator(get_coco_api_from_dataset(self.dataset), iou_types=['bbox'])
        
        # create directory to save tracking results
        assert self.score_thresh == self.model.runtime_track.score_thresh #FIXME: is this too ugly?
        self.result_dir = os.path.join(self.output_dir, 'preds_e{}'.format(self.epoch), self.dataset.data_split+'_thres_{}'.format(self.score_thresh))
        
        if dist.is_main_process():
            os.makedirs(self.result_dir, exist_ok=True)
            
            # save config 
            cfg_save_path = os.path.join(self.result_dir, 'config.yml')
            save_config(cfg, save_path=cfg_save_path, verbose=True)
       

        

    def filter_dt_by_score(self, track_instances_out: Instances) -> Instances:
        keep = track_instances_out.scores > self.score_thresh
        return track_instances_out[keep]

    def filter_dt_by_area(self, track_instances_out: Instances) -> Instances:
        wh = track_instances_out.boxes[:, 2:4] - track_instances_out.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > self.area_threshold
        return track_instances_out[keep]
    
    def filter_dt_by_object_id(self, track_instances_out: Instances) -> Instances:
        keep = track_instances_out.obj_ids >= 0
        return track_instances_out[keep]


    def keep_top_k(self, track_instances_out: Instances, topk: int = 100) -> Instances:
        keep = track_instances_out.scores.topk(100)[1]
        return track_instances_out[keep]

    def write_tracking_results(self, txt_path, dataset_name='mot17'):
        dataset_name = dataset_name.lower()
        if dataset_name in ['mot17', 'mot15']:
            save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n' 
        elif dataset_name in ['lmot']:
            save_format = '{frame},{id},{x1},{y1},{w},{h},1,{label},-1\n'
        else:
            raise NotImplementedError(dataset_name)
        frame_ids = sorted(list(self.track_results.keys()))
        with open(txt_path, 'w') as f:
            for frame_id in frame_ids:
                frame_res = self.track_results[frame_id]
                # import pdb; pdb.set_trace()
                bbox_xyxy=frame_res[:, :4]
                identities=frame_res[:, 6]
                labels=frame_res[:, 5]
                for xyxy, track_id, label in zip(bbox_xyxy, identities, labels):
                    # if track_id < 0 or track_id is None:
                    #     continue
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    if dataset_name in ['mot17', 'mot15']:
                        line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h)
                    elif dataset_name in ['lmot']:
                        line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h, label=int(label))
                    f.write(line)
    
    def eval_frame_detection_results(self, det_instances, image_id):
        if 'det' in self.mot_or_det:
            det_res = {
                image_id: {
                    'boxes': det_instances.boxes, # x1, y1, x2, y2
                    'labels': det_instances.labels,
                    'scores': det_instances.scores,
                }
            }
            self.coco_evaluator.update(det_res, verbose=False)
            
    def summarize_detection_metrics(self):
        if 'det' in self.mot_or_det:
            dist.sync_all_rank()
            self.coco_evaluator.synchronize_between_processes()
            self.coco_evaluator.accumulate()
            results = self.coco_evaluator.summarize()['bbox']

            bbox_metrics = self.coco_evaluator.coco_eval['bbox'].stats

            save_path = os.path.join(self.result_dir, 'detection_det_results.txt')
            if 'mot' in self.mot_or_det:
                save_path = os.path.join(self.result_dir, 'detection_mot_results.txt')
            if dist.is_main_process():
                with open(save_path, 'w') as f:
                    f.write(str(bbox_metrics)+'\n')
                    f.write(results)
                print('results saved to {}'.format(save_path)) 

    def eval_video_tracking_results(self):
        if 'mot' in self.mot_or_det:
            data_root = self.dataset.videos_root
            result_filename = os.path.join(self.result_dir, "{}.txt".format(self.dataset.current_video_name))
            evaluator = Evaluator(data_root, self.dataset.current_video_name)
            accs = evaluator.eval_file(result_filename)
            self.track_accs.append(accs)
            self.track_video_names.append(self.dataset.current_video_name)

    def summarize_tracking_metrics(self):
        if 'mot' in self.mot_or_det:
            dist.sync_all_rank()
            metrics = mm.metrics.motchallenge_metrics
            namemap=mm.io.motchallenge_metric_names
            metrics.extend(["deta_alpha", "assa_alpha", "hota_alpha"])
            namemap.update({
                "hota_alpha": "HOTA", 
                "assa_alpha": "ASSA", 
                "deta_alpha": "DETA"
            })

            # import pdb; pdb.set_trace()
            
            # # import pdb; pdb.set_trace()
            # import torch.distributed as tdist
            # world_size = dist.get_world_size()
            # data_list = [None] * world_size
            # tdist.all_gather_object(data_list, self.track_accs)


            track_accs = dist.all_gather(self.track_accs)
            track_video_names = dist.all_gather(self.track_video_names)
            track_accs_ = []
            track_video_names_ = []
            for ta, tvn in zip(track_accs, track_video_names):
                track_accs_.extend(ta)
                track_video_names_.extend(tvn)
            accs_names = zip(track_accs_, track_video_names_)
            accs_names = sorted(accs_names, key=lambda x: x[1])
            track_accs, track_video_names = zip(*accs_names)
            summary = Evaluator.get_summary(track_accs, track_video_names, metrics)
            # summary = Evaluator.get_summary(self.track_accs, self.track_video_names, metrics)
            
            strsummary = mm.io.render_summary(
                summary,
                formatters=mm.metrics.create().formatters,
                namemap=namemap
            )

            if dist.is_main_process():
                print(strsummary)

                try:
                    save_path = os.path.join(self.result_dir, 'tracking_results.csv')
                    Evaluator.save_summary(summary=summary, filename=save_path)
                    print('results saved to {}'.format(save_path))
                except:
                    save_path = os.path.join(self.result_dir, 'tracking_results.txt')
                    with open(save_path, 'w') as f:
                        print(strsummary, file=f)
                    print('results saved to {}'.format(save_path)) 

    def visualize_img_with_bbox(frame_id, img, object_instances: Instances, ref_pts=None, box_type='track'):
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
        if self.visualize == 'save':
            img_path = os.path.join(self.visualize_dir, box_type, 'frame_{}.jpg'.format(frame_id))
            cv2.imwrite(img_path, img_show)
        elif 'show' in self.visualize:
            cv2.imshow(box_type,img_show)
            pause = 'pause' in self.visualize
            if cv2.waitKey(0 if pause else 1) == 27: # press esc to quit
                # cv2.destroyAllWindows()
                sys.exit(0)

    def instances_to_mot_format(self, dt_instances: Instances):
        ret = []
        for i in range(len(dt_instances)):
            label = dt_instances.labels[i]
            if self.remap_category:
                label = LABEL2CATEGORY_DICT[self.remap_category][label]
            id = dt_instances.obj_ids[i] # + 1 #TODO: should id be added by 1?
            # if id < 0:
            #     continue
            box_with_score = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i+1]], axis=-1)
            ret.append(np.concatenate((box_with_score, [label, id])).reshape(1, -1))  # +1 as MOT benchmark requires positive

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

    def prepare_for_video(self, video_name):
        # clear all cached information in detection model
        self.model.clear()
        self.track_results = {}

        self.dataset.set_for_next_video(video=video_name)
        
        if self.visualize == 'save':
            self.visualize_dir = os.path.join(self.output_dir, 'visualize_e{}/{}/{}'.format(self.epoch, self.dataset.data_split+'_thres_{}'.format(self.score_thresh), video_name))
            if dist.is_main_process():
                os.makedirs(self.visualize_dir, exist_ok=True)

    def track_one_video(self, video_idx, total_num_videos):
        total_tracks = 0
        track_instances = None
        max_id = 0
        desc = 'Rank {}: {}, {}/{}'.format(dist.get_rank(), self.dataset.current_video_name, video_idx, total_num_videos)
        for i in tqdm(range(0, len(self.dataset)), position=dist.get_rank(), desc=desc):
            data = self.dataset[i]

            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')

            res = self.model.inference_single_image(
                            img=data['image'].unsqueeze(dim=0).cuda().float(), 
                            orig_img_size=data['origin_image_wh'], 
                            track_instances=track_instances
                        )
            
            if 'mot' in self.mot_or_det: # only evaluate the performance of detection
                track_instances = res['track_instances']

                if self.clip_box:
                    boxes = track_instances.boxes # x1, y1 x2, y2
                    boxes[0::2] = boxes[0::2].clamp(0, data['origin_image_wh'][0])
                    boxes[1::2] = boxes[1:2].clamp(0, data['origin_image_wh'][1])
                    track_instances.boxes = boxes
                
                # import pdb; pdb.set_trace()
                # track_instances = self.filter_dt_by_area(track_instances) #TODO:

                max_id = max(max_id, track_instances.obj_ids.max().item())

                all_ref_pts = res.get('ref_pts', None)
                all_ref_pts = all_ref_pts[0,:,:2].to('cpu').numpy() if all_ref_pts is not None else None

                track_instances_out = track_instances.to(torch.device('cpu'))
                
                # filter det instances by score.   
                track_instances_out = self.filter_dt_by_score(track_instances_out)
                track_instances_out = self.filter_dt_by_area(track_instances_out)
                track_instances_out = self.filter_dt_by_object_id(track_instances_out)
                
                total_tracks += len(track_instances_out)
                if self.visualize:
                    # for visual
                    assert i+1 == data['frame_id']
                    self.visualize_img_with_bbox(data['frame_id'], data['origin_image'], track_instances_out, ref_pts=all_ref_pts, box_type='track')

                tracker_outputs = self.instances_to_mot_format(track_instances_out)
                self.track_results[data['frame_id']] = tracker_outputs
                # self.write_tracking_results(txt_path=os.path.join(self.result_dir, '{}.txt'.format(self.dataset.current_video_name)),
                #                 frame_id=data['frame_id'],
                #                 bbox_xyxy=tracker_outputs[:, :4],
                #                 identities=tracker_outputs[:, 6],
                #                 labels=tracker_outputs[:, 5],
                #                 dataset_name=self.dataset.dataset_name)
            
            if 'det' in self.mot_or_det:
                # save detection results
                det_instances_out = res['det_instances']
                det_instances_out = self.keep_top_k(det_instances_out, topk=100)
                if self.clip_box:
                    boxes = det_instances_out.boxes # x1, y1 x2, y2
                    boxes[0::2] = boxes[0::2].clamp(0, data['origin_image_wh'][0])
                    boxes[1::2] = boxes[1:2].clamp(0, data['origin_image_wh'][1])
                    det_instances_out.boxes = boxes
                self.eval_frame_detection_results(det_instances_out, image_id=data['image_id'])
                if self.visualize:
                    self.visualize_img_with_bbox(data['frame_id'], data['origin_image'], det_instances_out.to(torch.device('cpu')), box_type='det')

                

            
            if self.visualize and 'targets' in data:
                    # for visual
                    self.visualize_img_with_bbox(data['frame_id'], data['origin_image'], data['targets'], box_type='gt')

        if 'mot' in self.mot_or_det:
            self.write_tracking_results(txt_path=os.path.join(self.result_dir, '{}.txt'.format(self.dataset.current_video_name)), dataset_name=self.dataset.dataset_name)
            self.eval_video_tracking_results()
            print("Rank {}: totally {} dts max_id={}".format(dist.get_rank(), total_tracks, max_id))


    def track(self):
        # assert dist.get_world_size() == 1, 'Currently, DDP not supported'

        # beging to tracking
        valid_idxes = [idx for idx in range(len(self.dataset.video_names)) if (idx % dist.get_world_size()) == dist.get_rank()]
        # valid_idxes = [idx for idx in range(5) if (idx % dist.get_world_size()) == dist.get_rank()]

        for i in range(len(valid_idxes)):
            idx  = valid_idxes[i]
            video_name = self.dataset.video_names[idx]
            # print("Rank {}: Evaluation on {}: {}/{}".format(dist.get_rank(), video_name, idx+1, len(self.dataset.video_names)))
            self.prepare_for_video(video_name)
            self.track_one_video(video_idx=i+1, total_num_videos=len(valid_idxes))

        self.summarize_detection_metrics()
        self.summarize_tracking_metrics()
        