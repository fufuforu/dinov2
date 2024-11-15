# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import os
import numpy as np
import copy
import motmetrics as mm
mm.lap.default_solver = 'lap'
import os
from typing import Dict
import numpy as np
import logging

def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    if data_type in ('mot', 'lab'):
        read_fun = read_mot_results
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))

    return read_fun(filename, is_gt, is_ignore)

# def read_mot_results(filename, is_gt, is_ignore):
#     results_dict = dict()
#     if os.path.isfile(filename):
#         with open(filename, 'r') as f:
#             for line in f.readlines():
#                 linelist = line.split(',')
#                 if len(linelist) < 7:
#                     continue
#                 fid = int(linelist[0])
#                 if fid < 1:
#                     continue
#                 results_dict.setdefault(fid, list())

#                 if is_gt:
#                     mark = int(float(linelist[6]))
#                     if mark == 0 :
#                         continue
#                     score = 1
#                 elif is_ignore:
#                     score = 1
#                 else:
#                     score = float(linelist[6])

#                 tlwh = tuple(map(float, linelist[2:6]))
#                 target_id = int(float(linelist[1]))
#                 results_dict[fid].append((tlwh, target_id, score))

#     return results_dict

def read_mot_results(filename, is_gt, is_ignore):
    if 'MOT15' in filename or 'MOT16' in filename or 'MOT17' in filename:
        valid_labels = {1}
        ignore_labels = {0, 2, 7, 8, 12}
    elif 'LMOT' in filename:
        valid_labels = {1,2,3,4,5,6}
        ignore_labels = {}
    else:
        # import pdb; pdb.set_trace()
        raise NotImplementedError('Unknown file: {}'.format(filename))
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())

                if is_gt: # load gt labels, only the considered boxes are kept
                    if 'MOT16' in filename or 'MOT17' in filename: 
                        label = int(float(linelist[7]))
                        mark = int(float(linelist[6]))
                        if mark == 0 or label not in valid_labels:
                            continue
                    elif 'MOT15' in filename:
                        label = 1
                        mark = int(float(linelist[6]))
                        if mark == 0 or label not in valid_labels:
                            continue
                    elif 'LMOT' in filename: # all boxes are considered
                        label = int(float(linelist[7]))
                    score = 1
                elif is_ignore: # load ignored boxes from gt
                    if 'MOT16' in filename or 'MOT17' in filename:
                        label = int(float(linelist[7]))
                        vis_ratio = float(linelist[8])
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    elif 'MOT15' in filename:
                        label = 1
                        mark = int(float(linelist[6]))
                        if label not in ignore_labels:
                            continue
                        if mark != 0:
                            continue
                    elif 'LMOT' in filename:
                        continue # all annotations in LMOT are considered for evaluation
                    else:
                        raise NotImplementedError
                    score = 1
                else: # load tracking results
                    score = float(linelist[6])
                    if 'MOT16-' in filename or 'MOT17-' in filename or 'MOT15' in filename:
                        label = 1 # only person is kept
                    elif 'LMOT-' in filename:
                        label = int(float(linelist[7]))
                    else:
                        raise NotImplementedError

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])
                

                results_dict[fid].append((tlwh, target_id, label, score))

    return results_dict

def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, labels, scores = zip(*objs)
    else:
        tlwhs, ids, labels, scores = [], [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)
    return tlwhs, ids, labels, scores


class Evaluator(object):
    def __init__(self, data_root, seq_name, data_type='mot', class_specific=False):

        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type
        self.class_specific = class_specific

        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        assert self.data_type == 'mot'

        gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, trk_labels, rtn_events=False):
        #TODO: Implement this to category-specific
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)
        trk_labels = np.copy(trk_labels)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids, gt_labels, gt_scores = unzip_objs(gt_objs)

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs, ignore_ids, ignore_labels, ignore_scores = unzip_objs(ignore_objs)
        
        # remove ignored results
        if len(ignore_tlwhs) > 0 and len(trk_tlwhs) > 0:
            keep = np.ones(len(trk_tlwhs), dtype=bool)
            iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]
            trk_labels = trk_labels[keep]

        # get distance matrix
        if np.size(gt_tlwhs) > 0 and np.size(trk_tlwhs)> 0:
            iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)
        else:
            iou_distance = np.empty((len(gt_tlwhs), len(trk_tlwhs)))

        if self.class_specific:
            mask = np.copy(gt_labels)[:, None] == trk_labels[None, :]
            iou_distance = np.where(mask, iou_distance, np.nan) 

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_file(self, filename):
        self.reset_accumulator()
        result_frame_dict = read_results(filename, self.data_type, is_gt=False)
        frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids, trk_labels, trk_scores = unzip_objs(trk_objs)
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, trk_labels, rtn_events=False)

        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()