"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
import copy
from typing import List, Optional

from src.core import register
from src.misc.instances import Instances, BatchInstances
from src.misc.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


__all__ = ['RTDETRForMOT', 'RuntimeTrackerBase']


@register
class RuntimeTrackerBase(object):
    def __init__(
            self, 
            score_thresh=0.7, # threshold to make a tracklet as active
            filter_score_thresh=0.6, # threshold to make a tracklet as disapeared
            miss_tolerance=5
    ):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

        assert self.score_thresh > self.filter_score_thresh

    def clear(self):
        self.max_obj_id = None

    def update(self, track_instances: Instances):

        bs = len(track_instances)
        if self.max_obj_id is None:
            self.max_obj_id = [0] * bs

        valid_mask = track_instances.valid_mask
        track_instances.disappear_time[(track_instances.scores >= self.score_thresh) & valid_mask] = 0

        # new active objects
        mask_active = (track_instances.obj_ids == -1) & (track_instances.scores >= self.score_thresh) & valid_mask# N

        # disappeared objects
        mask_disp = (track_instances.obj_ids >= 0) & (track_instances.scores < self.filter_score_thresh) & valid_mask

        # update the tracks 
        for b in range(bs):
            track_instances.obj_ids[b][mask_active[b]] = torch.arange(mask_active[b].sum()).to(track_instances.obj_ids) + self.max_obj_id[b]
            self.max_obj_id[b] = self.max_obj_id[b] + mask_active[b].sum()

        track_instances.disappear_time[mask_disp] += 1

        mask_disp = mask_disp & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_ids[mask_disp] = -1


        # for i in range(len(track_instances)):
        #     if track_instances.obj_ids[i] == -1 and track_instances.scores[i] >= self.score_thresh:
        #         # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
        #         track_instances.obj_ids[i] = self.max_obj_id
        #         self.max_obj_id += 1
        #     elif track_instances.obj_ids[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
        #         track_instances.disappear_time[i] += 1
        #         if track_instances.disappear_time[i] >= self.miss_tolerance:
        #             # Set the obj_id to -1.
        #             # Then this track will be removed by TrackEmbeddingLayer.
        #             track_instances.obj_ids[i] = -1


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, orig_image_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            orig_size: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        prob = out_logits.sigmoid()
        # prob = out_logits[...,:1].sigmoid()
        scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_w, img_h = orig_image_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = labels
        track_instances.remove('pred_logits')
        track_instances.remove('pred_boxes')
        return track_instances





@register
class RTDETRForMOT(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'memory_bank', 'criterion', 'track_embed', 'runtime_track']

    def __init__(
        self, 
        backbone,
        encoder, 
        decoder, 
        criterion,
        track_embed, # used to update the track query
        runtime_track, # used to update the state of tracks
        memory_bank=None, 
        multi_scale=None,
    ):
        super().__init__()

        self.backbone = backbone
        self.encoder = encoder 
        self.decoder = decoder
        self.multi_scale = multi_scale


        self.num_queries = self.decoder.num_queries
        self.hidden_dim = self.decoder.hidden_dim

        self.criterion = criterion

        # some modules for tracking
        self.track_embed = track_embed
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length
        self.post_process = TrackerPostProcess()
        self.runtime_track = runtime_track # RuntimeTrackerBase()


    def _generate_empty_tracks(self, batch_size:int):
        image_size = [(1,1)] * batch_size
        orig_image_size = [(1,1)] * batch_size
        image_id = [-1] * batch_size
        track_instances = BatchInstances(image_size=image_size, orig_image_size=orig_image_size, 
                                        image_id=image_id)
    
        device = self.decoder.dec_score_head[0].weight.device
            
        track_instances.query_feat = torch.zeros((batch_size,self.num_queries,self.hidden_dim), dtype=torch.float, device=device)
        track_instances.query_pos = torch.zeros((batch_size,self.num_queries,self.hidden_dim), dtype=torch.float, device=device)
        track_instances.ref_pts = torch.zeros((batch_size,self.num_queries, 4), dtype=torch.float, device=device) # unact, before sigmoid
        track_instances.output_embedding = torch.zeros((batch_size,self.num_queries, self.hidden_dim), device=device)
        track_instances.obj_ids = torch.full((batch_size,self.num_queries,), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((batch_size,self.num_queries,), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((batch_size,self.num_queries, ), dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((batch_size,self.num_queries,), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((batch_size,self.num_queries,), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((batch_size,self.num_queries,), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((batch_size,self.num_queries, 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((batch_size,self.num_queries, self.decoder.num_classes), dtype=torch.float, device=device) - torch.inf 
        # track_instances.mem_bank = torch.zeros((batch_size,self.num_queries, self.mem_bank_len, self.hidden_dim // 2), dtype=torch.float32, device=device)
        # track_instances.mem_padding_mask = torch.ones((batch_size,self.num_queries, self.mem_bank_len), dtype=torch.bool, device=device)
        # track_instances.save_period = torch.zeros((batch_size,self.num_queries, ), dtype=torch.float32, device=device)
        track_instances.valid_mask = torch.full((batch_size,self.num_queries,), True, dtype=torch.bool, device=device)
        
        track_instances = track_instances.to(device)
        # import pdb; pdb.set_trace()
        return track_instances


    def clear(self):
        self.runtime_track.clear()
        if self.memory_bank is not None:
            raise NotImplementedError


    def _forward_single_image(self, x, track_instances:BatchInstances, target_instances:BatchInstances=None, input_size=None):
        if input_size is not None:
            x = F.interpolate(x, size=input_size)
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, track_instances=track_instances, target_instances=target_instances)
        
        return x

    def _post_process_single_image(self, frame_res, track_instances, is_last):
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            track_scores, track_labels = frame_res['pred_logits'].sigmoid().max(dim=-1) # [b, num_queries]
        
        #TODO: show to use cls info for training and tracking?
        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits']
        track_instances.pred_boxes = frame_res['pred_boxes']
        track_instances.output_embedding = frame_res['hs']
        track_instances.labels = track_labels

        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            frame_res['decoder_num_layers'] = self.decoder.num_decoder_layers
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            # keep origin detection results for the evaluation of detection performance
            frame_res['det_instances'] = copy.deepcopy(track_instances)
            # each track will be assigned an unique global id by the track base.
            self.runtime_track.update(track_instances)

        if self.memory_bank is not None:
            raise NotImplementedError
            track_instances = self.memory_bank(track_instances)
            # track_instances.track_scores = track_instances.track_scores[..., 0]
            # track_instances.scores = track_instances.track_scores.sigmoid()
            if self.training:
                self.criterion.calc_loss_for_track_scores(track_instances)
        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks(batch_size=track_scores.shape[0])
        tmp['track_instances'] = track_instances
        if not is_last:
            out_track_instances = self.track_embed(tmp)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        
        return frame_res

    @torch.no_grad()
    def inference_single_image(self, img, orig_img_size, track_instances=None):
        # if not isinstance(img, NestedTensor):
        #     img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks(batch_size=img.shape[0])
        res = self._forward_single_image(img, track_instances=track_instances)
        res = self._post_process_single_image(res, track_instances, is_last=False)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, orig_img_size)
        ret = {'track_instances': track_instances}
        if 'det_instances' in res:
            det_instances = self.post_process(res['det_instances'], orig_img_size)
            ret['det_instances'] = det_instances
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_w, img_h = orig_img_size
            scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret    


    def forward(self, frames, gt_instances:List[BatchInstances]):
        """
            frames: list of tensor
            gt_instanes: list of targets
        """
        self.clear()
        
        # random sample an input size
        if self.multi_scale and self.training:
            input_size = np.random.choice(self.multi_scale)
            if isinstance(input_size, (int, float)):
                input_size = (input_size, input_size) # h, w
        else:
            input_size = None
        
        if self.training:
            self.criterion.initialize_for_single_clip(gt_instances)
        
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }
        
        track_instances = self._generate_empty_tracks(batch_size=frames[0].shape[0]) # BatchInstances
        
        for frame_index, frame in enumerate(frames):
            # import pdb; pdb.set_trace()
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1
            frame_gt = gt_instances[frame_index]

            # if (track_instances.query_feat != track_instances.query_feat).sum() > 0:
            #     import pdb; pdb.set_trace()

            # frame = nested_tensor_from_tensor_list([frame])
            frame_res = self._forward_single_image(frame, track_instances=track_instances, target_instances=frame_gt, input_size=input_size)

            # if (track_instances.query_feat != track_instances.query_feat).sum() > 0:
            #     import pdb; pdb.set_trace()

            frame_res = self._post_process_single_image(frame_res, track_instances, is_last)


            # if (track_instances.query_feat != track_instances.query_feat).sum() > 0:
            #     import pdb; pdb.set_trace()

            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs
        
