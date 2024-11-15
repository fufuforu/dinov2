# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import random
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, List

from src.zoo.rtdetr.utils import inverse_sigmoid
from src.misc.boxes import Boxes, pairwise_iou
from src.misc.box_ops import box_cxcywh_to_xyxy
from src.misc.instances import Instances

from src.nn.quantization.lsq_plus import LinearLSQ
from src.nn.quantization.multi_head_attention import QuantMultiheadAttention
from src.core import register

def drop_ratio_tracks(track_instances: Instances, drop_probability: float) -> Instances:
    if drop_probability > 0 and len(track_instances) > 0:
        keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
        track_instances = track_instances[keep_idxes]
    return track_instances


class QueryInteractionBase(nn.Module):

    def _build_layers(self):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0, n_bit=None):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn) if n_bit is None else LinearLSQ(d_model, d_ffn, nbits_w=n_bit)
        self.activation = nn.ReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model) if n_bit is None else LinearLSQ(d_model, d_ffn, nbits_w=n_bit)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt


@register
class QueryInteractionModule(QueryInteractionBase):
    def __init__(self, 
            dim, 
            hidden_dim, 
            dropout=0., # dropout in the layer
            drop_ratio=0.1, # probability to drop tracks
            fp_ratio=0.3, # probability to add fp tracks
            update_query_pos=True, # update the position embedding of queries
            n_bit=None
        ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_bit = n_bit
        self.drop_ratio = drop_ratio
        self.fp_ratio = fp_ratio
        self.update_query_pos = update_query_pos

        self._build_layers()
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _build_layers(self):
        dim = self.dim
        hidden_dim = self.hidden_dim
        dropout = self.dropout
        n_bit = self.n_bit

        # self.self_attn = nn.MultiheadAttention(dim, 8, dropout) if n_bit is None else QuantMultiheadAttention(dim, 8, n_bit=n_bit, dropout=dropout)
        self.self_attn = QuantMultiheadAttention(dim, 8, n_bit=n_bit, dropout=dropout)
        
        self.linear1 = nn.Linear(dim, hidden_dim) if n_bit is None else LinearLSQ(dim, hidden_dim, nbits_w=n_bit)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim) if n_bit is None else LinearLSQ(hidden_dim, dim, nbits_w=n_bit)

        # if self.update_query_pos:
        #     self.linear_pos1 = nn.Linear(dim, hidden_dim) if n_bit is None else LinearLSQ(dim, hidden_dim, nbits_w=n_bit)
        #     self.linear_pos2 = nn.Linear(hidden_dim, dim) if n_bit is None else LinearLSQ(hidden_dim, dim, nbits_w=n_bit)
        #     self.dropout_pos1 = nn.Dropout(dropout)
        #     self.dropout_pos2 = nn.Dropout(dropout)
        #     self.norm_pos = nn.LayerNorm(dim)

        self.linear_feat1 = nn.Linear(dim, hidden_dim) if n_bit is None else LinearLSQ(dim, hidden_dim, nbits_w=n_bit)
        self.linear_feat2 = nn.Linear(hidden_dim, dim) if n_bit is None else LinearLSQ(hidden_dim, dim, nbits_w=n_bit)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if self.update_query_pos:
            self.norm3 = nn.LayerNorm(dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if self.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _drop_ratio_tracks(self, track_instances: Instances) -> Instances:
        return drop_ratio_tracks(track_instances, self.drop_ratio)

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
        inactive_instances = track_instances[track_instances.obj_ids < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[torch.bernoulli(fp_prob).bool()]

        if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
            num_fp = len(selected_active_track_instances)
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_active_boxes = Boxes(box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                # select the fp with the largest IoU for each active track.
                fp_indexes = ious.max(dim=0).indices

                # remove duplicate fp.
                fp_indexes = torch.unique(fp_indexes)
                fp_track_instances = inactive_instances[fp_indexes]

            #TODO: set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            # fp_track_instances.obj_ids = torch.zeros_like(fp_track_instances.obj_ids) - 2 # MOTR do not do this
            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_ids >= 0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._drop_ratio_tracks(active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[track_instances.obj_ids >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        tgt = track_instances.output_embedding
        dim = tgt.shape[1]
        query_feat = track_instances.query_feat
        query_pos = track_instances.query_pos
        q = k = query_pos + tgt

        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # if self.update_query_pos:
        #     query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
        #     query_pos = query_pos + self.dropout_pos2(query_pos2)
        #     query_pos = self.norm_pos(query_pos)
        #     track_instances.query_pos[:, :dim] = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_feat = query_feat

        track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :track_instances.ref_pts.shape[1]].detach().clone())
        return track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances: Instances = data['init_track_instances']
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances

