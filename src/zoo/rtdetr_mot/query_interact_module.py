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
from src.misc.instances import Instances, BatchInstances

from src.nn.quantization.lsq_plus import LinearLSQ
from src.nn.quantization.multi_head_attention import QuantMultiheadAttention
from src.core import register


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

    def _drop_ratio_tracks(self, track_instances: BatchInstances) -> BatchInstances:
        if self.drop_ratio > 0 and track_instances.valid_mask.sum() > 0:
            keep_mask = torch.rand_like(track_instances.scores) > self.drop_ratio # b, N
            # import pdb; pdb.set_trace()
            track_instances.valid_mask = track_instances.valid_mask * keep_mask
        return track_instances


    def _add_fp_tracks(self, track_instances: BatchInstances) -> BatchInstances:
        inactive_mask = track_instances.obj_ids < 0 # b, n
        active_mask = track_instances.valid_mask # b, n
        assert (inactive_mask * active_mask).sum() == 0, 'inactive and active tracks overlapped!'

        fp_prob = torch.ones_like(track_instances.scores) * self.fp_ratio # b, n
        selected_active_mask = torch.bernoulli(fp_prob).bool() * active_mask # b, n

        if inactive_mask.sum() > 0 and selected_active_mask.sum() > 0:
            boxes1 = box_cxcywh_to_xyxy(track_instances.pred_boxes).detach() # b, n, 4
            
            ious = pairwise_iou(boxes1, boxes1) # b, n, n
            
            # select the inactive tracks only for active tracks
            ious_mask = selected_active_mask.unsqueeze(dim=-1) * inactive_mask.unsqueeze(dim=-2) # b, n, n
            eye_mask = torch.eye(ious.shape[1], ious.shape[2]).unsqueeze(dim=0).repeat(ious.shape[0],1,1).to(ious_mask)
            ious_mask = ious_mask & (~eye_mask)
            ious[~ious_mask] = -torch.inf

            # select the fp with the largest IoU for each active track.
            fp_ious, fp_indexes = ious.max(dim=-1) # b, n

            # import pdb; pdb.set_trace();
            # selected_active_indexes = selected_active_mask.nonzero() # m, 2
            # track_instances.valid_mask[selected_active_indexes[0], selected_active_indexes[1]]

            for b in range(fp_indexes.shape[0]):
                track_instances.valid_mask[b, fp_indexes[b][selected_active_mask[b]]] = True
                # TODO: set -2 instead of -1 to ensure that these tracks will not be selected in matching.
                # track_instances.obj_ids[b, fp_indexes[b][selected_active_mask[b]]] = -2  # MOTR do not do this

        return track_instances


    def _select_active_tracks(self, data: dict) -> BatchInstances:
        track_instances: BatchInstances = data['track_instances']
        if self.training:
            active_mask = (track_instances.obj_ids >= 0) & (track_instances.iou > 0.5) # b, n
            track_instances.valid_mask = track_instances.valid_mask * active_mask
            track_instances = self._drop_ratio_tracks(track_instances)
            if self.fp_ratio > 0:
                track_instances = self._add_fp_tracks(track_instances)
        else:
            active_mask = track_instances.obj_ids >= 0 # b, n
            track_instances.valid_mask = track_instances.valid_mask * active_mask

        return track_instances

    def _update_track_embedding(self, track_instances: BatchInstances) -> BatchInstances:
        if track_instances.valid_mask.sum() <= 0:
            return track_instances

        tgt = track_instances.output_embedding # b, n, d
        query_feat = track_instances.query_feat  # b, n, d
        query_pos = track_instances.query_pos  # b, n, d
        q = k = query_pos + tgt  # b, n, d

        #TODO: get the attn mask for valid trackss
        if track_instances.valid_mask.sum() < track_instances.valid_mask.numel():
            attn_mask = torch.full([tgt.shape[0], tgt.shape[1], tgt.shape[1]], False, dtype=torch.bool, device=tgt.device)
            for b in range(tgt.shape[0]):
                attn_mask[b, ~track_instances.valid_mask[b], :] = True
                attn_mask[b, :, ~track_instances.valid_mask[b]] = True
            attn_mask = attn_mask.unsqueeze(dim=1).tile(1, self.self_attn.num_heads, 1, 1).reshape(tgt.shape[0]*self.self_attn.num_heads, tgt.shape[1], tgt.shape[1]) 
        else:
            attn_mask = None
        tgt2 = self.self_attn(q.transpose(0,1), k.transpose(0,1), value=tgt.transpose(0,1), attn_mask=attn_mask)[0].transpose(0,1) # b, n, d
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
        track_instances.query_feat = query_feat # b, n, d

        track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[..., :track_instances.ref_pts.shape[-1]].detach().clone())
        
        return track_instances


    def forward(self, data) -> BatchInstances:
        track_instances = self._select_active_tracks(data)
        track_instances = BatchInstances.remove_invalid(track_instances)
        track_instances = self._update_track_embedding(track_instances)
        init_track_instances: BatchInstances = data['init_track_instances']
        merged_track_instances = BatchInstances.cat([init_track_instances, track_instances])

        return merged_track_instances

