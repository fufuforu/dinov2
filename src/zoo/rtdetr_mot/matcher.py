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
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.misc.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
# from src.misc.instances import Instances, BatchInstances
from src.core import register


@register
class HungarianMatcherMOT(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    
    __share__ = ['use_focal_loss', ]
    
    def __init__(self,
                 weight_dict,
                 use_focal_loss=False,
                 alpha=0.25,
                 gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"
    

    @torch.no_grad()
    def forward(self, outputs, targets, outputs_batch_index, targets_batch_index, batch_size):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [num_queries, 4] with the predicted box coordinates

            targets: This is a dict contains at least these entries:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            outputs_batch_index: Tensor of dim [num_queries], the value is the batch index of predictions
            targets_batch_index: Tensor of dim [num_target_boxes], the value is the batch index of predictions
            
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        # predictions
        if self.use_focal_loss:
            out_prob = outputs["pred_logits"].sigmoid()
        else:
            out_prob = outputs["pred_logits"].softmax(-1)  # [num_queries, num_classes]
        out_bbox = outputs["pred_boxes"]  # [num_queries, 4]

        # gts
        tgt_ids = targets['labels'] # [num_targets] 
        tgt_bbox = targets['boxes'] # [num_targets, 4]

        # Compute the classification cost.
        if self.use_focal_loss:
            tgt_ids = tgt_ids.to(torch.int64)
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # num_queries, num_targets

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou # num_queries, num_targets
        C = C.to('cpu')
        
        batch_idxes = torch.unique(outputs_batch_index).tolist()
        device = out_prob.device
        indices = []
        non_idx = torch.zeros((0,), dtype=torch.int64, device=device)
        pred_idx_start = 0
        tgt_idx_start = 0
        for bi in range(batch_size):
            if bi not in batch_idxes:
                indices.append((non_idx, non_idx))
            else:
                num_pred = (outputs_batch_index == bi).sum()
                num_tgt = (targets_batch_index == bi).sum()
                match_idx = linear_sum_assignment(C[pred_idx_start:pred_idx_start+num_pred, tgt_idx_start:tgt_idx_start+num_tgt])
                indices.append((torch.as_tensor(match_idx[0], dtype=torch.int64, device=device), torch.as_tensor(match_idx[1], dtype=torch.int64, device=device)))
                pred_idx_start += num_pred 
                tgt_idx_start += num_tgt
        
        return indices


