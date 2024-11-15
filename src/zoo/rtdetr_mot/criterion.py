import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from src.core import register
from src.zoo.rtdetr.rtdetr_criterion import SetCriterion
from src.misc.instances import Instances, BatchInstances
from src.misc.boxes import Boxes, matched_boxlist_iou
from src.misc.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from src.misc.dist import is_dist_available_and_initialized, get_world_size
from .matcher import HungarianMatcherMOT
import torchvision
from typing import List


def get_tensor_with_index(tensor:torch.Tensor, index:torch.Tensor):
    # import pdb; pdb.set_trace()
    pass 


@register
class ClipSetCriterion(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss']
    __inject__ = ['matcher', ]

    def __init__(
            self, 
            matcher,
            weight_dict,
            losses,
            num_classes=80,
            use_focal_loss=True,
            alpha=0.2,
            gamma=2.0,
            dec_layers_for_track=None,
            ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        # import pdb; pdb.set_trace()
        self.matcher = matcher
        assert isinstance(matcher, HungarianMatcherMOT)

        self.weight_dict = weight_dict
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma

        self.use_focal_loss = use_focal_loss
        self.losses_dict = {}
        self._current_frame_idx = 0
        self.dec_layers_for_track = dec_layers_for_track

    def initialize_for_single_clip(self, gt_instances: List[List[Instances]]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        raise RuntimeError('This function has not been checked!')
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
            'pred_boxes': track_instances.pred_boxes[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        loss_name = 'labels'
        for name in self.losses:
            if 'labels' in name:
                loss_name = name
                break
        track_losses = self.get_loss(loss_name,
                                     outputs=outputs,
                                     gt_instances=[gt_instances],
                                     indices=[(src_idx, tgt_idx)],
                                     num_boxes=1)
        self.losses_dict.update(
            {'frame_{}_track_{}'.format(frame_id, key): value for key, value in
             track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    # def _get_src_permutation_idx(self, indices):
    #     # permute predictions following indices
    #     batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    #     src_idx = torch.cat([src for (src, _) in indices])
    #     return batch_idx, src_idx

    # def _get_tgt_permutation_idx(self, indices):
    #     # permute targets following indices
    #     batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    #     tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    #     return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'labels_vfl': self.loss_labels_vfl,
            # 'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs, gt_instances: BatchInstances, indices: torch.Tensor, num_boxes, query_valid_mask=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        indices = indices[indices[:,2] != -1]
        src_boxes = outputs['pred_boxes'][indices[:,0], indices[:,1]]
        target_boxes = gt_instances.boxes[indices[:,0], indices[:,2]]

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = gt_instances.obj_ids[indices[:,0], indices[:,2]]
        mask = target_obj_ids != -1

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes[mask]),
            box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_labels(self, outputs, gt_instances: BatchInstances, indices: torch.Tensor, num_boxes, log=False, query_valid_mask=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits'] # b, num_query, num_class

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)  # set the class id of backgound query to be the number of classes
        # The matched gt for disappear track query is set -1.
        indices_f = indices[indices[:,2] != -1]
        target_classes[indices_f[:,0], indices_f[:,1]] = gt_instances.labels[indices_f[:, 0], indices_f[:,2]]

        if self.use_focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = torchvision.ops.sigmoid_focal_loss(src_logits, gt_labels_target, self.alpha, self.gamma, reduction='none') # B x N x C
            if query_valid_mask is not None:
                loss_ce = loss_ce * query_valid_mask.unsqueeze(dim=-1) # B x N x C
                loss_ce = (loss_ce.sum(dim=1)/query_valid_mask.sum(dim=1).unsqueeze(dim=-1)).sum() * query_valid_mask.sum(dim=1).mean() / num_boxes
            else:    
                loss_ce = loss_ce.mean(dim=1).sum() * src_logits.shape[1] / num_boxes
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction=None)
            if query_valid_mask is None:
                loss_ce = loss_ce.mean()
            else:
                query_valid_mask = query_valid_mask.to(loss_ce)
                loss_ce = loss_ce.mean(dim=-1).sum() / query_valid_mask.sum()
        losses = {'loss_label': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[indices[:,0],indices[:,1]], target_classes[indices[:, 0], indices[:,1]])[0]
        
        return losses

    def loss_labels_vfl(self, outputs, gt_instances: BatchInstances, indices: torch.Tensor, num_boxes, log=False, query_valid_mask=None):
        assert 'pred_boxes' in outputs
        
        src_boxes = outputs['pred_boxes'][indices[:, 0], indices[:, 1]] # num_match x 4
        
        indices_f = indices[indices[:, 2] != -1] # The matched gt for disappear track query is set -1.
        target_boxes = torch.zeros_like(src_boxes)
        # import pdb; pdb.set_trace()
        target_boxes[indices[:, 2] != -1] = gt_instances.boxes[indices_f[:, 0], indices_f[:, 2]]
        target_boxes[indices[:, 2] == -1] = src_boxes[indices[:, 2] == -1].detach() # disapperaed tracks is confirmed, so its iou weight should be 1
        ious = matched_boxlist_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)).detach() # num_match #TODO: check iou

        src_logits = outputs['pred_logits']
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)  # set the class id of backgound query to be the number of classes
        target_classes[indices_f[:, 0], indices_f[:, 1]] = gt_instances.labels[indices_f[:, 0], indices_f[:, 2]]
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1] # b, num_queries

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[indices[:, 0], indices[:, 1]] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(dim=-1) * target # b, num_queries, 1

        pred_score = src_logits.sigmoid().detach() # b, num_queries, num_classes
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score # b, num_queries, num_classes
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none') # b, num_queries, num_classes
        
        
        if query_valid_mask is not None:
            # if query_valid_mask.sum() < query_valid_mask.numel():
            #     import pdb; pdb.set_trace()
            query_valid_mask = query_valid_mask.to(loss)
            loss = loss * query_valid_mask.unsqueeze(dim=-1) # B x N x C
            loss = (loss.sum(dim=1)/query_valid_mask.sum(dim=1).unsqueeze(dim=-1)).sum() * query_valid_mask.sum(dim=1).mean() / num_boxes
        else:    
            loss = loss.mean(dim=1).sum() * src_logits.shape[1] / num_boxes

        losses = {'loss_label': loss}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[indices[:,0],indices[:,1]], target_classes[indices[:, 0], indices[:,1]])[0]

        return losses


    def match_for_single_frame(self, outputs: dict):

        outputs_without_aux = {k: v for k, v in outputs.items() if ('aux_outputs' not in k and 'dn_' not in k)}

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image. BatchInstances
        track_instances: BatchInstances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image. [b, n]
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image. [b, n]
        outputs_i = {
            'pred_logits': pred_logits_i,
            'pred_boxes': pred_boxes_i,
        }

        bsz = pred_logits_i.shape[0]
        device = pred_logits_i.device
        num_tracks = pred_logits_i.shape[1]
        num_gts = gt_instances_i.obj_ids.shape[-1]

        # step1. inherit and update the previous tracks.
        # import pdb; pdb.set_trace()
        # valid_track_mask = (track_instances.obj_ids >= 0) * track_instances.valid_mask # b, num_tracks
        # invalid_track_index = (~valid_track_mask).nonzero()
        det_query_index = ((track_instances.obj_ids < 0) * track_instances.valid_mask).nonzero() # num, 2
        track_instances.matched_gt_idxes[det_query_index[:,0], det_query_index[:,1]] = -1

        matched_mask = track_instances.obj_ids.unsqueeze(dim=-1) == gt_instances_i.obj_ids.unsqueeze(dim=-2) # b, num_tracks, num_gts
        matched_mask = matched_mask & (track_instances.valid_mask & (track_instances.obj_ids >= 0)).unsqueeze(dim=-1) & gt_instances_i.valid_mask.unsqueeze(dim=-2) # b, num_tracks, num_gts
        matched_idxes = matched_mask.nonzero() #  num_matched, 3. The existing tracked objects
        track_instances.matched_gt_idxes[matched_idxes[:,0], matched_idxes[:,1]] = matched_idxes[:, 2]
        unmatched_mask = (matched_mask.sum(dim=-1) == 0) & track_instances.valid_mask & (track_instances.obj_ids >= 0) # b, num_tracks    # the disappeared objects
        unmatched_idxes = unmatched_mask.nonzero() # num_dis, 2
        track_instances.matched_gt_idxes[unmatched_idxes[:, 0], unmatched_idxes[:, 1]] = -1
        num_disappear_track = unmatched_idxes.shape[0]

        prev_matched_idxes = matched_idxes #torch.cat([valid_track_mask.nonzero(), track_instances.matched_gt_idxes[valid_track_mask].unsqueeze(dim=-1)], dim=-1) # num_valid, 3

        # step2. select the unmatched track instances.
        # note that the FP tracks whose obj_ids are -2 will not be selected here.
        unmatched_track_idxes = ((track_instances.obj_ids == -1) * track_instances.valid_mask).nonzero() # num_unmatched_track, 2
        
        # step3. select the untracked gt instances (new tracks).
        tracked_gt_idxes = [torch.stack([torch.full((num_tracks,), b, dtype=torch.long, device=device), track_instances.matched_gt_idxes[b,:]], dim=-1) for b in range(bsz)]
        tracked_gt_idxes = torch.cat(tracked_gt_idxes, dim=0) # b*num_tracks, 2
        tracked_gt_idxes = tracked_gt_idxes[track_instances.valid_mask.reshape(-1)] # remove invalid tracks
        tracked_gt_idxes = tracked_gt_idxes[tracked_gt_idxes[:,1] != -1] # num_matched_gt, 2
        untracked_gt_mask = torch.full((bsz, num_gts,), True, dtype=torch.bool, device=device) # b, num_gts
        # if tracked_gt_idxes.numel() > 0 and (tracked_gt_idxes[:,1].min() < 0 or tracked_gt_idxes[:,1].max() >= num_gts):
        #     import pdb; pdb.set_trace()
        untracked_gt_mask[tracked_gt_idxes[:,0], tracked_gt_idxes[:,1]] = False # b, num_gts
        untracked_gt_idxes = (untracked_gt_mask & gt_instances_i.valid_mask).nonzero() # num_unmatched_gt, 2
        

        def match_for_single_decoder_layer(
                outputs_, targets_, matcher, 
                outputs_batch_index, targets_batch_index, batch_size,
                outputs_index_map=None, targets_index_map=None,
            ):
            new_match_indices = matcher(
                    outputs=outputs_, 
                    targets=targets_,
                    outputs_batch_index=outputs_batch_index,
                    targets_batch_index=targets_batch_index,
                    batch_size=batch_size
                )  # list[tuple(src_idx, tgt_idx)
            
            if outputs_index_map is not None or targets_index_map is not None:
                new_match_indices_ = []
                for bi in range(batch_size):
                    src_idx = outputs_index_map[new_match_indices[bi][0]] if outputs_index_map is not None else new_match_indices[bi][0]
                    tgt_idx = targets_index_map[new_match_indices[bi][1]] if targets_index_map is not None else new_match_indices[bi][1]
                    new_match_indices_.append((src_idx, tgt_idx))
                new_match_indices = new_match_indices_

            new_match_indices = [torch.stack([torch.full((idx[0].shape[0],), bi).to(idx[0]), idx[0], idx[1]], dim=1) for bi, idx in enumerate(new_match_indices)]

            return torch.cat(new_match_indices, dim=0)


        # step4. do matching between the unmatched track instances and gt instances.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes[:,0], unmatched_track_idxes[:,1]],
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes[:,0], unmatched_track_idxes[:,1]],
        }
        unmatch_targets = {
            'labels': gt_instances_i.labels[untracked_gt_idxes[:, 0], untracked_gt_idxes[:, 1]],
            'boxes': gt_instances_i.boxes[untracked_gt_idxes[:, 0], untracked_gt_idxes[:, 1]],
        }

        # if (unmatched_outputs['pred_boxes'] != unmatched_outputs['pred_boxes']).sum() > 0:
        #     import pdb; pdb.set_trace()

        new_matched_idxes = match_for_single_decoder_layer(
                                    outputs_=unmatched_outputs, 
                                    targets_=unmatch_targets,
                                    matcher=self.matcher,
                                    outputs_batch_index=unmatched_track_idxes[:,0],
                                    targets_batch_index=untracked_gt_idxes[:,0],
                                    outputs_index_map=unmatched_track_idxes[:,1],
                                    targets_index_map=untracked_gt_idxes[:,1],
                                    batch_size=bsz)

        # step5. update obj_ids according to the new matching result.
        track_instances.obj_ids[new_matched_idxes[:,0], new_matched_idxes[:,1]] = gt_instances_i.obj_ids[new_matched_idxes[:,0], new_matched_idxes[:,2]]
        track_instances.matched_gt_idxes[new_matched_idxes[:,0], new_matched_idxes[:,1]] = new_matched_idxes[:, 2]
    
        # step6. calculate iou.
        active_idxes = (track_instances.obj_ids >= 0) & (track_instances.matched_gt_idxes >= 0) & track_instances.valid_mask # b, num_tracks
        active_idxes = active_idxes.nonzero() # num_active, 2
        active_track_boxes = track_instances.pred_boxes[active_idxes[:,0], active_idxes[:, 1]] # num_active, 4
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[active_idxes[:,0], track_instances.matched_gt_idxes[active_idxes[:, 0], active_idxes[:, 1]]] # num_active, 4
            active_track_boxes = box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes[:, 0], active_idxes[:, 1]] = matched_boxlist_iou(active_track_boxes, gt_boxes)

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_idxes, prev_matched_idxes], dim=0)

        # step8. calculate losses.
        self.num_samples += (gt_instances_i.valid_mask.sum() + num_disappear_track)
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=gt_instances_i,
                                           indices=matched_indices,
                                           num_boxes=1,
                                           query_valid_mask=track_instances.valid_mask)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        # step 9: comput loss for dn, aux, query selection and so on
        all_track_idxes = track_instances.valid_mask.nonzero() # num_track, 2
        all_gt_idxes = gt_instances_i.valid_mask.nonzero() # num_gt, 2
        all_targets = {
            'labels': gt_instances_i.labels[all_gt_idxes[:, 0], all_gt_idxes[:, 1]],
            'boxes': gt_instances_i.boxes[all_gt_idxes[:, 0], all_gt_idxes[:, 1]],
        }

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if self.dec_layers_for_track is None or i in self.dec_layers_for_track:
                    unmatched_outputs_layer = {
                        'pred_logits': aux_outputs['pred_logits'][unmatched_track_idxes[:,0], unmatched_track_idxes[:,1]],
                        'pred_boxes': aux_outputs['pred_boxes'][unmatched_track_idxes[:,0], unmatched_track_idxes[:,1]],
                    }
                    new_matched_idxes_layer = match_for_single_decoder_layer(
                                                    outputs_=unmatched_outputs_layer, 
                                                    targets_=unmatch_targets,
                                                    matcher=self.matcher,
                                                    outputs_batch_index=unmatched_track_idxes[:,0],
                                                    targets_batch_index=untracked_gt_idxes[:,0],
                                                    outputs_index_map=unmatched_track_idxes[:,1],
                                                    targets_index_map=untracked_gt_idxes[:,1],
                                                    batch_size=bsz)
                    matched_idexes_layer = torch.cat([new_matched_idxes_layer, prev_matched_idxes], dim=0)
                else:
                    outputs_layer = {
                        'pred_logits': aux_outputs['pred_logits'][all_track_idxes[:,0], all_track_idxes[:,1]],
                        'pred_boxes': aux_outputs['pred_boxes'][all_track_idxes[:,0], all_track_idxes[:,1]],
                    }
                    matched_idexes_layer = match_for_single_decoder_layer(
                                                    outputs_=outputs_layer, 
                                                    targets_=all_targets,
                                                    matcher=self.matcher,
                                                    outputs_batch_index=all_track_idxes[:,0],
                                                    targets_batch_index=all_gt_idxes[:,0],
                                                    outputs_index_map=all_track_idxes[:,1],
                                                    targets_index_map=all_gt_idxes[:,1],
                                                    batch_size=bsz)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           outputs=aux_outputs,
                                           gt_instances=gt_instances_i,
                                           indices=matched_idexes_layer,
                                           num_boxes=1,
                                           query_valid_mask=track_instances.valid_mask)
                    self.losses_dict.update({'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in l_dict.items()})
        
        if 'query_selection_aux_outputs' in outputs:
            selected_query_mask = torch.full((bsz, outputs['query_selection_aux_outputs']['pred_logits'].shape[1]), True, dtype=torch.bool, device=device)
            selected_query_indxes = selected_query_mask.nonzero() 
            query_selection_aux_outputs = {
                'pred_logits': outputs['query_selection_aux_outputs']['pred_logits'][selected_query_indxes[:,0], selected_query_indxes[:,1]],
                'pred_boxes': outputs['query_selection_aux_outputs']['pred_boxes'][selected_query_indxes[:,0], selected_query_indxes[:,1]]
            }
            matched_idxes_query_selection = match_for_single_decoder_layer(
                                                outputs_=query_selection_aux_outputs, 
                                                targets_=all_targets,
                                                matcher=self.matcher,
                                                outputs_batch_index=selected_query_indxes[:,0],
                                                targets_batch_index=all_gt_idxes[:,0],
                                                outputs_index_map=selected_query_indxes[:,1],
                                                targets_index_map=all_gt_idxes[:,1],
                                                batch_size=bsz)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                l_dict = self.get_loss(loss,
                                        outputs=outputs['query_selection_aux_outputs'],
                                        gt_instances=gt_instances_i,
                                        indices=matched_idxes_query_selection,
                                        num_boxes=1)
                self.losses_dict.update({'frame_{}_aux_qs_{}'.format(self._current_frame_idx, key): value for key, value in l_dict.items()})
        
        if 'dn_aux_outputs' in outputs:
            def get_cdn_matched_indices(dn_meta, targets):
                '''get_cdn_matched_indices
                '''
                dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
                num_gts = targets.valid_mask.sum(dim=-1).tolist() 
                device = targets.labels.device
                
                dn_match_indices = []
                for i, num_gt in enumerate(num_gts):
                    if num_gt > 0:
                        gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                        gt_idx = gt_idx.tile(dn_num_group)
                        assert len(dn_positive_idx[i]) == len(gt_idx)
                        dn_match_indices.append((torch.full((len(gt_idx),), i).to(gt_idx), dn_positive_idx[i], gt_idx))
                    else:
                        dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                                                 torch.zeros(0, dtype=torch.int64, device=device), \
                                                 torch.zeros(0, dtype=torch.int64, device=device)))
                dn_match_indices = [torch.stack(idx, dim=-1) for idx in dn_match_indices]
                return torch.cat(dn_match_indices)

            assert 'dn_meta' in outputs, 'dn meta is required for computing the loss'
            indices = get_cdn_matched_indices(outputs['dn_meta'], gt_instances_i)
            num_boxes = 1 * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue

                    l_dict = self.get_loss(loss, 
                                            outputs=aux_outputs, 
                                            gt_instances=gt_instances_i, 
                                            indices=indices, 
                                            num_boxes=num_boxes)
                    # l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    self.losses_dict.update({'frame_{}_aux_dn{}_{}'.format(self._current_frame_idx, i, key): value for key, value in l_dict.items()})
        elif outputs.get('trained_with_dn', True): # the second frame sometimes has no gt boxes, and dn will not be performed. We need to initialize these losses for DDP training
            l_dict = {}
            for i in range(outputs['decoder_num_layers']):
                for loss in ['loss_bbox', 'loss_label', 'loss_giou']:
                    l_dict['frame_{}_aux_dn{}_{}'.format(self._current_frame_idx, i, loss)] = (pred_boxes_i * 0).mean().detach()
            self.losses_dict.update(l_dict)
        
        self._step()
        
        return track_instances

    def forward(self, outputs, targets=None):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            # losses[loss_name] /= num_samples
            losses[loss_name] = losses[loss_name] / num_samples

            if loss_name.endswith('loss_label'):
                losses[loss_name] = losses[loss_name] * self.weight_dict['loss_label']
            elif loss_name.endswith('loss_bbox'):
                losses[loss_name] = losses[loss_name] * self.weight_dict['loss_bbox']
            elif loss_name.endswith('loss_giou'):
                losses[loss_name] = losses[loss_name] * self.weight_dict['loss_giou']
            else:
                raise RuntimeError('Unknown type of loss: {}'.format(loss_name))

        return losses



@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res