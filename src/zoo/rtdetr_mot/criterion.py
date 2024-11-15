import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from src.core import register
from src.zoo.rtdetr.rtdetr_criterion import SetCriterion
from src.misc.instances import Instances
from src.misc.boxes import Boxes, matched_boxlist_iou
from src.misc.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from src.misc.dist import is_dist_available_and_initialized, get_world_size
import torchvision
from typing import List



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
        self.weight_dict = weight_dict
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma

        self.use_focal_loss = use_focal_loss
        self.losses_dict = {}
        self._current_frame_idx = 0
        self.dec_layers_for_track = dec_layers_for_track

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
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

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'labels_vfl': self.loss_labels_vfl,
            # 'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))

        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes[mask]),
            box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)  # set the class id of backgound query to be the number of classes
        # The matched gt for disappear track query is set -1.
        target_classes_o = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.zeros_like(J) + self.num_classes
            # set labels of track-appear instances to negative samples.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            target_classes_o.append(labels_per_img)
        target_classes_o = torch.cat(target_classes_o)
        target_classes[idx] = target_classes_o
        
        if self.use_focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            # loss_ce = sigmoid_focal_loss(src_logits.flatten(1),
            #                                  gt_labels_target.flatten(1),
            #                                  alpha=self.alpha,
            #                                  gamma=self.gamma,
            #                                  num_boxes=num_boxes, mean_in_dim1=False)
            # loss_ce = loss_ce.sum()
            loss_ce = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none') # B x N x C
            loss_ce = loss_ce.mean(dim=1).sum() * src_logits.shape[1] / num_boxes
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_label': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_labels_vfl(self, outputs, gt_instances, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx] # num_match x 4
        target_boxes = []
        target_idx = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            # set labels of track-appear instances to negative samples.
            target_idx.append(J)
            target_boxes_ = torch.zeros(len(J), 4).to(src_boxes)
            if len(gt_per_img) > 0 and len(J[J!=-1]) > 0:
                # The matched gt for disappear track query is set -1.
                target_boxes_[J[J!=-1]] = gt_per_img.boxes[J[J != -1]]
            target_boxes.append(target_boxes_)

        target_idx = torch.cat(target_idx)
        target_boxes = torch.cat(target_boxes)
        target_boxes[target_idx==-1] = src_boxes[target_idx==-1] # disapperaed tracks is confirmed, so its iou weight should be 1

        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach() # num_match

        src_logits = outputs['pred_logits']
        target_classes_o = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.zeros_like(J) + self.num_classes
            # set labels of track-appear instances to negative samples.
            if len(gt_per_img) > 0: # The matched gt for disappear track query is set -1.
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            target_classes_o.append(labels_per_img)
        target_classes_o = torch.cat(target_classes_o)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)     
        target_classes[idx] = target_classes_o
        
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_label': loss}



    def match_for_single_frame(self, outputs: dict):
        outputs_without_aux = {k: v for k, v in outputs.items() if ('aux_outputs' not in k and 'dn_' not in k)}
        # import pdb; pdb.set_trace()

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.

        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_boxes': pred_boxes_i.unsqueeze(0),
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        for j in range(len(track_instances)):
            obj_id = track_instances.obj_ids[j].item()
            # set new target idx.
            if obj_id >= 0:
                if obj_id in obj_idx_to_gt_idx:
                    track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
                else:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[j] = -1  # track-disappear case.
            else:
                track_instances.matched_gt_idxes[j] = -1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(pred_logits_i.device)
        matched_track_idxes = (track_instances.obj_ids >= 0)  # occu 
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(
            pred_logits_i.device)

        # step2. select the unmatched track instances.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_ids == -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]
        tgt_state = torch.zeros(len(gt_instances_i)).to(pred_logits_i.device)
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i)).to(pred_logits_i.device)[tgt_state == 0]
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]

        def match_for_single_decoder_layer(outputs_, gt_instance_, matcher, output_index_map=None, gt_index_map=None):

            new_track_indices = matcher(outputs_, [gt_instance_])  # list[tuple(src_idx, tgt_idx)]

            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            
            if output_index_map is not None:
                src_idx = output_index_map[src_idx]
            
            if gt_index_map is not None:
                tgt_idx = gt_index_map[tgt_idx]

            new_matched_indices = torch.stack([src_idx, tgt_idx], dim=1).to(pred_logits_i.device)

            return new_matched_indices


        # step4. do matching between the unmatched track instances and gt instances.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
        }
        new_matched_indices = match_for_single_decoder_layer(
                                    outputs_=unmatched_outputs, 
                                    gt_instance_=untracked_gt_instances,
                                    output_index_map=unmatched_track_idxes,
                                    gt_index_map=untracked_tgt_indexes,
                                    matcher=self.matcher)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_ids[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_ids >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if self.dec_layers_for_track is None or i in self.dec_layers_for_track:
                    unmatched_outputs_layer = {
                        'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                        'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                    }
                    new_matched_indices_layer = match_for_single_decoder_layer(
                                                    outputs_=unmatched_outputs_layer, 
                                                    gt_instance_=untracked_gt_instances,
                                                    output_index_map=unmatched_track_idxes,
                                                    gt_index_map=untracked_tgt_indexes,
                                                    matcher=self.matcher)
                    matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                else:
                    matched_indices_layer = match_for_single_decoder_layer(
                                                outputs_=aux_outputs, 
                                                gt_instance_=gt_instances_i,
                                                output_index_map=None,
                                                gt_index_map=None,
                                                matcher=self.matcher)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update({'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in l_dict.items()})
        
        if 'query_selection_aux_outputs' in outputs:
            matched_indices_query_selection = match_for_single_decoder_layer(
                                                    outputs_=outputs['query_selection_aux_outputs'], 
                                                    gt_instance_=gt_instances_i,
                                                    output_index_map=None,
                                                    gt_index_map=None,
                                                    matcher=self.matcher)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                l_dict = self.get_loss(loss,
                                        outputs['query_selection_aux_outputs'],
                                        gt_instances=[gt_instances_i],
                                        indices=[(matched_indices_query_selection[:, 0], matched_indices_query_selection[:, 1])],
                                        num_boxes=1)
                self.losses_dict.update({'frame_{}_aux_qs_{}'.format(self._current_frame_idx, key): value for key, value in l_dict.items()})
        
        if 'dn_aux_outputs' in outputs:
            def get_cdn_matched_indices(dn_meta, targets):
                '''get_cdn_matched_indices
                '''
                dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
                num_gts = [len(t.labels) for t in targets]
                device = targets[0].labels.device
                
                dn_match_indices = []
                for i, num_gt in enumerate(num_gts):
                    if num_gt > 0:
                        gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                        gt_idx = gt_idx.tile(dn_num_group)
                        assert len(dn_positive_idx[i]) == len(gt_idx)
                        dn_match_indices.append((dn_positive_idx[i], gt_idx))
                    else:
                        dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                            torch.zeros(0, dtype=torch.int64,  device=device)))
        
                return dn_match_indices

            assert 'dn_meta' in outputs, 'dn meta is required for computing the loss'
            indices = get_cdn_matched_indices(outputs['dn_meta'], [gt_instances_i])
            num_boxes = 1 * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, 
                                            aux_outputs, 
                                            gt_instances=[gt_instances_i], 
                                            indices=indices, 
                                            num_boxes=num_boxes)
                    # l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    self.losses_dict.update({'frame_{}_aux_dn{}_{}'.format(self._current_frame_idx, i, key): value for key, value in l_dict.items()})
        else: # the second frame sometimes has no gt boxes, and dn will not be performed. We need to initialize these losses for DDP training
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