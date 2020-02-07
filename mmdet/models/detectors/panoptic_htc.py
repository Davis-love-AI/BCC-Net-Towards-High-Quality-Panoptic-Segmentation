import torch
import torch.nn.functional as F
from mmdet.core import (bbox2roi, bbox2result, build_assigner, build_sampler,
                        merge_aug_masks, merge_aug_masks_tensor)
from mmdet.models.detectors.test_mixins import SemanticTestMixin

from .htc import HybridTaskCascade
from .. import builder
from ..registry import DETECTORS
from mmdet.models.detectors.test_mixins import SemanticTestMixin
from mmdet.core import (bbox2roi, bbox2result, build_assigner, build_sampler,
                        merge_aug_masks)


@DETECTORS.register_module
class PanopticHTC(HybridTaskCascade, SemanticTestMixin):

    def __init__(self, **kwargs):
        super(PanopticHTC, self).__init__(**kwargs)
        if self.semantic_head is not None:
            self.semantic_head.init_weights()

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()

        # RPN part, the same as normal two-stage detectors
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        if self.with_semantic:
            # panoptic
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_semantic = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_semantic
        else:
            semantic_feat = None
        semantic_pred_cls = semantic_pred
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
            bbox_sampler = build_sampler(rcnn_train_cfg.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                    gt_labels[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss

            # SMP: semantic prediction to help bbox prediction
            loss_bbox, rois, bbox_targets, bbox_pred = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat, semantic_pred_cls)
            roi_labels = bbox_targets[0]

            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []
                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_bboxes[j],
                                gt_bboxes_ignore[j], gt_labels[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_labels[j],
                                feats=[lvl_feat[j][None] for lvl_feat in x])
                            sampling_results.append(sampling_result)
                # SMP: semantic prediction to help mask prediction
                loss_mask = self._mask_forward_train(i, x, sampling_results,
                                                     gt_masks, rcnn_train_cfg,
                                                     semantic_feat, semantic_pred_cls)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        # panoptic
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            semantic_result = self.simple_test_semantic(
                semantic_pred, img_meta, rescale=rescale)
        else:
            semantic_feat = None
            semantic_result = None

        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            cls_score, bbox_pred = self._bbox_forward_test(
                i, x, rois, semantic_feat=semantic_feat,
                semantic_pred=semantic_pred_cls)
            ms_scores.append(cls_score)

            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=rescale,
                    nms_cfg=rcnn_test_cfg)
                bbox_result = bbox2result(det_bboxes, det_labels,
                                          bbox_head.num_classes)
                ms_bbox_result['stage{}'.format(i)] = bbox_result

                if self.with_mask:
                    mask_head = self.mask_head[i]
                    if det_bboxes.shape[0] == 0:
                        segm_result = [
                            [] for _ in range(mask_head.num_classes - 1)
                        ]
                    else:
                        _bboxes = (
                            det_bboxes[:, :4] * scale_factor
                            if rescale else det_bboxes)
                        mask_pred = self._mask_forward_test(

                            i, x, _bboxes, semantic_feat=semantic_feat, semantic_pred=semantic_pred_cls)
                        segm_result = mask_head.get_seg_masks(
                            mask_pred, _bboxes, det_labels, rcnn_test_cfg,
                            ori_shape, scale_factor, rescale)
                    ms_segm_result['stage{}'.format(i)] = segm_result

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])

        cls_score = sum(ms_scores) / float(len(ms_scores))
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [
                    [] for _ in range(self.mask_head[-1].num_classes - 1)
                ]
            else:
                _bboxes = (
                    det_bboxes[:, :4] * scale_factor
                    if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                mask_pred_feat = self.mask_pred_roi_extractor([semantic_pred_cls], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                    mask_feats = torch.cat([mask_feats, mask_pred_feat], dim=1)

                last_feat = None
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_meta] * self.num_stages,
                                               self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result

        assert self.with_mask and self.with_semantic and not self.test_cfg.keep_all_stages
        return ms_bbox_result['ensemble'], ms_segm_result['ensemble'], semantic_result
