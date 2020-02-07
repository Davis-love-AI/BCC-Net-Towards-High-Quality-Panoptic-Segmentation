import random
import torch

from .panoptic_htc import PanopticHTC
from .test_mixins import MaskTestMixin, OcclusionTestMixin, SRMTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import (bbox2roi, bbox2result, merge_aug_masks)


@DETECTORS.register_module
class PanopticHTCOcclusion(PanopticHTC, MaskTestMixin, OcclusionTestMixin):

    def __init__(self,
                 occ_head,
                 **kwargs):
        super(PanopticHTCOcclusion, self).__init__(**kwargs)
        assert occ_head is not None

        self.occ_head = builder.build_head(occ_head)
        self.occ_head.init_weights()

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      gt_occ=None,
                      proposals=None):

        losses = dict()

        # Accelerate only label mode
        if self.occ_head.only_label:
            assert self.with_occ
            gt_bboxes_list = []
            gt_label_list = []
            target_binary_list = []

            num_imgs = img.size(0)
            for i in range(num_imgs):
                gt_bboxes_list.append([])  # list of lists

                for occ in gt_occ[i]:
                    gt_ind_i, gt_ind_j, gt_ind_top = occ
                    if gt_ind_i == -1 or gt_ind_j == -1:  # iscrowd
                        continue
                    assert gt_ind_i != gt_ind_j
                    gt_ind_bottom = gt_ind_i + gt_ind_j - gt_ind_top

                    # add gt to training
                    ind_target_list = [(gt_ind_top, gt_ind_bottom, torch.ones(1)),
                                       (gt_ind_bottom, gt_ind_top, torch.zeros(1))]
                    random.shuffle(ind_target_list)
                    for ind1, ind2, target_binary in ind_target_list:  # index of gt
                        gt_bboxes_list[i] += [gt_bboxes[i][ind1], gt_bboxes[i][ind2]]
                        gt_label_list += [gt_labels[i][ind1], gt_labels[i][ind2]]
                        target_binary_list.append(target_binary)

            if len(target_binary_list) != 0:
                # occlusion pairs found
                num_pairs = len(target_binary_list)  # n
                gt_label_tensor_list = torch.tensor(gt_label_list).float().cuda()
                gt_label_tensor_list = gt_label_tensor_list.view(num_pairs, 2)

                # detach
                gt_label_tensor_list = gt_label_tensor_list.detach()

                occ_preds = self.occ_head(None, None,
                                          gt_label_tensor_list)  # (pairs, 512, 14, 14) (pairs, 2, 28, 28) (pairs, 2)
                loss_occ = self.occ_head.loss(occ_preds.view(-1), torch.cat(target_binary_list).cuda())
            else:
                occ_pred = self.occ_head(None,  None, torch.ones(1, 2).cuda())
                loss_occ = self.occ_head.loss(occ_pred, torch.zeros(1).cuda())
                loss_occ['loss_occ'] = loss_occ['loss_occ'] * 0.0

            losses.update(loss_occ)
            return losses

        x = self.extract_feat(img)

        mask_head = self.mask_head[-1]
        mask_roi_extractor = self.mask_roi_extractor[-1]
        assert self.with_semantic and 'mask' in self.semantic_fusion
        semantic_pred, semantic_feat = self.semantic_head(x)

        assert self.with_occ
        gt_bboxes_list = []
        gt_label_list = []
        target_binary_list = []

        num_imgs = img.size(0)
        for i in range(num_imgs):
            gt_bboxes_list.append([])  # list of lists

            for occ in gt_occ[i]:
                gt_ind_i, gt_ind_j, gt_ind_top = occ
                if gt_ind_i == -1 or gt_ind_j == -1:  # iscrowd
                    continue
                assert gt_ind_i != gt_ind_j
                gt_ind_bottom = gt_ind_i + gt_ind_j - gt_ind_top

                # add gt to training
                ind_target_list = [(gt_ind_top, gt_ind_bottom, torch.ones(1)),
                                   (gt_ind_bottom, gt_ind_top, torch.zeros(1))]
                random.shuffle(ind_target_list)
                for ind1, ind2, target_binary in ind_target_list:  # index of gt
                    gt_bboxes_list[i] += [gt_bboxes[i][ind1], gt_bboxes[i][ind2]]
                    gt_label_list += [gt_labels[i][ind1], gt_labels[i][ind2]]
                    target_binary_list.append(target_binary)

        if len(target_binary_list) != 0:
            # occlusion pairs found
            gt_bboxes_tensor_list = [(torch.stack(bboxes) if len(bboxes) else torch.zeros((0, 5)).cuda()) for bboxes in gt_bboxes_list]
            multi_rois = bbox2roi(gt_bboxes_tensor_list)
            multi_feats = mask_roi_extractor(
                x[:mask_roi_extractor.num_inputs], multi_rois)  # (n*2, 256, 14, 14) for mask

            # added with semantic_feat
            multi_semantic_feats = self.semantic_roi_extractor(
                [semantic_feat], multi_rois)
            assert multi_semantic_feats.size()[-2:] == multi_feats.size()[-2:]
            multi_feats += multi_semantic_feats

            multi_pred = mask_head(multi_feats, return_feat=False)  # (n*2, 81, 28, 28) for mask
            # find specific class by gt_label
            multi_bin_pred = multi_pred[range(multi_pred.size()[0]), gt_label_list]  # (n*2, 28, 28)

            num_pairs = len(target_binary_list)  # n
            multi_feats = multi_feats.view(num_pairs, 2, 256, 14, 14).view(num_pairs, 512, 14, 14)
            multi_bin_pred = multi_bin_pred.view(num_pairs, 2, 28, 28)
            gt_label_tensor_list = torch.tensor(gt_label_list).float().cuda()
            gt_label_tensor_list = gt_label_tensor_list.view(num_pairs, 2)

            # detach
            multi_feats = multi_feats.detach()
            multi_bin_pred = multi_bin_pred.detach()
            gt_label_tensor_list = gt_label_tensor_list.detach()

            occ_preds = self.occ_head(multi_feats, multi_bin_pred, gt_label_tensor_list)  # (pairs, 512, 14, 14) (pairs, 2, 28, 28) (pairs, 2)
            loss_occ = self.occ_head.loss(occ_preds.view(-1), torch.cat(target_binary_list).cuda())
        else:
            # fake it
            two_rois = bbox2roi([torch.stack([gt_bboxes[0][0],
                                              gt_bboxes[0][0]], dim=0)])
            two_mask_feats = mask_roi_extractor(
                x[:mask_roi_extractor.num_inputs], two_rois)
            two_mask_pred = mask_head(two_mask_feats, return_feat=False)
            two_binary_mask_pred = torch.stack([two_mask_pred[0][gt_labels[0][0]], two_mask_pred[0][gt_labels[0][0]]], dim=0)

            # detach
            two_mask_feats = two_mask_feats.detach()
            two_mask_feats = two_mask_feats.view(1, 512, 14, 14)
            two_binary_mask_pred = two_binary_mask_pred.detach()
            two_binary_mask_pred = two_binary_mask_pred.view(1, 2, 28, 28)

            occ_pred = self.occ_head(two_mask_feats, two_binary_mask_pred, torch.ones(1, 2).cuda())
            loss_occ = self.occ_head.loss(occ_pred, torch.zeros(1).cuda())
            loss_occ['loss_occ'] = loss_occ['loss_occ'] * 0.0

        losses.update(loss_occ)
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x[:len(self.semantic_head.featmap_strides)])
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
                i, x, rois, semantic_feat=semantic_feat)
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
                            i, x, _bboxes, semantic_feat=semantic_feat)
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
                occ_matrix = []
                occ_ind_map = {}
            else:
                _bboxes = (
                    det_bboxes[:, :4] * scale_factor
                    if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
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

                # occ
                # ========== occ ==========
                assert self.with_occ
                num_instances = det_labels.size()[0]
                assert det_bboxes.size()[0] == num_instances

                scores = det_bboxes[:, 4].cpu().numpy()
                indices = list(range(num_instances))
                indices.sort(key=lambda ind: scores[ind])
                indices.reverse()
                old2_ranking = [0] * len(indices)
                for rank, old in enumerate(indices):
                    old2_ranking[old] = rank

                occ_ind_map = {}
                new_ind = -1
                for label in range(80):
                    for old_ind in range(num_instances):
                        if det_labels[old_ind] == label:
                            new_ind += 1
                            occ_ind_map[old2_ranking[old_ind]] = new_ind  # map score ranking to new_ind

                merged_mask_pred = torch.from_numpy(merged_masks).cuda()
                aligned_masks = self.get_aligned_masks(
                    merged_mask_pred, _bboxes, det_labels, rcnn_test_cfg, img_shape=img_meta[0]['img_shape'])
                assert aligned_masks.size()[0] == num_instances

                occ_matrix = self.simple_test_occ(det_bboxes, mask_feats, merged_mask_pred, aligned_masks, det_labels, old2_ranking)
                occ_matrix = occ_matrix.tolist()
                # ========== occ ==========

            ms_segm_result['ensemble'] = segm_result

        assert self.with_mask and self.with_semantic and not self.test_cfg.keep_all_stages
        assert self.with_occ
        aux_results = dict(
            occ_results=dict(
                matrix=occ_matrix,
                ind_map=occ_ind_map))
        return ms_bbox_result['ensemble'], ms_segm_result['ensemble'], semantic_result, aux_results


@DETECTORS.register_module
class PanopticHTCOcclusionSRM(PanopticHTCOcclusion, SRMTestMixin):

    def __init__(self, srm_head, **kwargs):
        super(PanopticHTCOcclusionSRM, self).__init__(**kwargs)
        assert srm_head is not None
        self.srm_head = builder.build_head(srm_head)
        self.srm_head.init_weights()

    def forward_train(self,
                      img,
                      img_meta,
                      gt_srm_inputs=None,
                      gt_semantic_80=None,
                      **kwargs):
        losses = super(PanopticHTCOcclusionSRM, self).forward_train(img, img_meta, **kwargs)

        assert self.with_srm
        srm_pred = self.srm_head(gt_srm_inputs)
        loss_srm = self.srm_head.loss(srm_pred, gt_semantic_80)
        losses.update(loss_srm)
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x[:len(self.semantic_head.featmap_strides)])
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
                i, x, rois, semantic_feat=semantic_feat)
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
                            i, x, _bboxes, semantic_feat=semantic_feat)
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
                occ_matrix = []
                occ_ind_map = {}
            else:
                _bboxes = (
                    det_bboxes[:, :4] * scale_factor
                    if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
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

                # srm
                assert self.with_srm
                merged_mask_pred = torch.from_numpy(merged_masks).cuda()
                aligned_masks = self.get_aligned_masks(
                    merged_mask_pred, _bboxes, det_labels, rcnn_test_cfg, img_shape=img_meta[0]['img_shape'])
                srm_results_flat = self.simple_test_srm(aligned_masks, det_labels, return_flat=True)

                # occ
                # ========== occ ==========
                assert self.with_occ
                num_instances = det_labels.size()[0]
                assert det_bboxes.size()[0] == num_instances

                indices = list(range(num_instances))
                indices.sort(key=lambda ind: srm_results_flat[ind])  # sorted by srm scores
                indices.reverse()
                old2_ranking = [0] * len(indices)
                for rank, old in enumerate(indices):
                    old2_ranking[old] = rank

                occ_ind_map = {}
                new_ind = -1
                for label in range(80):
                    for old_ind in range(num_instances):
                        if det_labels[old_ind] == label:
                            new_ind += 1
                            occ_ind_map[old2_ranking[old_ind]] = new_ind  # map score ranking to new_ind

                assert aligned_masks.size()[0] == num_instances

                occ_matrix = self.simple_test_occ(mask_feats, merged_mask_pred, aligned_masks, det_labels, old2_ranking)
                occ_matrix = occ_matrix.tolist()
                # ========== occ ==========

            ms_segm_result['ensemble'] = segm_result

        assert self.with_mask and self.with_semantic and not self.test_cfg.keep_all_stages
        assert self.with_occ
        aux_results = dict(
            occ_results=dict(
                matrix=occ_matrix,
                ind_map=occ_ind_map))
        return ms_bbox_result['ensemble'], ms_segm_result['ensemble'], semantic_result, aux_results