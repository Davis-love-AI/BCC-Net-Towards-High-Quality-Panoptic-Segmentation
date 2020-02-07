from .test_mixins import SRMTestMixin, MaskTestMixin
from .. import builder
from mmdet.core import (bbox2roi, bbox2result, merge_aug_masks)


from .panoptic_fpn import PanopticFPN
from .panoptic_htc import PanopticHTC
from ..registry import DETECTORS


@DETECTORS.register_module
class PanopticFPNSRM(PanopticFPN, SRMTestMixin):

    def __init__(self, srm_head, **kwargs):
        super(PanopticFPNSRM, self).__init__(**kwargs)
        assert srm_head is not None
        self.srm_head = builder.build_head(srm_head)
        self.srm_head.init_weights()

    def forward_train(self,
                      img,
                      img_meta,
                      gt_srm_inputs=None,
                      gt_semantic_80=None):
        losses = dict()
        # img and img_meta is useless here
        assert self.with_srm
        srm_pred = self.srm_head(gt_srm_inputs)
        loss_srm = self.srm_head.loss(srm_pred, gt_semantic_80)
        losses.update(loss_srm)
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            # masks
            segm_results, aligned_masks = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale, return_aligned_masks=True)

            # panoptic
            if self.with_semantic:
                semantic_pred, _ = self.semantic_head(x)
                semantic_results = self.simple_test_semantic(
                    semantic_pred, img_meta, rescale=rescale)

                assert self.with_srm
                if self.with_srm:
                    srm_results = self.simple_test_srm(aligned_masks, det_labels)
                    aux_results = dict(
                        srm_results=srm_results)
                    return bbox_results, segm_results, semantic_results, aux_results
                else:
                    return bbox_results, segm_results, semantic_results
            else:
                return bbox_results, segm_results


@DETECTORS.register_module
class PanopticHTCSRM(PanopticHTC, SRMTestMixin, MaskTestMixin):

    def __init__(self, srm_head, **kwargs):
        super(PanopticHTCSRM, self).__init__(**kwargs)
        assert srm_head is not None
        self.srm_head = builder.build_head(srm_head)
        self.srm_head.init_weights()

    def forward_train(self,
                      img,
                      img_meta,
                      gt_srm_inputs=None,
                      gt_semantic_80=None):
        losses = dict()
        # img and img_meta is useless here
        assert self.with_srm
        srm_pred = self.srm_head(gt_srm_inputs)
        loss_srm = self.srm_head.loss(srm_pred, gt_semantic_80)
        losses.update(loss_srm)
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
                srm_results = [
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
                if self.with_srm:
                    aligned_masks = self.get_aligned_masks(
                        merged_masks, _bboxes, det_labels, rcnn_test_cfg, img_shape=img_meta[0]['img_shape'])
                    srm_results = self.simple_test_srm(aligned_masks, det_labels)

            ms_segm_result['ensemble'] = segm_result

        assert self.with_mask and self.with_semantic and not self.test_cfg.keep_all_stages
        assert self.with_srm
        aux_results = dict(
            srm_results=srm_results)
        return ms_bbox_result['ensemble'], ms_segm_result['ensemble'], semantic_result, aux_results
