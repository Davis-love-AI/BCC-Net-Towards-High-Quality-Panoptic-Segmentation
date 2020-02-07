import torch
import torch.nn.functional as F
import mmcv
import numpy as np
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_proposals,
                        merge_aug_bboxes, merge_aug_masks, merge_aug_semantics, multiclass_nms)
from mmdet.datasets.transforms import SRMInputTransform
from mmdet.datasets.utils import to_tensor
iter_cnt = 0
from scipy import ndimage
class RPNTestMixin(object):

    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # reorganize the order of 'img_metas' to match the dimensions
        # of 'aug_proposals'
        aug_img_metas = []
        for i in range(imgs_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, rpn_test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals


class BBoxTestMixin(object):

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin(object):

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False,
                         return_aligned_masks=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
            if return_aligned_masks:
                aligned_masks = torch.zeros((0, *img_shape[:2])).cuda()
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(
                mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, rescale)

            if return_aligned_masks:
                aligned_masks = self.get_aligned_masks(
                    mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, img_shape)
        if return_aligned_masks:
            return segm_result, aligned_masks
        else:
            return segm_result

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                mask_pred = self.mask_head(mask_feats)
                # convert to numpy array to save memory
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas,
                                           self.test_cfg.rcnn)

            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg.rcnn,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_result

    def get_aligned_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, img_shape):  # tensor operation used
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor / np.ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            img_shape: scaled image size (rounded to int)

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, np.ndarray):
            mask_pred = torch.from_numpy(mask_pred).cuda()
        assert isinstance(mask_pred, torch.Tensor)

        mask_pred = mask_pred.sigmoid()
        bboxes = det_bboxes[:, :4]
        labels = det_labels + 1

        # make inputs for srm
        num_instance = bboxes.size()[0]
        im_masks_shape = (num_instance, *img_shape[:2])
        im_masks = torch.zeros(im_masks_shape).cuda()
        for i in range(num_instance):
            bbox = (bboxes[i, :]).long()  # convert to int, remove fraction part
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            # for mask_head: assert not class_agnostic
            mask_pred_ = mask_pred[i, label, :, :]

            bbox_mask = F.interpolate(mask_pred_[None, None, :, :], size=(h, w), mode='bilinear', align_corners=True)[0, 0]
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary)
            im_masks[i][bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

        return im_masks


# panoptic htc
class SemanticTestMixin(object):

    def simple_test_semantic(self,
                             semantic_pred,
                             img_meta,
                             rescale=False):
        """ Only be called during val & test
            semantic_pred.size() is like pad_shape  e.g. (800, 1280, 3)
            ori_shape is img's original shape from the png  e.g. (406, 640, 3)
            img_shape is img's shape, after scaled by (1333, 800), i.e. scaled by scale_factor and rounded.  e.g. (800, 1261, 3)
        """
        ori_shape = img_meta[0]['ori_shape']
        img_shape = img_meta[0]['img_shape']

        assert rescale is True

        if isinstance(semantic_pred, torch.Tensor):
            semantic_pred = semantic_pred[..., :img_shape[0], :img_shape[1]].softmax(
                dim=1).cpu().numpy()  # crop upper-left corner
        assert isinstance(semantic_pred, np.ndarray)

        # Semantic Unknown Erasing
        semantic_pred_cls = self.unknown_erase(semantic_pred)

        img_h, img_w = ori_shape[:2]
        semantic_result = mmcv.imresize(semantic_pred_cls.transpose(1, 2, 0), (img_w, img_h), interpolation='nearest')
        return semantic_result

    def aug_test_seg(self, feats, img_metas):
        # TODO
        pass

    def unknown_erase(self, semantic_pred, quality_threshold=0.5, iters=50):
        '''
        Algorithm for the Unknown Erasing
        :param semantic_pred: the output from semantic head (NxCxHxW)
        :param quality_threshold: the threshold for segment score
        :param iters: iterations for the morphology operation
        :return semantic_pred_cls: the final semantic segmentation result (NxHxW)
        '''

        # obtain the pixel-wise class prediction
        semantic_pred_cls = semantic_pred.argmax(axis=1)

        for i in range(semantic_pred.shape[0]):     # iterate the batch

            # find the classes appeared in the image
            pred_cls = np.unique(semantic_pred_cls[i])

            for j in range(len(pred_cls)):
                binmask = np.zeros_like(semantic_pred_cls[i])
                cls = pred_cls[j]
                binmask[semantic_pred_cls[i] == cls] = 1

                # dilate the binmask
                struct = ndimage.generate_binary_structure(2, 2)
                binmask_dilated = ndimage.binary_dilation(binmask, structure=struct, iterations=iters).astype(
                    binmask.dtype)

                # find the connected components
                labeled, nr_objects = ndimage.label(binmask_dilated)

                for k in range(1, nr_objects + 1):
                    ccmask = np.zeros_like(binmask)
                    ccmask[labeled == k] = 1

                    # remove the dilated pixels
                    ccmask = ccmask * binmask

                    # average the confidence scores
                    quality_score = np.sum(semantic_pred[i, cls, ...] * ccmask) / np.sum(ccmask)

                    # erase the low quality cc
                    if quality_score < quality_threshold:
                        inverse_ccmask = 1 - ccmask
                        semantic_pred_cls[i] *= inverse_ccmask

        return semantic_pred_cls


# srm
class SRMTestMixin(object):

    def simple_test_srm(self,
                        aligned_masks,
                        det_labels,
                        return_flat=False):  # tensor operation used
        """
        :param aligned_masks: tensor
        :param det_labels: np array (#instance,) (each number in this 1D array is within 0 ~ 79)
        :param return_flat:
        :return:
        """
        assert isinstance(aligned_masks, torch.Tensor)

        srm_inputs = SRMInputTransform()(aligned_masks, det_labels)
        srm_inputs = srm_inputs[None].float()  # (C, H, W) to (1, C, H, W)

        srm_score_pred = self.srm_head(srm_inputs, test_mode=True)
        srm_results_flat = self.srm_head.get_srm_avg_scores(srm_score_pred, aligned_masks, det_labels, srm_inputs[0])

        if return_flat:
            return srm_results_flat

        # split resulting List into List[List] (for later evaluation)
        if len(srm_results_flat) == 0:
            srm_results = [[] for label in range(80)]
        else:
            srm_results = [[avg_score for idx, avg_score in enumerate(srm_results_flat) if det_labels[idx] == label] for label in range(80)]
        return srm_results

    def aug_test_srm(self):
        # TODO
        pass


class OcclusionTestMixin(object):
    def simple_test_occ(self,det_bboxes, mask_feats, mask_pred, aligned_masks, det_labels, old2_ranking):
        """
        mask_feats, tensor of size (#instance, 256, 14, 14)
        mask_pred, tensor of size (#instance, 81, 28, 28)
        """
        global iter_cnt
        if self.occ_head.only_label and iter_cnt==0:
            self.vis()
            iter_cnt=1
        assert len(aligned_masks.size()) == 3
        num_instance = aligned_masks.size()[0]
        assert mask_feats.size() == (num_instance, 256, 14, 14)
        assert mask_pred.size() == (num_instance, 81, 28, 28)

        occ_matrix = torch.zeros((num_instance, num_instance)).cuda()
        mask_area = aligned_masks.view(aligned_masks.size()[0], -1).sum(dim=1)

        labels = det_labels + 1  # 1 ~ 80
        assert mask_area.size()[0] == labels.size()[0]  # #instance

        # different from original paper, matrix here contains 0 or 1 only.
        # Entries(i,j) are only valid if i > j
        for i in range(num_instance):
            for j in range(i):
                intersection_area = (aligned_masks[i] * aligned_masks[j]).sum()
                if mask_area[i] == 0 or mask_area[j] == 0:
                    continue
                if intersection_area / mask_area[i] < 0.2 and intersection_area / mask_area[j] < 0.2:
                    continue

                first_binary_mask_pred = mask_pred[i][labels[i]]
                second_binary_mask_pred = mask_pred[j][labels[j]]
                bbox_pair = (det_bboxes[i], det_bboxes[j])


                multi_bin_pred = torch.stack([first_binary_mask_pred, second_binary_mask_pred, second_binary_mask_pred, first_binary_mask_pred], dim=0)  # 2 * 28 * 28
                occ_preds = self.occ_head(mask_feats[[i, j, j, i]].view(2, 512, 14, 14), multi_bin_pred.view(2, 2, 28, 28), torch.tensor([[labels[i], labels[j]], [labels[j], labels[i]]]).float().cuda(),
                                          bbox_pair)

                occ_pred_pos, occ_pred_neg = occ_preds.sigmoid()

                rank_i, rank_j = old2_ranking[i], old2_ranking[j]
                occ_matrix[rank_i][rank_j] = int(occ_pred_pos > occ_pred_neg)
                occ_matrix[rank_j][rank_i] = int(occ_pred_pos < occ_pred_neg)

        return occ_matrix

    def vis(self):
        occ_matrix = np.zeros((81, 81))
        print("visualizing occ_matrix")
        # different from original paper, matrix here contains 0 or 1 only.
        # Entries(i,j) are only valid if i > j
        for i in range(80):
            for j in range(80):
                occ_preds = self.occ_head(None, None, torch.tensor([[i+1, j+1], [j+1, i+1]]).float().cuda())

                occ_pred_pos, occ_pred_neg = occ_preds.sigmoid()
                occ_matrix[i+1][j+1] = int(occ_pred_pos > occ_pred_neg)
                occ_matrix[j+1][i+1] = int(occ_pred_pos < occ_pred_neg)

        occ_matrix.dump('only_label_fc_output.dat')

    def aug_test_seg(self, feats, img_metas):
        # TODO
        pass
