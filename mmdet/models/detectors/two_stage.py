import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin, SemanticTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
from torch.autograd import grad
from torch.autograd import Function
from torch.nn import functional as F
import numpy as np
iters = 0
n_iters = 1833*14

@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin, SemanticTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 semantic_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        if semantic_head is not None:
            self.semantic_head = builder.build_head(semantic_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        print('Using Discriminator')
        self.tasks = ['instance', 'semantic']
        self.task_dict = {x: i for i, x in enumerate(self.tasks)}
        dscr_d = 2
        dscr_k = 1
        width_decoder = 256
        self.dscr_d = dscr_d
        self.dscr_k = dscr_k
        self.task_label_shape = (128, 128)
        self.discriminator = self._get_discriminator(width_decoder)
        self.rev_layer = ReverseLayerF()
        self.criterion_classifier = torch.nn.CrossEntropyLoss(ignore_index=255)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
        # pantoptic
        if self.with_semantic:
            self.semantic_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

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

        # RPN forward and loss
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

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        # panoptic
        if self.with_semantic:
            semantic_pred, _ = self.semantic_head(x)
            loss_semantic = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses["loss_semantic_seg"] = loss_semantic

        # GBFPN
        global iters, n_iters
        alpha = 2. / (1. + np.exp(-10 * ((iters + 1) / n_iters))) - 1
        iters += 1
        loss_rpn = sum(rpn_losses['loss_rpn_cls']) + sum(rpn_losses['loss_rpn_bbox'])
        loss_dscr_instance = self.compute_dscr_loss(loss_mask['loss_mask']+loss_bbox['loss_bbox']+loss_bbox['loss_cls']+loss_rpn, x, 'instance', alpha)
        loss_dscr_semantic = self.compute_dscr_loss(loss_semantic, x, 'semantic', alpha)
        losses["loss_dscr_instance"] = loss_dscr_instance * 0.1
        losses["loss_dscr_semantic"] = loss_dscr_semantic * 0.1

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
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)

            # panoptic
            if self.with_semantic:
                semantic_pred, _ = self.semantic_head(x)
                semantic_results = self.simple_test_semantic(
                    semantic_pred, img_meta, rescale=rescale)
                return bbox_results, segm_results, semantic_results
            else:
                return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):  # TODO: modify for semantic_head
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            if self.with_semantic:
                semantic_results = self.aug_test_semantic(
                    self.extract_feats(imgs), img_metas)
                return bbox_results, segm_results, semantic_results
            else:
                return bbox_results, segm_results
        else:
            return bbox_results

    def compute_dscr_loss(self, curr_loss, fpn_features, task, alpha):

        # take fpn result as input
        task_label = self._create_task_labels(fpn_features[0], task).to(fpn_features[0].device)
        input_dscr_list = []
        scale_factor = 1
        with torch.enable_grad():
            for features in fpn_features[:-1]:
                grads = grad(curr_loss, features, create_graph=True, allow_unused=True)[0]
                grads_norm = grads.norm(p=2, dim=1).unsqueeze(1) + 1e-10
                input_dscr_scale = grads / grads_norm
                input_dscr_scale = self.rev_layer.apply(input_dscr_scale, alpha)
                input_dscr_list.append(F.interpolate(input_dscr_scale, scale_factor=scale_factor, mode='nearest'))
                scale_factor *= 2
            input_dscr = torch.cat(input_dscr_list, dim=1)
            outputs_dscr = self.discriminator(input_dscr)
            losses_dscr = self.criterion_classifier(outputs_dscr, task_label)
        return losses_dscr

    def _get_discriminator(self, width_decoder):
        discriminator = FullyConvDiscriminator(in_channels=width_decoder*4, n_classes=len(self.tasks),
                                               kernel_size=self.dscr_k, depth=self.dscr_d)

        return discriminator

    def _create_task_labels(self, features, task):
        valid = torch.ones([features.shape[0], features.shape[2], features.shape[3]]) * self.task_dict[task]

        return valid.long()


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class FullyConvDiscriminator(nn.Module):
    def __init__(self, in_channels, n_classes, kernel_size=1, depth=1):
        super(FullyConvDiscriminator, self).__init__()

        padding = (kernel_size - 1) / 2
        assert (padding == int(padding))
        padding = int(padding)

        print('\nInitializing Fully Convolutional Discriminator with depth: {} and kernel size: {}'
              .format(depth, kernel_size))
        if depth == 1:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, n_classes, kernel_size=kernel_size, padding=padding, bias=True))
        elif depth == 2:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=False),
                # nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, n_classes, kernel_size=kernel_size, padding=padding, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x
