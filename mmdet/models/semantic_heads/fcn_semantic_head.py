import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import kaiming_init
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from math import log


class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsample_factor=2, conv_cfg=None, norm_cfg=None):
        super(UpsampleBlock, self).__init__()
        self.kernel_size = kernel_size
        self.upsample_factor = upsample_factor

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)  # initialised, and with relu as activation function

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


@HEADS.register_module
class FCNSemanticHead(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes,
                 featmap_strides,
                 loss_semantic,
                 ignore_label=0,
                 upsample_out_channels=256,
                 embedding_out_channels=256,
                 conv_cfg=None,
                 norm_cfg=None):
        super(FCNSemanticHead, self).__init__()

        self.num_scales = len(featmap_strides)
        self.featmap_strides = featmap_strides
        self.in_channels = in_channels
        self.upsample_out_channels = upsample_out_channels
        self.embedding_out_channels = embedding_out_channels
        self.num_classes = num_classes

        self.loss_semantic = build_loss(loss_semantic)
        self.ignore_label = ignore_label

        self.upsamples = nn.ModuleList()
        self.upsamples.append(
            ConvModule(
                self.in_channels,
                self.upsample_out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))

        for i in range(1, self.num_scales):
            num_blocks = int(log((self.featmap_strides[i] // 4), 2))

            blocks = []
            for j in range(0, num_blocks - 1):
                blocks.append(UpsampleBlock(in_channels, in_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
            blocks.append(UpsampleBlock(in_channels, upsample_out_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg))

            sequential = nn.Sequential(*blocks)
            self.upsamples.append(sequential)

        self.conv_logits = nn.Conv2d(self.upsample_out_channels, self.num_classes, 1)
        self.conv_embedding = ConvModule(self.upsample_out_channels, self.embedding_out_channels, 1)

    def init_weights(self):
        # ConvModule will initialize their weights automatically when created
        # kaiming_init(self.conv_logits)
        pass

    def forward(self, x):
        x = x[:len(self.featmap_strides)]
        assert len(x) == self.num_scales

        upsampled_feats = []
        for i, upsample in enumerate(self.upsamples):
            upsampled_feats.append(upsample(x[i]))

        for i in range(self.num_scales - 1, 0, -1):
            upsampled_feats[i - 1] = upsampled_feats[i - 1] + upsampled_feats[i]

        semantic_feat = upsampled_feats[0]
        semantic_pred = self.conv_logits(semantic_feat)
        semantic_pred = F.interpolate(semantic_pred, scale_factor=4, mode='bilinear', align_corners=True)

        semantic_feat = self.conv_embedding(semantic_feat)
        return semantic_pred, semantic_feat  # scale 1/1, 1/4

    def loss(self, semantic_pred, gt_semantic_seg):
        gt_semantic_seg = gt_semantic_seg.squeeze(1).long()
        loss_semantic = self.loss_semantic(semantic_pred, gt_semantic_seg, ignore_index=self.ignore_label)
        return loss_semantic

    def get_semantic_cls(self, semantic_pred, ori_shape, img_shape, scale_factor, rescale):
        """Only be called during val & test
        semantic_pred.size() is like pad_shape  e.g. (800, 1280, 3)
        ori_shape is img's original shape from the png  e.g. (406, 640, 3)
        img_shape is img's shape, after scaled by (1333, 800), i.e. scaled by scale_factor and rounded.  e.g. (800, 1261, 3)
        """
        assert rescale is True

        if isinstance(semantic_pred, torch.Tensor):
            semantic_pred = semantic_pred[..., :img_shape[0], :img_shape[1]].softmax(dim=1).cpu().numpy()  # crop upper-left corner
        semantic_pred_cls = semantic_pred.argmax(axis=1)

        assert isinstance(semantic_pred, np.ndarray)
        assert isinstance(semantic_pred_cls, np.ndarray)

        img_h, img_w = ori_shape[:2]
        seg_image = mmcv.imresize(semantic_pred_cls.transpose(1, 2, 0), (img_w, img_h), interpolation='nearest')

        return seg_image


# if __name__ == "__main__":
#     model = FCNSemanticHead(
#         in_channels=256,
#         upsample_out_channels=128,
#         num_classes=54,
#         featmap_strides=[4, 8, 16, 32],
#         loss_semantic=dict(
#             type='CrossEntropyLoss'))
#     inputs = []
#     for i in [8, 4, 2, 1]:
#         inputs.append(torch.from_numpy(np.random.randn(2, 256, i, i)).float())
#     outputs = model(inputs)
#     assert outputs.size() == (2, 54, 32, 32)
