import torch
import torch.nn as nn
import numpy as np

from torch.nn.functional import softmax
from mmcv.cnn import kaiming_init
from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module
class SRMHead(nn.Module):

    def __init__(self,
                 num_classes=80,
                 ignore_label=200,
                 loss_srm=dict(type='CrossEntropyLoss')):
        super(SRMHead, self).__init__()

        self.num_classes = num_classes

        self.loss_srm = build_loss(loss_srm)
        self.ignore_label = ignore_label

        self.conv_large_kernel_1 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=(1, 7), padding=(0, 3))
        self.conv_large_kernel_2 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=(7, 1), padding=(3, 0))

    def init_weights(self):
        kaiming_init(self.conv_large_kernel_1)
        kaiming_init(self.conv_large_kernel_2)

    def forward(self, processed_masks, test_mode=False):
        assert processed_masks.size()[1] == self.num_classes  # (batch, 80, H, W)

        # processed_masks contain 0 or 1
        srm_score_pred = self.conv_large_kernel_2(self.conv_large_kernel_1(processed_masks))

        return srm_score_pred

    def loss(self, srm_score_pred, gt_semantic_80):
        """
        :param srm_score_pred:
        :param gt_semantic_80: (img_per_gpu, 1?, H, W) (values in 0~79, or 200 (200 will be ignored))
        :return:
        """
        gt_semantic_80 = gt_semantic_80.squeeze(1).long()  # become 3D
        loss_srm = self.loss_srm(srm_score_pred, gt_semantic_80, ignore_index=self.ignore_label)
        return dict(loss_srm=loss_srm)

    def get_srm_avg_scores(self, srm_score_pred, aligned_masks, det_labels, srm_inputs):
        """Only be called during val & test (imgs_per_gpu is 1)
        :param srm_score_pred: Tensor(1, 80, H, W)
        :param aligned_masks: Tensor(#instance, H, W)
        :param det_labels: Tensor(#instance,) (each number in this 1D array is within 0~79)
        :param srm_inputs: Tensor(80, H, W)
        :return List[int]: a list of averaged srm scores for each instance
        """
        srm_score_pred = softmax(srm_score_pred, dim=1)  # apply softmax to conv results
        srm_score_pred = srm_score_pred[0]  # 4D tensor to 3D
        assert srm_score_pred.size()[0] == self.num_classes  # 80
        assert aligned_masks.size()[0] == det_labels.size()[0]

        inter_class_occlusion_mask = torch.sum(srm_inputs, dim=0) > 1
        assert len(inter_class_occlusion_mask.size()) == 2
        assert len(aligned_masks.size()) == 3
        assert inter_class_occlusion_mask.size() == aligned_masks.size()[1:]

        aligned_masks = aligned_masks * inter_class_occlusion_mask.float()  # take intersection!!
        avg_scores = []
        for i in range(det_labels.size()[0]):
            cls = det_labels[i]
            sum = torch.sum(srm_score_pred[cls][aligned_masks[i] == 1])
            count = torch.sum(aligned_masks[i])
            if int(count) == 0:
                avg_scores.append(0.0)
            else:
                avg_scores.append(float(sum / count))
        return avg_scores


if __name__ == "__main__":
    model = SRMHead(
        num_classes=80,
        loss_srm=dict(
            type='CrossEntropyLoss'))
    inputs = torch.LongTensor(1, 80, 880, 880).random_(0, 2).float()  # random binary numbers
    outputs = model(inputs)
    assert outputs.size() == (1, 80, 880, 880)
