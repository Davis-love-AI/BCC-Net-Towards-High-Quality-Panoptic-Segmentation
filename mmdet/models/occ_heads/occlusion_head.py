import torch
import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init
from ..builder import build_loss
from ..registry import HEADS
import numpy as np
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
@HEADS.register_module
class OcclusionHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 num_fcs=2,
                 roi_feat_size=14,
                 in_channels=514,
                 conv_out_channels=512,
                 fc_out_channels=1024,
                 loss_occ=dict(type='CrossEntropyLoss', use_sigmoid=True),
                 num_class=81,
                 mode=0,
                 cat_stats='',
                 cat_threshold=0.75,
                 min_thing_area = 0.5):
        super(OcclusionHead, self).__init__()

        modes = {
            0: 'feature+mask',      # OCFusion
            1: 'feature+mask+label',    # OCFusion + label
            2: 'label',     # Fitted Matrix
            3: 'stats',     # Statistical matrix
            4: 'SHR'    # Spatial Hierarchical Relation
        }

        self.mode = modes[mode]
        self.num_class = num_class
        self.only_label = True if self.mode =='label' else False
        if self.mode == 'stats':
            cat_stats = np.load(cat_stats, allow_pickle=True).astype(np.float)
            mat_size = cat_stats.shape[0]
            cat_occ = np.full([mat_size, mat_size], -1)
            for i in range(mat_size):
                for j in range(mat_size):
                    if cat_stats[i, j] + cat_stats[j, i] == 0:
                        continue
                    score_cat = cat_stats[i, j] / (cat_stats[i, j] + cat_stats[j, i])
                    if score_cat >= cat_threshold:
                        cat_occ[i, j] = 1
                        cat_occ[j, i] = 0
            self.cat_occ = torch.from_numpy(cat_occ)
        if self.mode == 'SHR':
            self.min_thing_area = min_thing_area
        # structure following mask_rcnn_head
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.loss_occ = build_loss(loss_occ)
        # self.fp16_enabled = False

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            if i == 0:
                # concatenation of mask feature (256 * 2) and mask prediction (1 * 2)
                in_channels = self.in_channels
            else:
                in_channels = self.conv_out_channels
            stride = 2 if i == num_convs - 1 else 1
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    stride=stride,
                    padding=1))

        self.fcs = nn.ModuleList()
        # expand first fc 2 dims to incorporate classes
        for i in range(num_fcs):
            if self.mode == 'feature+mask':
                in_channels = self.conv_out_channels * (
                        roi_feat_size // 2) ** 2 if i == 0 else self.fc_out_channels
            if self.mode == 'feature+mask+label':
                in_channels = self.conv_out_channels * (
                        roi_feat_size // 2) ** 2 + 2*self.num_class if i == 0 else self.fc_out_channels
            if self.mode == 'label':
                in_channels = 2 * self.num_class if i == 0 else self.fc_out_channels
            if self.mode == 'stats':
                in_channels = 2 * self.num_class if i == 0 else self.fc_out_channels
            self.fcs.append(nn.Linear(in_channels, self.fc_out_channels))

        self.fc_final = nn.Linear(self.fc_out_channels, 1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)

    def init_weights(self):
        for conv in self.convs:
            kaiming_init(conv)
        for fc in self.fcs:
            kaiming_init(
                fc,
                a=1,
                mode='fan_in',
                nonlinearity='leaky_relu',
                distribution='uniform')
        normal_init(self.fc_final, std=0.01)

    def forward(self, mask_feats, mask_preds, classes, bbox_pair=None):  # modified to take in classes as input
        if not self.only_label:
            assert len(mask_feats.size()) == 4 and mask_feats.size()[1:] == (512, 14, 14)
            assert len(mask_preds.size()) == 4 and mask_preds.size()[1:] == (2, 28, 28)
        assert len(classes.size()) == 2 and classes.size()[1:] == (2,)

        if self.mode == 'feature+mask':
            mask_preds = mask_preds.sigmoid()
            mask_pred_pooled = self.max_pool(mask_preds)

            x = torch.cat([mask_feats, mask_pred_pooled], dim=1)
            for conv in self.convs:
                x = self.relu(conv(x))
            x = x.view(x.size(0), -1)
            for fc in self.fcs:
                x = self.relu(fc(x))
            logits = self.fc_final(x)
            return logits

        if self.mode == 'feature+mask+label':
            # one hot coding
            classes = classes.long()
            cls_pred = torch.zeros(classes.shape[0], 2 * self.num_class).cuda()

            for i in range(classes.shape[0]):
                cls_pred[i, [classes[i, 0], self.num_class + classes[i, 1]]] = 1

            mask_preds = mask_preds.sigmoid()
            mask_pred_pooled = self.max_pool(mask_preds)

            x = torch.cat([mask_feats, mask_pred_pooled], dim=1)
            for conv in self.convs:
                x = self.relu(conv(x))
            x = x.view(x.size(0), -1)
            x = torch.cat([x, cls_pred], dim=-1)
            for fc in self.fcs:
                x = self.relu(fc(x))
            logits = self.fc_final(x)
            return logits

        if self.mode == 'label':
            # one hot coding
            classes = classes.long()
            cls_pred = torch.zeros(classes.shape[0], 2*self.num_class).cuda()

            for i in range(classes.shape[0]):
                cls_pred[i, [classes[i, 0], self.num_class+classes[i, 1]]] = 1
            x = cls_pred
            for fc in self.fcs:
                x = self.relu(fc(x))
            logits = self.fc_final(x)
            return logits

        if self.mode == 'stats':
            classes = classes.int()
            logits = []
            for i in range(classes.shape[0]):
                logits.append(self.cat_occ[classes[i,0]][classes[i,1]])
            logits = torch.tensor(logits).float().cuda()
            return logits

        if self.mode == 'SHR':
            logit_one = torch.ones(1).float().cuda()
            if bbox_pair==None:
                raise Exception("SHR is only available in testing")
            # bbox_overlaps take bbox1 as foreground obj by default!
            bbox1, bbox2 = [x.unsqueeze(dim=0).cpu().numpy() for x in bbox_pair]
            if bbox_overlaps(bbox1, bbox2, mode='iof')[0] > self.min_thing_area:
                return torch.stack([logit_one * 1, logit_one * 0])
            if bbox_overlaps(bbox2, bbox1, mode='iof')[0] > self.min_thing_area:
                return torch.stack([logit_one * 0, logit_one * 1])
            return torch.stack([logit_one * -1, logit_one * -1])

    def loss(self, occ_pred, target_binaries):
        loss_occ = self.loss_occ(occ_pred, target_binaries)
        return dict(loss_occ=loss_occ)

# if __name__ == "__main__":
#     model = OcclusionHead(
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
