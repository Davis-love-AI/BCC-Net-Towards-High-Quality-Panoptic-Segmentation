import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS


class FasterNASFPN(nn.Module):
    """
    NAS Feature Pyramid Network for Faster-RCNN

    """

    def __init__(self, inplanes, outplanes, bn=None, normalize=None, initializer=None):
        """
        Arguments:
            - inplanes (:obj:`list` of :obj:`int`): input channel
            - outplanes (:obj:`list` of :obj:`int`): output channel, all layers are the same
            - bn (:obj:`dict`): config of BatchNorm (deprecated, use normalize instead)
            - normalize (:obj:`dict`): config of Normalization Layer
            - initializer (:obj:`dict`): config for model parameter initialization

        """
        super(FasterNASFPN, self).__init__()

        self.inplanes = inplanes
        self.outplanes = outplanes

        if isinstance(inplanes, list):
            for inp in inplanes:
                assert inp == outplanes
        planes = outplanes

        # if bn is not None:
        #     normalize = parse_deprecated_bn_style(bn)

        self.p2_1_conv = build_conv_norm(planes,
                                         planes,
                                         kernel_size=3,
                                         padding=1,
                                         normalize=normalize,
                                         relu_first=True,
                                         activation=True)
        self.p4_1_conv = build_conv_norm(planes,
                                         planes,
                                         kernel_size=3,
                                         padding=1,
                                         normalize=normalize,
                                         relu_first=True,
                                         activation=True)
        self.p4_2_conv = build_conv_norm(planes,
                                         planes,
                                         kernel_size=3,
                                         padding=1,
                                         normalize=normalize,
                                         relu_first=True,
                                         activation=True)
        self.p4_3_conv = build_conv_norm(planes,
                                         planes,
                                         kernel_size=3,
                                         padding=1,
                                         normalize=normalize,
                                         relu_first=True,
                                         activation=True)
        self.p3_1_conv = build_conv_norm(planes,
                                         planes,
                                         kernel_size=3,
                                         padding=1,
                                         normalize=normalize,
                                         relu_first=True,
                                         activation=True)
        self.p5_1_conv = build_conv_norm(planes,
                                         planes,
                                         kernel_size=3,
                                         padding=1,
                                         normalize=normalize,
                                         relu_first=True,
                                         activation=True)
        self.p6_1_conv = build_conv_norm(planes,
                                         planes,
                                         kernel_size=3,
                                         padding=1,
                                         normalize=normalize,
                                         relu_first=True,
                                         activation=True)

        initialize_from_cfg(self, initializer)

    def forward(self, input):
        """
        .. note::

        Arguments:
            - input (:obj:`dict`): output of ``Backbone``

        Returns:
            - out (:obj:`dict`):

        """
        p2_0, p3_0, p4_0, p5_0, p6_0 = input
        # for p in input['features']:
        #     logger.info(f'p.shape:{p.shape}')

        p4_1 = self.merge_gp(p6_0, p4_0)
        p4_1 = self.p4_1_conv(p4_1)

        p4_2 = self.merge_sum(p4_0, p4_1)
        p4_2 = self.p4_2_conv(p4_2)

        p3_1 = self.merge_sum(p4_2, p3_0)
        p3_1 = self.p3_1_conv(p3_1)

        p2_1 = self.merge_sum(p3_1, p2_0)
        p2_1 = self.p2_1_conv(p2_1)

        p4_3 = self.merge_sum(p3_1, p4_2)
        p4_3 = self.p4_3_conv(p4_3)

        p5_1 = self.merge_sum(self.merge_gp(p3_1, p4_3), p5_0)
        p5_1 = self.p5_1_conv(p5_1)

        p6_1 = self.merge_gp(p5_1, p6_0)
        p6_1 = self.p6_1_conv(p6_1)

        features = [p2_1, p3_1, p4_3, p5_1, p6_1]

        return features

    def get_outplanes(self):
        return self.outplanes

    def align_to(self, x, out):
        if x.shape[-2:] == out.shape[-2:]:
            return x
        x_h, x_w = x.shape[-2:]
        out_h, out_w = out.shape[-2:]
        if x_h < out_h:
            return F.interpolate(x, size=(out_h, out_w), mode='nearest')
        else:
            assert x_h % out_h == 0 and x_w % out_w == 0, f'{x.shape} vs {out.shape}'
            scale = (x_h // out_h, x_w // out_w)
            return F.max_pool2d(x, kernel_size=scale, stride=scale)

    def merge_gp(self, x1, x2, out=None):
        if out is None:
            out = x2
        x1 = self.align_to(x1, out)
        x2 = self.align_to(x2, out)
        gp = F.adaptive_avg_pool2d(x1, 1)
        att = torch.sigmoid(gp)
        return att * x2 + x1

    def merge_sum(self, x1, x2, out=None):
        if out is None:
            out = x2
        x1 = self.align_to(x1, out)
        x2 = self.align_to(x2, out)
        return x1 + x2


@NECKS.register_module
class StackedNASFPN(nn.Module):
    """
    Stacked NAS Feature Pyramid Network

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 fpn_type='Faster',
                 fpn_num=7,
                 checkpoint=False,
                 bn=None,
                 normalize=None,
                 initializer=None):
        super(StackedNASFPN, self).__init__()
        assert fpn_type == 'Faster'
        assert bn is None
        assert normalize is None

        self.fpn_type = fpn_type
        self.checkpoint = checkpoint
        self.initializer = initializer
        fpn_module = {'Faster': FasterNASFPN}[fpn_type]

        in_channels += [in_channels[-1]]

        self.reduction_neck = nn.ModuleList()
        for inp in in_channels:
            self.reduction_neck.append(nn.Conv2d(inp, out_channels, kernel_size=1, padding=0))
        # if self.fpn_type == 'Faster':
        #     self.reduction_neck.append(nn.Conv2d(inplanes[-1], outplanes, kernel_size=1, padding=0))
        # else:
        #     self.reduction_neck.append(nn.Conv2d(inplanes[-1], outplanes, kernel_size=1, padding=0))
        #     self.reduction_neck.append(nn.Conv2d(inplanes[-1], outplanes, kernel_size=1, padding=0))

        fpn_layers = []
        for i in range(fpn_num):
            fpn = fpn_module(out_channels, out_channels, bn, normalize, initializer=self.initializer)
            fpn_layers.append(fpn)
        self.stacked_fpn = nn.Sequential(*fpn_layers)

    def init_weights(self):
        initialize_from_cfg(self, self.initializer)

    def get_outplanes(self):
        return self.stacked_fpn[-1].get_outplanes()

    def get_outstrides(self):
        return self.stacked_fpn[-1].get_outstrides()

    @auto_fp16()
    def forward(self, x):
        x = list(x)
        c5 = x[-1]
        c6 = F.max_pool2d(c5, kernel_size=2, stride=2)
        x.append(c6)

        x = [reduction(f) for reduction, f in zip(self.reduction_neck, x)]

        out = self.stacked_fpn(x)
        # for f in out['features']:
        #     logger.info(f'stacked fpn out:{f.shape}')
        return out  # stride [4, 8, 16, 32, 64]


# ===========================================================================

def init_weights_normal(module, std=0.01):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(
                m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, std=std)
            if m.bias is not None:
                m.bias.data.zero_()


def init_weights_xavier(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(
                m, nn.ConvTranspose2d):
            xavier_init(m, distribution='uniform')


def init_weights_msra(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(
                m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data, a=1)
            if m.bias is not None:
                m.bias.data.zero_()


def init_bias_focal(module, cls_loss_type, init_prior, num_classes):
    if cls_loss_type == 'sigmoid':
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                # to keep the torch random state
                m.bias.data.normal_(-math.log(1.0 / init_prior - 1.0), init_prior)
                torch.nn.init.constant_(m.bias, -math.log(1.0 / init_prior - 1.0))
    elif cls_loss_type == 'softmax':
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                m.bias.data.normal_(0, 0.01)
                for i in range(0, m.bias.data.shape[0], num_classes):
                    fg = m.bias.data[i + 1:i + 1 + num_classes - 1]
                    mu = torch.exp(fg).sum()
                    m.bias.data[i] = math.log(mu * (1.0 - init_prior) / init_prior)
    else:
        raise NotImplementedError(f'{cls_loss_type} is not supported')


def initialize(model, method, **kwargs):
    # initialize BN
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # initialize Conv & FC
    if method == 'normal':
        init_weights_normal(model, **kwargs)
    elif method == 'msra':
        init_weights_msra(model)
    elif method == 'xavier':
        init_weights_xavier(model)
    else:
        raise NotImplementedError(f'{method} not supported')


def initialize_from_cfg(model, cfg):
    if cfg is None:
        initialize(model, 'xavier')
        return

    cfg = copy.deepcopy(cfg)
    method = cfg.pop('method')
    initialize(model, method, **cfg)


def build_conv_norm(in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    normalize=None,
                    activation=False,
                    relu_first=False):

    conv = nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        groups=groups, bias=(normalize is None))

    # for compability
    if (normalize is None) and (not activation):
        return conv

    seq = nn.Sequential()
    if relu_first and activation:
        seq.add_module('relu', nn.ReLU(inplace=True))
    seq.add_module('conv', conv)
    if normalize is not None:
        norm_name, norm = build_norm_layer(out_channels, normalize)
        seq.add_module(norm_name, norm)
    if activation:
        if not relu_first:
            seq.add_module('relu', nn.ReLU(inplace=True))
    return seq


_norm_cfg = {
    'solo_bn': ('bn', torch.nn.BatchNorm2d),
}


def build_norm_layer(num_features, cfg, postfix=''):
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg = cfg.copy()
    layer_type = cfg.pop('type')
    kwargs = cfg.get('kwargs', {})

    if layer_type not in _norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = _norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    layer = norm_layer(num_features, **kwargs)
    return name, layer


if __name__ == "__main__":
    fpn = StackedNASFPN([256, 512, 1024, 2048], 256)
    inputs = [
        torch.rand((2, 256, 200, 304)).float(),
        torch.rand((2, 512, 100, 152)).float(),
        torch.rand((2, 1024, 50, 76)).float(),
        torch.rand((2, 2048, 25, 38)).float()]
    outputs = fpn(inputs)
    for output in outputs:
        print(output.size())
