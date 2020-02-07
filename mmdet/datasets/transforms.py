import mmcv
import numpy as np
import torch

__all__ = [
    'ImageTransform', 'BboxTransform', 'MaskTransform', 'SegMapTransform',
    'SRMInputTransform', 'SegMapTransform134To80', 'Numpy2Tensor',
]


class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        return img, img_shape, pad_shape, scale_factor


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


class BboxTransform(object):
    """Preprocess gt bboxes.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
        gt_bboxes = bboxes * scale_factor
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)
        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes


class MaskTransform(object):
    """Preprocess masks.

    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __call__(self, masks, pad_shape, scale_factor, flip=False):
        masks = [
            mmcv.imrescale(mask, scale_factor, interpolation='nearest')
            for mask in masks
        ]
        if flip:
            masks = [mask[:, ::-1] for mask in masks]
        padded_masks = [
            mmcv.impad(mask, pad_shape[:2], pad_val=0) for mask in masks
        ]
        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks


class SegMapTransform(object):
    """Preprocess semantic segmentation maps.

    1. rescale the segmentation map to expected size
    3. flip the image (if needed)
    4. pad the image (if needed)
    """

    def __init__(self, size_divisor=None):
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            img = mmcv.imrescale(img, scale, interpolation='nearest')
        else:
            img = mmcv.imresize(img, scale, interpolation='nearest')
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
        return img


class SRMInputTransform(object):
    """Merge masks into gt_srm_inputs
    """

    def __init__(self):
        pass

    def __call__(self, masks, labels):
        """
        :param masks: a binary (np_array / tensor) of size (#instance, H, W)
        :param labels: a list of classes (for each instance)
        :return: gt_srm_inputs with size (80, H, W) (binary as well)
        """
        assert len(masks.shape) == 3 and len(labels.shape) == 1
        assert masks.shape[0] >= labels.shape[0]  # see CocoDataset._parse_ann_info
        for label in labels:
            assert 0 <= label
            assert label <= 79

        _, H, W = masks.shape
        instance_count = labels.shape[0]

        if isinstance(masks, np.ndarray):
            gt_srm_inputs = np.zeros((80, H, W))
            for i in range(instance_count):
                cls = labels[i]
                gt_srm_inputs[cls, :, :] = np.maximum(gt_srm_inputs[cls, :, :], masks[i, :, :])  # TODO change to logical_or
        elif isinstance(masks, torch.Tensor):
            gt_srm_inputs = torch.zeros((80, H, W)).cuda()
            for i in range(instance_count):
                cls = labels[i]
                gt_srm_inputs[cls, :, :] = torch.max(gt_srm_inputs[cls, :, :], masks[i, :, :])  # TODO change to logical_or
        else:
            raise Exception
        return gt_srm_inputs


class SegMapTransform134To80(object):
    """Transform semantic gt from 134 classes to 81 classes
    """

    def __init__(self):
        pass

    def __call__(self, gt_seg, gt_srm_inputs):
        """
        :param gt_seg: (1, H, W) for pixel-wise class label
        :param gt_srm_inputs (80, H, W)
        :return:
        """
        assert len(gt_seg.shape) == 3 and gt_seg.shape[0] == 1
        assert len(gt_srm_inputs.shape) == 3 and gt_srm_inputs.shape[0] == 80

        # focus on occlusion region
        inter_class_occlusion_mask = np.sum(gt_srm_inputs, axis=0) > 1

        gt_semantic_80 = np.copy(gt_seg)
        gt_semantic_80[gt_semantic_80 == 0] = 201
        gt_semantic_80[gt_semantic_80 > 80] = 201
        gt_semantic_80[0][np.logical_not(inter_class_occlusion_mask)] = 201
        gt_semantic_80 = gt_semantic_80 - 1
        # 200 : ignored
        # 0~79: thing classes

        return gt_semantic_80


class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])
