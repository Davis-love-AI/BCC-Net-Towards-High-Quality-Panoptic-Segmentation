import mmcv
import numpy as np
from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class CocoPanopticDataset(CocoDataset):

    CLASSES = CocoDataset.CLASSES + (
        'banner', 'blanket', 'bridge', 'cardboard',
        'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
        'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
        'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf',
        'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
        'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged',
        'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged',
        'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged',
        'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged',
        'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged')
    # the last 53 classes are for stuff (background)

    def __init__(self, ann_file, cat_pan_file_discrete, num_semantic=134, **kwargs):
        super(CocoPanopticDataset, self).__init__(ann_file, **kwargs)

        self.ann_file = ann_file
        self.cat_pan_file_discrete = cat_pan_file_discrete

        cat_json = mmcv.load(self.cat_pan_file_discrete)  # discrete

        self.num_semantic = num_semantic
        if self.num_semantic == 134:
            self.cat_134_to_200 = dict()
            self.cat_134_to_200[0] = 0
            for idx, data in enumerate(cat_json):
                self.cat_134_to_200[idx + 1] = data['id']  # all class map back to original discrete categories
        elif self.num_semantic == 54:
            self.cat_54_to_200 = dict()
            self.cat_54_to_200[0] = 0
            for idx, data in enumerate(cat_json[80:]):
                self.cat_54_to_200[idx + 1] = data['id']
        else:
            raise Exception('num_semantic incorrect!')

    def _parse_ann_info(self, ann_info, with_mask=True):
        if hasattr(self, "with_occ") and self.with_occ:
            return self._parse_ann_info_occ(ann_info, with_mask=with_mask)
        elif hasattr(self, "with_srm") and self.with_srm:
            return self._parse_ann_info_srm(ann_info, with_mask=with_mask)
        else:
            return super(CocoPanopticDataset, self)._parse_ann_info(ann_info, with_mask=with_mask)

    def _parse_ann_info_srm(self, ann_info, with_mask=True):
        # for srm
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []

            gt_masks_iscrowd = []
            gt_mask_polys_iscrowd = []
            gt_poly_lens_iscrowd = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
                if with_mask:
                    gt_masks_iscrowd.append(self.coco.annToMask(ann))
                    mask_polys = [
                        p for p in ann['segmentation'] if len(p) >= 6
                    ]  # valid polygons have >= 3 points (6 coordinates)
                    poly_lens = [len(p) for p in mask_polys]
                    gt_mask_polys_iscrowd.append(mask_polys)
                    gt_poly_lens_iscrowd.extend(poly_lens)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if with_mask:
                    gt_masks.append(self.coco.annToMask(ann))
                    mask_polys = [
                        p for p in ann['segmentation'] if len(p) >= 6
                    ]  # valid polygons have >= 3 points (6 coordinates)
                    poly_lens = [len(p) for p in mask_polys]
                    gt_mask_polys.append(mask_polys)
                    gt_poly_lens.extend(poly_lens)

        # concat gt_x_iscrowd array to the end of gt_x
        gt_masks += gt_masks_iscrowd
        gt_mask_polys += gt_mask_polys_iscrowd
        gt_poly_lens += gt_poly_lens_iscrowd

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def _parse_ann_info_occ(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        assert with_mask
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []

        # occ
        ann_ids = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                # occ
                ann_ids.append(ann['id'])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        # occ
        inst_id2_idx = dict()
        for idx, ann_id in enumerate(ann_ids):
            inst_id2_idx[ann_id] = idx

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens

        # occ
        aux_gt = dict(inst_id2_idx=inst_id2_idx)
        return ann, aux_gt