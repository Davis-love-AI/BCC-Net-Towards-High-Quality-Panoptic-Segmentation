import mmcv
import json
import numpy as np
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .recall import eval_recalls


def coco_eval(result_files, result_types, coco, max_dets=(100, 300, 1000)):
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'
        ]

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    if result_types == ['proposal_fast']:
        ar = fast_eval_recall(result_files, coco, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return

    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        img_ids = coco.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


def fast_eval_recall(results,
                     coco,
                     max_dets,
                     iou_thrs=np.arange(0.5, 0.96, 0.05)):
    if mmcv.is_str(results):
        assert results.endswith('.pkl')
        results = mmcv.load(results)
    elif not isinstance(results, list):
        raise TypeError(
            'results must be a list of numpy arrays or a filename, not {}'.
            format(type(results)))

    gt_bboxes = []
    img_ids = coco.getImgIds()
    for i in range(len(img_ids)):
        ann_ids = coco.getAnnIds(imgIds=img_ids[i])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            gt_bboxes.append(np.zeros((0, 4)))
            continue
        bboxes = []
        for ann in ann_info:
            if ann.get('ignore', False) or ann['iscrowd']:
                continue
            x1, y1, w, h = ann['bbox']
            bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[0] == 0:
            bboxes = np.zeros((0, 4))
        gt_bboxes.append(bboxes)

    recalls = eval_recalls(
        gt_bboxes, results, max_dets, iou_thrs, print_summary=False)
    ar = recalls.mean(axis=1)
    return ar


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def proposal2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                json_results.append(data)
    return json_results


def segm2json(dataset, results):
    bbox_json_results = []
    segm_json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        det, seg = results[idx]
        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                bbox_json_results.append(data)

            # segm results
            # some detectors use different score for det and segm
            if len(seg) == 2:
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score[i])
                data['category_id'] = dataset.cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results


def panoptic2json(dataset, results):
    json_results = []
    seg_fcn_results = []
    inst_idx = -1
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        det, seg, seg_fcn = results[idx][:3]
        for label in range(len(det)):
            bboxes = det[label]
            segms = seg[label]  # TODO: modify for mask_scoring_rcnn
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]

                # CCS: not officially required, for auxiliary purposes
                # unique for every predicted instance from the dataset
                inst_idx += 1
                data['inst_idx'] = inst_idx

                json_results.append(data)

        # Calculate result of semantic segmentation (stuff)
        if dataset.num_semantic == 134:
            cat_map_dict = dataset.cat_134_to_200
        elif dataset.num_semantic == 54:
            cat_map_dict = dataset.cat_54_to_200
        else:
            raise Exception('num_semantic incorrect!')
        unique_category_id = np.unique(seg_fcn)
        for i in unique_category_id:
            if dataset.num_semantic == 134 and i <= 80:
                continue  # ignore void class and thing classes (simply not writing them into the semantic result)
            new_dict = {}
            binary_mask = np.zeros(seg_fcn.shape, dtype=np.uint8)
            position = np.where(seg_fcn == i)
            binary_mask[position[0], position[1]] = 1
            binary_mask = np.asfortranarray(binary_mask)
            pre_binary_mask = mask_util.encode(binary_mask)
            new_dict['image_id'] = img_id
            new_dict['segmentation'] = pre_binary_mask
            new_dict['category_id'] = cat_map_dict[i]
            seg_fcn_results.append(new_dict)

    return json_results, seg_fcn_results


def aux2json(dataset, results):
    aux_results = dict()
    # srm
    if len(results[0]) >= 4 and "srm_results" in results[0][3]:
        inst_idx2_srm_scores = dict()  # maps inst_idx -> srm_avg_score
        inst_idx = -1
        for idx in range(len(dataset)):
            aux_result = results[idx][3]
            srm_result = aux_result.get("srm_results")
            for label in range(len(srm_result)):  # for each of 80 classes
                avg_scores = srm_result[label]
                for i in range(len(avg_scores)):
                    # srm
                    inst_idx += 1
                    inst_idx2_srm_scores[inst_idx] = avg_scores[i]
        aux_results['inst_idx2_srm_scores'] = inst_idx2_srm_scores
    # occ
    if len(results[0]) >= 4 and "occ_results" in results[0][3]:
        imgid2_occ = dict()
        for idx in range(len(dataset)):
            img_id = dataset.img_ids[idx]
            aux_result = results[idx][3]
            occ_result = aux_result.get("occ_results")
            imgid2_occ[img_id] = occ_result
        aux_results['imgid2_occ'] = imgid2_occ
    return aux_results


def results2json(dataset, results, out_file, out_file_semantic=None, out_file_aux=None):
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        mmcv.dump(json_results, result_files['bbox'])
    elif isinstance(results[0], tuple) and len(results[0]) == 2:
        json_results = segm2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        mmcv.dump(json_results, result_files['proposal'])

    elif isinstance(results[0], tuple) and len(results[0]) > 2:
        # panoptic
        json_results, json_results_semantic = panoptic2json(dataset, results)
        aux_results = aux2json(dataset, results)
        # json_results:          results of instance segmentation
        # json_results_semantic: results of semantic segmentation
        # aux_results:           results of srm, occ, etc.

        class MyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, bytes):
                    return str(obj, encoding='utf-8')
                return json.JSONEncoder.default(self, obj)

        mmcv.dump(json_results, out_file)
        json.dump(json_results_semantic, open(out_file_semantic, 'w'), cls=MyEncoder)
        json.dump(aux_results, open(out_file_aux, 'w'), cls=MyEncoder)
        # result_files will still be an empty dict
    else:
        raise TypeError('invalid type of results')
    return result_files
