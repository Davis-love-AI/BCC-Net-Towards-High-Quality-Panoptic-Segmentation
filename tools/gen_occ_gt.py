#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import argparse
import numpy as np
import json
import time
import multiprocessing

import PIL.Image as Image

from panopticapi.utils import get_traceback, IdGenerator, rgb2id, save_json

try:
    from pycocotools import mask as COCOmask
    from pycocotools.coco import COCO as COCO
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")


@get_traceback
def generate_occlusion_ground_truth_single_core(
        proc_id, coco_detection, annotations_set, categories, segmentations_folder, threshold
):
    id_generator = IdGenerator(categories)

    occlusion_gt = []
    for working_idx, annotation in enumerate(annotations_set):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id,
                                                                 working_idx,
                                                                 len(annotations_set)))
        img_id = annotation['image_id']
        img = coco_detection.loadImgs(int(img_id))[0]
        overlaps_map = np.zeros((img['height'], img['width']), dtype=np.uint32)

        anns_ids = coco_detection.getAnnIds(img_id)
        anns = coco_detection.loadAnns(anns_ids)

        occlusion_record = dict()
        occlusion_record['image_id'] = img_id
        file_name = '{}.png'.format(annotation['file_name'].rsplit('.')[0])

        # Read instance segments from panoptic segmentation gt
        try:
            pan_format = np.array(
                Image.open(os.path.join(segmentations_folder, file_name)), dtype=np.uint32
            )
        except IOError:
            raise KeyError('no prediction png file for id: {}'.format(annotation['image_id']))
        pan = rgb2id(pan_format)
        pan_mask = {}
        for segm_info in annotation['segments_info']:
            if categories[segm_info['category_id']]['isthing'] != 1:
                continue
            mask = (pan == segm_info['id']).astype(np.uint8)
            pan_mask[segm_info['id']] = mask

        # Read instance segments from instance segmentation gt
        segments_info = []
        ins_mask = {}
        overlap_pairs = []
        for ann in anns:
            if ann['category_id'] not in categories:
                raise Exception('Panoptic coco categories file does not contain \
                    category with id: {}'.format(ann['category_id'])
                                )
            _, color = id_generator.get_id_and_color(ann['category_id'])
            mask = coco_detection.annToMask(ann)
            overlaps_map += mask
            pan_format[mask == 1] = color
            ann.pop('segmentation')
            ann.pop('image_id')
            # ann['id'] kept as the same
            segments_info.append(ann)

            ins_mask[ann['id']] = mask

        # match segment ID in instance segmentation and panoptic segmentation by IOU
        ins2pan = {}
        pan2ins = {}
        for pan_id in pan_mask:
            iou_max = 0
            match = None
            for ins_id in ins_mask:
                if ins_id in ins2pan:
                    continue
                mask_sum = pan_mask[pan_id] + ins_mask[ins_id]
                iou = np.sum(mask_sum > 1) / np.sum(mask_sum > 0)
                if iou > iou_max:
                    iou_max = iou
                    match = ins_id
            if not match:
                print("Inconsistent panoptic annotation and instance annotation")
            else:
                ins2pan[match] = pan_id
                pan2ins[pan_id] = match

        if np.sum(overlaps_map > 1) != 0:
            for i, _ in enumerate(segments_info):
                for j in range(i + 1, len(segments_info)):
                    id_i = segments_info[i]['id']
                    id_j = segments_info[j]['id']
                    mask_i = ins_mask[id_i]
                    mask_j = ins_mask[id_j]
                    mask_merge = mask_i + mask_j
                    r_i = np.sum(mask_merge > 1) / np.sum(mask_i)
                    r_j = np.sum(mask_merge > 1) / np.sum(mask_j)
                    if r_i >= threshold or r_j >= threshold:
                        if id_i not in ins2pan or id_j not in ins2pan:
                            continue
                        pan_id_i = ins2pan[id_i]
                        pan_id_j = ins2pan[id_j]
                        pan_id_top = None
                        max_cnt = 0
                        pan_intersection = pan[mask_merge > 1]
                        candidate_ids, candidate_cnts = np.unique(pan_intersection, return_counts=True)    # count the number of different segments
                        if candidate_ids.size == 0:
                            print("candidate_ids: ")
                            print(candidate_ids)
                            print("candidate_cnts: ")
                            print(candidate_cnts)
                            print("filename: ")
                            print(file_name)
                            print("imgid={} ".format(img_id))
                            raise Exception("Wrong intersection.")
                        for it in range(candidate_ids.size):
                            if candidate_ids[it] == 0:  # remove background 0
                                continue
                            if candidate_cnts[it] > max_cnt:
                                max_cnt = candidate_cnts[it]
                                pan_id_top = int(candidate_ids[it])
                        if pan_id_top and pan_id_top in [pan_id_i, pan_id_j]:
                            # overlap_pairs.append((pan_id_i, pan_id_j, pan_id_top))
                            overlap_pairs.append((pan2ins[pan_id_i], pan2ins[pan_id_j], pan2ins[pan_id_top]))
        occlusion_record['overlap_pairs'] = overlap_pairs
        occlusion_gt.append(occlusion_record)

    print('Core: {}, all {} images processed'.format(proc_id, len(annotations_set)))
    return occlusion_gt


def generate_occlusion_ground_truth(input_json_file_ins,
                                    input_json_file_pan,
                                    segmentations_folder,
                                    output_json_file,
                                    categories_json_file,
                                    threshold):
    start_time = time.time()

    if not os.path.isdir(segmentations_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(segmentations_folder))
        os.mkdir(segmentations_folder)

    print("CONVERTING...")
    print("COCO detection format:")
    print("\tJSON file: {}".format(input_json_file_ins))
    print("TO")
    print("COCO panoptic format")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(input_json_file_pan))
    print('\n')

    coco_detection = COCO(input_json_file_ins)
    with open(input_json_file_pan, 'r') as f:
        d_coco = json.load(f)
    annotations_panoptic = d_coco['annotations']

    with open(categories_json_file, 'r') as f:
        categories_list = json.load(f)
    categories = {category['id']: category for category in categories_list}

    # cpu_num = multiprocessing.cpu_count()
    cpu_num = 4
    annotations_split = np.array_split(annotations_panoptic, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotations_set in enumerate(annotations_split):
        p = workers.apply_async(generate_occlusion_ground_truth_single_core,
                                (proc_id, coco_detection, annotations_set, categories, segmentations_folder, threshold))
        processes.append(p)
    occlusion_gt = []
    for p in processes:
        occlusion_gt.extend(p.get())

    save_json(occlusion_gt, output_json_file)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts detection COCO format to panoptic \
            COCO format. See this file's head for more information."
    )
    parser.add_argument('--input_json_file_ins', type=str,
                        help="JSON file with detection COCO format")
    parser.add_argument('--input_json_file_pan', type=str,
                        help="JSON file with panoptic COCO format")
    parser.add_argument(
        '--segmentations_folder', type=str, default=None, help="Folder with \
         panoptic COCO format segmentations. Default: X if output_json_file is \
         X.json"
    )
    parser.add_argument('--output_json_file', type=str,
                        help="Json file for occlusion ground truth")
    parser.add_argument('--categories_json_file', type=str,
                        help="JSON file with Panoptic COCO categories information")
    parser.add_argument('--threshold', type=float,
                        help="Json file for occlusion ground truth",
                        default=0.2)
    args = parser.parse_args()
    generate_occlusion_ground_truth(args.input_json_file_ins,
                                    args.input_json_file_pan,
                                    args.segmentations_folder,
                                    args.output_json_file,
                                    args.categories_json_file,
                                    args.threshold)
