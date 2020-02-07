#!/usr/bin/env python
'''
This script converts data in panoptic COCO format to semantic segmentation. All
segments with the same semantic class in one image are combined together.

Additional option:
- using option '--things_others' the script combine all segments of thing
classes into one segment with semantic class 'other'.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import argparse
import numpy as np
import json
import mmcv
import time
import multiprocessing

import PIL.Image as Image

from panopticapi.utils import get_traceback

try:
    # set up path for pycocotools
    # sys.path.append('./cocoapi-master/PythonAPI/')
    from pycocotools import mask as COCOmask
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")


@get_traceback
def extract_semantic_single_core(proc_id,
                                 img_name_set,
                                 folder_200,
                                 folder_134,
                                 cat_200_to_134):
    for working_idx, img_name in enumerate(img_name_set):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, working_idx, len(img_name_set)))
        try:
            semantic_200_img = mmcv.imread(os.path.join(folder_200, img_name), flag="unchanged")
        except IOError:
            raise KeyError('no prediction png file for id: {}'.format(img_name))

        semantic_134_img = np.zeros(semantic_200_img.shape, dtype=np.uint8)

        unique_cat_list = list(np.unique(semantic_200_img))
        for cat in unique_cat_list:
            mask = semantic_200_img == cat
            semantic_134_img[mask] = cat_200_to_134[cat]

        Image.fromarray(semantic_134_img).save(os.path.join(folder_134, img_name))

    print('Core: {}, all {} images processed'.format(proc_id, len(img_name_set)))
    return []


def extract_semantic(folder_200, folder_134):
    start_time = time.time()
    print("Number of images found: ")
    img_names = [name for name in os.listdir(folder_200)]
    print(len(img_names))

    cat_json = json.load(open('./panopticapi/panoptic_coco_categories.json'))  # discrete
    cat_200_to_134 = dict()
    cat_200_to_134[0] = 0
    for idx, data in enumerate(cat_json):
        cat_200_to_134[data['id']] = idx + 1
        # thing class: mapped tp 1 - 80
        # stuff class: mapped to 81 - 133

    print("category mapping from 200 to 134:")
    print(cat_200_to_134)

    if not os.path.exists(folder_134):
        os.mkdir(folder_134)

    cpu_num = multiprocessing.cpu_count()
    img_name_split = np.array_split(img_names, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(img_name_split[0])))

    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, img_name_set in enumerate(img_name_split):
        p = workers.apply_async(extract_semantic_single_core,
                                (proc_id, list(img_name_set), folder_200, folder_134, cat_200_to_134))
        processes.append(p)
    temp = []
    for p in processes:
        temp.extend(p.get())

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts data in panoptic COCO format to \
        semantic segmentation. All segments with the same semantic class in one \
        image are combined together. See this file's head for more information."
    )
    parser.add_argument('--semantic_seg_folder_200', type=str, default=None,
                        help="input folder")
    parser.add_argument('--semantic_seg_folder_134', type=str, default=None,
                        help="output folder")
    args = parser.parse_args()
    extract_semantic(args.semantic_seg_folder_200, args.semantic_seg_folder_134)
