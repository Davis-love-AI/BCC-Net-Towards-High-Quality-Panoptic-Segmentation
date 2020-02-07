#!/bin/bash
job_name=${PWD##*/}
partition=
gpus=8
g=$((${gpus}<8?${gpus}:8))
log_dir=${root_path}/experiments/${job_name}
cfg=${log_dir}/${job_name}.yaml

srun -u --mpi=pmi2 --partition=${partition} --job-name=${job_name} \
-n${gpus} --gres=gpu:${g} --ntasks-per-node=${g} \
python gen_occ_gt.py --input_json_file_ins ../data/coco/annotations/instances_train2017.json \
--input_json_file_pan ../data/coco/annotations/panoptic_train2017.json \
--segmentations_folder ../data/coco/annotations/panoptic_train2017 \
--output_json_file ./occ_gt_train.json --categories_json_file ../panopticapi/panoptic_coco_categories.json
