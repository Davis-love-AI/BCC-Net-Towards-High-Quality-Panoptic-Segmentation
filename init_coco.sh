mkdir data/coco/annotations/semantic134/

# train dataset
PYTHONPATH=$(pwd):$PYTHONPATH:$(pwd)/panopticapi \
    python panopticapi/converters/panoptic2semantic_segmentation.py \
    --input_json_file data/coco/annotations/panoptic_train2017.json \
    --categories_json_file panopticapi/panoptic_coco_categories.json \
    --semantic_seg_folder data/coco/annotations/semantic134/semantic200_train2017

PYTHONPATH=$(pwd):$PYTHONPATH:$(pwd)/panopticapi \
    python ./tools/semantic_200_to_134.py \
    --semantic_seg_folder_200 data/coco/annotations/semantic134/semantic200_train2017 \
    --semantic_seg_folder_134 data/coco/annotations/semantic134/semantic134_train2017

# val dataset
PYTHONPATH=$(pwd):$PYTHONPATH:$(pwd)/panopticapi \
    python panopticapi/converters/panoptic2semantic_segmentation.py \
    --input_json_file data/coco/annotations/panoptic_val2017.json \
    --categories_json_file panopticapi/panoptic_coco_categories.json \
    --semantic_seg_folder data/coco/annotations/semantic134/semantic200_val2017

PYTHONPATH=$(pwd):$PYTHONPATH:$(pwd)/panopticapi \
    python ./tools/semantic_200_to_134.py \
    --semantic_seg_folder_200 data/coco/annotations/semantic134/semantic200_val2017 \
    --semantic_seg_folder_134 data/coco/annotations/semantic134/semantic134_val2017

# remove intermediate png files
rm -rf data/coco/annotations/semantic134/semantic200_train2017
rm -rf data/coco/annotations/semantic134/semantic200_val2017
