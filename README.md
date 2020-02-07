# BCC-Net: Towards High Quality Panoptic Segmentation
This is the implementation of *BCC-Net: Towards High Quality Panoptic Segmentation* using the open source object detection toolbox [MMDetection](https://github.com/open-mmlab/mmdetection).
## Proposed Modules

| Proposed Module                   | Relevant Code                                                |
| --------------------------------- | ------------------------------------------------------------ |
| Unknown Erasing                   | [mmdet/models/detectors/test_mixins.py#L258](https://github.com/under-review/BCC-Net-Towards-High-Quality-Panoptic-Segmentation/blob/master/mmdet/models/detectors/test_mixins.py#L258) |
| Occlusion Handling                 | [mmdet/models/occ_heads/occlusion_head.py#L29](https://github.com/under-review/BCC-Net-Towards-High-Quality-Panoptic-Segmentation/blob/master/mmdet/models/occ_heads/occlusion_head.py#L29) |
| Gradient-Balancing FPN            | [mmdet/models/detectors/two_stage.py#L222](https://github.com/under-review/BCC-Net-Towards-High-Quality-Panoptic-Segmentation/blob/master/mmdet/models/detectors/two_stage.py#L222) |
| Cascaded Semantic Mask Projection | [mmdet/models/detectors/panoptic_htc.py#L89](https://github.com/under-review/BCC-Net-Towards-High-Quality-Panoptic-Segmentation/blob/master/mmdet/models/detectors/panoptic_htc.py#L89)<br>[mmdet/models/detectors/panoptic_htc.py#L122](https://github.com/under-review/BCC-Net-Towards-High-Quality-Panoptic-Segmentation/blob/master/mmdet/models/detectors/panoptic_htc.py#L122)<br>[mmdet/models/detectors/htc.py#L115](https://github.com/under-review/BCC-Net-Towards-High-Quality-Panoptic-Segmentation/blob/master/mmdet/models/detectors/htc.py#L115) |