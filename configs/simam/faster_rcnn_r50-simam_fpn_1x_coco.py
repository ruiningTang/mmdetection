_base_ = [
    '../_base_/models/faster_rcnn_r50simam_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

find_unused_parameters=True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
work_dir = './work_dirs/coco/simam/faster_rcnn_r50-simam_fpn_1x_coco'
