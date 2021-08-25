_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_180k.py', '../_base_/default_runtime.py'
]

evaluation = dict(interval=20000, metric='bbox')
checkpoint_config = dict(by_epoch=False, interval=20000)
work_dir = 'work_dirs/coco/faster_rcnn/faster_rcnn_r50_fpn_4x2_180k_coco'