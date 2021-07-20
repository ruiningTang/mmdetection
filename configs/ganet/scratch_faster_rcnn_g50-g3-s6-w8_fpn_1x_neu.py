_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/neu_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='BN',  requires_grad=True)
model = dict(
    backbone=dict(type='GANet_b3',
        depth=50,
        branch=3,
        scales=6,
        base_width=8,
        frozen_stages=-1, zero_init_residual=False, norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(bbox_head=dict(num_classes=6,norm_cfg=norm_cfg)))