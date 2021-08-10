_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        frozen_stages=0,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', checkpoint='/media/amax/Passport_4T/swav_800ep_pretrain.pth.tar')))
work_dir = 'work_dirs/coco/pretrain/swav/faster_rcnn_swav-pretrain_fpn_1x_coco'
