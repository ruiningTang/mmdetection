_base_ = './tood_r50_fpn_mstrain_2x_coco.py'
model = dict(
    type='TOOD',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))

data = dict(samples_per_gpu=2,
            workers_per_gpu=2)
work_dir = 'work_dirs/coco/tood/tood_x101_64x4d_fpn_mstrain_2x_coco'