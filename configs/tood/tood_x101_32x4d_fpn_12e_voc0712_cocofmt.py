_base_ = './tood_r50_fpn_12e_voc0712_cocofmt.py'
model = dict(
    type='TOOD',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')))
work_dir = 'work_dirs/voc/tood/tood_x101_32x4d_fpn_12e_voc0712_cocofmt'