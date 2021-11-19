_base_ = './tood_r101_fpn_dconv_c3-c5_mstrain_1.5x_coco.py'
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
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')),
    bbox_head=dict(num_dcn_on_head=2))
work_dir = 'work_dirs/coco/tood/tood_x101_32x4d_fpn_dconv_c4-c5_mstrain_1.5x_coco'