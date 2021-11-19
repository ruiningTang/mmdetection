_base_ = './umop_r50_fpn_1.5x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet101')))
work_dir = 'work_dirs/coco/umop/umop_r101_fpn_mdconv_c3-c5_1.5x_4x2_coco'