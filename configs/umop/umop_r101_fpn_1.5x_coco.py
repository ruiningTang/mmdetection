_base_ = './umop_r50_fpn_1.5x_coco.py'
model = dict(
    backbone=dict(
        # ResNet-101
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet101')))
work_dir = 'work_dirs/coco/umop/umop_r101_fpn_1x_4x2_coco'