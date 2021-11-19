_base_ = './tood_r2-101_fpn_1x_minicoco.py'
# multi-scale training
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline))
# learning policy
lr_config = dict(step=[16,22])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=24)  
work_dir = 'work_dirs/minicoco/tood/tood_r2-101_fpn_mstrasin_2x_minicoco'