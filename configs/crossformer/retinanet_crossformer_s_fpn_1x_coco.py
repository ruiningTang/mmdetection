_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py'
]
model = dict(
    pretrained='/media/amax/Passport_4T/backbone-crossformer-s.pth',
    backbone=dict(
        type='CrossFormer_S',
        group_size=[7, 7, 7, 7],
        crs_interval=[8, 4, 2, 1]),
    neck=dict(
        type='FPN',
        in_channels=[96,192,384,768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))
# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = 'work_dirs/coco/crossformer/retinanet_crossformer_s_fpn_1x_coco'