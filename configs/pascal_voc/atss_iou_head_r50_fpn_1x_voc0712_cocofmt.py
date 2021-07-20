_base_ = '../ddod/atss_iou_head_r50_fpn_1x_coco.py'

model = dict(bbox_head = dict(num_classes=20))

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/amax/VOCdevkit/'

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root+'voc0712_trainval.json',
            img_prefix=data_root,
            pipeline=train_pipeline,
            classes=CLASSES)),
    val=dict(
        type=dataset_type,
        ann_file=data_root+'voc07_test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=data_root+'voc07_test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=CLASSES))

evaluation = dict(interval=1, metric='bbox')
# optimizer
optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

work_dir = './work_dirs/voc/atss/atss_iou_head_r50_fpn_1x_voc0712_cocofmt'
