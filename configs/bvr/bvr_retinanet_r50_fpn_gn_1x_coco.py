_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='BVR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    bbox_head=dict(
        type='BVRHead',
        bbox_head_cfg=dict(
            type='RetinaHead',
            num_classes=80,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        keypoint_pos='input',
        keypoint_head_cfg=dict(
            type='KeypointHead',
            num_classes=80,
            in_channels=256,
            stacked_convs=2,
            strides=[8, 16, 32, 64, 128],
            shared_stacked_convs=0,
            logits_convs=1,
            head_types=['top_left_corner', 'bottom_right_corner', 'center'],
            corner_pooling=False,
            loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            loss_cls=dict(type='GaussianFocalLoss', loss_weight=0.25),
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        cls_keypoint_cfg=dict(
            keypoint_types=['center'],
            with_key_score=False,
            with_relation=True),
        reg_keypoint_cfg=dict(
            keypoint_types=['top_left_corner', 'bottom_right_corner'],
            with_key_score=False,
            with_relation=True),
        keypoint_cfg=dict(max_keypoint_num=20, keypoint_score_thr=0.0),
        feature_selection_cfg=dict(
            selection_method='index',
            cross_level_topk=50,
            cross_level_selection=True),
        num_attn_heads=8,
        scale_position=False,
        pos_cfg=dict(base_size=[400, 400], log_scale=True, num_layers=2)),
    train_cfg=dict(
        bbox=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        keypoint=dict(
            assigner=dict(type='PointKptAssigner'),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
work_dir = 'work_dirs/coco/bvr/bvr_retinanet_r50_fpn_1x_coco'
