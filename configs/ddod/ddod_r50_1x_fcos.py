_base_ = 'ddod_r50_fpn_1x_coco.py'

model = dict(
    bbox_head=dict(
        _delete_=True,
        type='DDODFCOSHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        # bbox_coder=dict(
        #     type='DeltaXYWHBBoxCoder',
        #     target_means=[.0, .0, .0, .0],
        #     target_stds=[0.1, 0.1, 0.2, 0.2]),
        bbox_coder=dict(
            type='TBLRCenterCoder',
            normalizer=1/4.,
            normalize_by_wh=True
        ), 
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_iou=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
	)

work_dir = 'work_dirs/coco/ddod/ddod_fcos_r50_fpn_1x_coco'