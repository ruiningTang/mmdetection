_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'

model = dict(
	neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_input',
            num_outs=5),
        dict(
            type='SEPC',
            out_channels=256,
            Pconv_num=4,
            pconv_deform=False,
            lcconv_deform=False,
            iBN=False,  # when open, please set imgs/gpu >= 4
        )
    ],
    bbox_head=dict(type='SepcRetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)))

work_dir = 'work_dirs/coco/spec/pconv_retinanet_r50_fpn_1x_coco'