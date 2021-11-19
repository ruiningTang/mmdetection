_base_ = [
    '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
]
# optimizer
model = dict(
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=dict(type='BN', requires_grad=True)))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

work_dir = 'work_dirs/coco/faster_rcnn/faster_rcnn_r50_fpn-os_1x_coco'