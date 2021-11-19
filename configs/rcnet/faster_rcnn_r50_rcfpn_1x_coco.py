_base_ = [
    '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
]
# optimizer
model = dict(
    neck=dict(type='RCFPN', norm_cfg=dict(type='BN', requires_grad=True)))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

work_dir = 'work_dirs/coco/rcnet/faster_rcnn_r50_rcfpn_1x_coco'