_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/minicoco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model
model = dict(bbox_head=dict(reg_decoded_bbox=True,loss_bbox=dict(type='EIoULoss', loss_weight=1.0, linear=False, using_grad_factor=False, smooth_point=0.2)))
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
work_dir = './work_dirs/minicoco/eious/retinanet/retinanet_r50_fpn_eiou-smooth_1x_minicoco'

