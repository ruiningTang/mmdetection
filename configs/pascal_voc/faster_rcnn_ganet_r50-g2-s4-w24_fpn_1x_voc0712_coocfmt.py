_base_ = 'faster_rcnn_r50_fpn_1x_voc0712_cocofmt.py'
model = dict(
    pretrained='/media/amax/Passport_4T/ganet50-g2-s4-w24.pth',
    backbone=dict(
        type='GANet',
        depth=50,
        branch=2,
        scales=4,
        base_width=24))

work_dir = 'work_dirs/voc/backbone/ganet/faster_rcnn_g50-g2-s4-w24_fpn_1x_voc0712_cocofmt'
