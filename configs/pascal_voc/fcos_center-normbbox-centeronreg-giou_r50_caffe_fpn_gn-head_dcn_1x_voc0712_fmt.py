_base_ = 'fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_voc0712_cocofmt.py'

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    bbox_head=dict(
        dcn_on_last_conv=True))
    # training and testing settings
work_dir = './work_dirs/voc/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_voc0712_cocofmt'
