_base_ = 'faster_rcnn_r50_fpn_1x_voc0712_cocofmt.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

work_dir = './work_dirs/voc/faster_rcnn/faster_rcnn_r50_dconv_c3-c5_fpn_1x_voc0712_cocofmt'
