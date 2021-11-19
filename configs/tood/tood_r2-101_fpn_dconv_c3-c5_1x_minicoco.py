_base_ = './tood_r2-101_fpn_1x_minicoco.py'
# dcn
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    bbox_head=dict(num_dcn_on_head=2))
#saved dir
work_dir = 'work_dirs/minicoco/tood/tood_r2-101_fpn_dconv_c3-c5_1x_minicoco'