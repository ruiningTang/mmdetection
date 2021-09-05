_base_ = './tood_r101_fpn_mstrain_2x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    bbox_head=dict(num_dcn_on_head=2))
work_dir = 'work_dirs/coco/tood/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco'