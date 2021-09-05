_base_ = './tood_x101_64x4d_fpn_mstrain_2x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    bbox_head=dict(num_dcn_on_head=2)
)
work_dir = 'work_dirs/coco/tood/tood_x101_64x4d_fpn_dconv_c4-c5_mstrain_2x_coco'