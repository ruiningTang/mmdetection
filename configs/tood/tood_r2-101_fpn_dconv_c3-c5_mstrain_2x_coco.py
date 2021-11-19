_base_ = './tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py'
# res2net
model = dict(
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://res2net101_v1d_26w_4s')))
work_dir = 'work_dirs/coco/tood/tood_r2-101_fpn_dconv_c3-c5_mstrain_2x_coco'