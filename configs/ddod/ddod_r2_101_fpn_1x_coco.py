_base_ = 'ddod_r101_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://res2net101_v1d_26w_4s')))

work_dir = 'work_dirs/coco/ddod/ddod_r2_101_fpn_1x_coco'