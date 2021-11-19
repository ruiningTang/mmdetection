_base_ = './tood_r101_fpn_1x_minicoco.py'
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
work_dir = 'work_dirs/minicoco/tood/tood_r2-101_fpn_1x_minicoco'