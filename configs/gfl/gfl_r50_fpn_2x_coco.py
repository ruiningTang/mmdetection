_base_ = './gfl_r50_fpn_1x_coco.py'
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
work_dir = 'work_dirs/coco/gfl/gfl_r50_fpn_2x_coco'