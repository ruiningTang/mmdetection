_base_ = './tood_r50_fpn_1x_coco.py'
model = dict(
    bbox_head=dict(
        anchor_type='anchor_based'))
work_dir = 'work_dirs/coco/tood/tood_r50_fpn_anchor_based_1x_coco'