_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='CIoULoss', loss_weight=12.0))))
work_dir = 'work_dirs/coco/faster_rcnn/faster_rcnn_r50_fpn_ciou-w12_1x_coco'