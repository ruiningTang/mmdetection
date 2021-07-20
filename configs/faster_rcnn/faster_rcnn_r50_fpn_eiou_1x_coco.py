_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='EIoULoss', loss_weight=10.0, linear=True, using_grad_factor=False))))

work_dir = './work_dirs/coco/eious/faster_rcnn/faster_rcnn_r50_fpn_eiou_1x_coco'
