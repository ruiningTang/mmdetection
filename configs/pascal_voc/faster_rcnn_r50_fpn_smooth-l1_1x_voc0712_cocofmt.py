_base_ = './faster_rcnn_r50_fpn_1x_voc0712_cocofmt.py'
model = dict(
    rpn_head=dict(loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        bbox_head=dict(
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))

work_dir = './work_dirs/voc/reg_loss/faster_rcnn/faster_rcnn_r50_fpn_smooth-l1_1x_voc0712_cocofmt'
