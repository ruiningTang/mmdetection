_base_ = './faster_rcnn_r50_fpn_1x_voc0712_cocofmt.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='IoULoss', loss_weight=10.0))))

work_dir = './work_dirs/voc/reg_loss/faster_rcnn/faster_rcnn_r50_fpn_iou_1x_voc0712_cocofmt'

