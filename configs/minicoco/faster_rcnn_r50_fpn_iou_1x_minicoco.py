_base_ = './faster_rcnn_r50_fpn_1x_minicoco.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='IoULoss', loss_weight=10.0))))
