_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
	rpn_head=dict(
		loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        bbox_head=dict(
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))
