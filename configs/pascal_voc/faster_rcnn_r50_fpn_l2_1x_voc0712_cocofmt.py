_base_ = './faster_rcnn_r50_fpn_1x_voc0712_cocofmt.py'
model = dict(
    rpn_head=dict(loss_bbox=dict(type='MSELoss', loss_weight=1.0)),
    roi_head=dict(
        bbox_head=dict(
            loss_bbox=dict(type='MSELoss', loss_weight=1.0))))

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

work_dir = './work_dirs/voc/reg_loss/faster_rcnn/faster_rcnn_r50_fpn_l2_1x_voc0712_cocofmt'
