_base_ = './faster_rcnn_r50_fpn_1x_voc0712_cocofmt.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='EIoULoss', linear=False, using_grad_factor=False, smooth_point=0.2, loss_weight=10.0))))

work_dir = './work_dirs/voc/reg_loss/faster_rcnn/faster_rcnn_r50_fpn_eiou-smooth_1x_voc0712_cocofmt'

