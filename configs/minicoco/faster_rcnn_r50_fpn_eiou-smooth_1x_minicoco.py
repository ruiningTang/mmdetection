_base_ = './faster_rcnn_r50_fpn_1x_minicoco.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='EIoULoss', loss_weight=10, linear=False, smooth_point=0.2, using_grad_factor=False))))
#optimizer_config = dict(
 #   _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
work_dir = './work_dirs/minicoco/eious/faster_rcnn/faster_rcnn_r50_fpn_eiou-smooth_1x_minicoco'
