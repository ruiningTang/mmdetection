_base_ = './faster_rcnn_r50_fpn_1x_voc0712_cocofmt.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

work_dir = './work_dirs/voc/faster_rcnn/faster_rcnn_r101_fpn_1x_voc0712_cocofmt'
