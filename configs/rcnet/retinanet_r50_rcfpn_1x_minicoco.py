_base_ = 'retinanet_r50_rcfpn_1x_coco.py'
#minicoco
data_root = '/home/amax/coco2017/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'annotations/instances_minitrain2017.json'))

work_dir = 'work_dirs/minicoco/rcnet/retinanet_r50_rcfpn_1x_minicoco'