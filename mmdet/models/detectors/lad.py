import torch
from mmcv.runner import _load_checkpoint
from ..builder import (DETECTORS, build_backbone, build_neck, build_head)
from .single_stage import SingleStageDetector

@DETECTORS.register_module()
class LAD(SingleStageDetector):
    """Label Assignment Distillation https://arxiv.org/abs/2108.10520"""

    def __init__(self,
                 # student
                 backbone,
                 neck,
                 bbox_head,
                 # teacher
                 teacher_backbone,
                 teacher_neck,
                 teacher_bbox_head,
                 teacher_pretrained,
                 # cfgs
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg,
                         test_cfg, pretrained)
        self.teacher_backbone = build_backbone(teacher_backbone)
        if teacher_neck is not None:
            self.teacher_neck = build_neck(teacher_neck)
        teacher_bbox_head.update(train_cfg=train_cfg)
        teacher_bbox_head.update(test_cfg=test_cfg)
        self.teacher_bbox_head = build_head(teacher_bbox_head)
        self.init_teacher_weights(teacher_pretrained=teacher_pretrained)

    def init_teacher_weights(self, teacher_pretrained):
        ckpt = _load_checkpoint(teacher_pretrained, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        teacher_ckpt = dict()
        for key in ckpt:
            teacher_ckpt['teacher_' + key] = ckpt[key]
        self.load_state_dict(teacher_ckpt, strict=False)
        print("Init teacher weights done")

    def teacher_eval(self):
        self.teacher_backbone.eval()
        self.teacher_neck.eval()
        self.teacher_bbox_head.eval()

    def student_eval(self):
        self.backbone.eval()
        self.neck.eval()
        self.bbox_head.eval()

    def student_train(self):
        self.backbone.train()
        self.neck.train()
        self.bbox_head.train()

    def extract_teacher_feat(self, img):
        x = self.teacher_backbone(img)
        if self.with_neck:
            x = self.teacher_neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)

        # forward student
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        # teacher infers Label Assignment results
        with torch.no_grad():
            # MUST force teacher to `eval` every training step, b/c at the
            # beginning of epoch, the runner calls all nn.Module elements
            # to be `train`
            self.teacher_eval()

            # assignment result is obtained based on only teacher
            x_teacher = self.extract_teacher_feat(img)
            la_results = self.teacher_bbox_head.forward_la(
                x_teacher, img_metas, gt_bboxes, gt_labels,
                gt_bboxes_ignore=None)

        # student receives the assignment results to learn
        losses = self.bbox_head.forward_train_wo_la(
            outs, img_metas, gt_bboxes, gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore, la_results=la_results)

        return losses