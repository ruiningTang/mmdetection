import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet


class GABottleneck(_Bottleneck):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 scales=4,
                 branch=2,
                 base_width=8,
                 base_channels=64,
                 stage_type='normal',
                 **kwargs):
        """Bottle2neck block for Res2Net.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(GABottleneck, self).__init__(inplanes, planes, **kwargs)
        assert scales > 1, 'Res2Net degenerates to ResNet when scales = 1.'
        width = int(math.floor(self.planes * (base_width / base_channels)))
        path = int(math.floor(scales/branch))
        #print('width=',width)

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width * scales, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width * scales,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        if stage_type == 'stage' and self.conv2_stride != 1:
            self.pool = nn.AvgPool2d(
                kernel_size=3, stride=self.conv2_stride, padding=1)
        convs = []
        bns = []

        convs_1=[]
        bns_1=[]

        convs_2=[]
        bns_2=[]

        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            for i in range(path - 1):
                convs.append(
                    build_conv_layer(
                        self.conv_cfg,
                        width,
                        width,
                        kernel_size=3,
                        stride=self.conv2_stride,
                        padding=self.dilation,
                        dilation=self.dilation,
                        bias=False))
                bns.append(
                    build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)

            for i in range(path - 1):
                convs_1.append(
                    build_conv_layer(
                        self.conv_cfg,
                        width,
                        width,
                        kernel_size=3,
                        stride=self.conv2_stride,
                        padding=self.dilation,
                        dilation=self.dilation,
                        bias=False))
                bns_1.append(
                    build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])
            self.convs_1 = nn.ModuleList(convs_1)
            self.bns_1 = nn.ModuleList(bns_1)

            for i in range(branch - 1):
                convs_2.append(
                    build_conv_layer(
                        self.conv_cfg,
                        path*width,
                        path*width,
                        kernel_size=3,
                        padding=1,
                        bias=False))
                bns_2.append(
                    build_norm_layer(self.norm_cfg, path*width, postfix=i + 1)[1])
            self.convs_2 = nn.ModuleList(convs_2)
            self.bns_2 = nn.ModuleList(bns_2)

        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            for i in range(path - 1):
                convs.append(
                    build_conv_layer(
                        self.conv_cfg,
                        width,
                        width,
                        kernel_size=3,
                        stride=self.conv2_stride,
                        padding=self.dilation,
                        dilation=self.dilation,
                        bias=False))
                bns.append(
                    build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)

            for i in range(path - 1):
                convs_1.append(
                    build_conv_layer(
                        self.conv_cfg,
                        width,
                        width,
                        kernel_size=3,
                        stride=self.conv2_stride,
                        padding=self.dilation,
                        dilation=self.dilation,
                        bias=False))
                bns_1.append(
                    build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])
            self.convs_1 = nn.ModuleList(convs_1)
            self.bns_1 = nn.ModuleList(bns_1)

            for i in range(branch - 1):
                convs_2.append(
                    build_conv_layer(
                        self.dcn,
                        path*width,
                        path*width,
                        kernel_size=3,
                        padding=1,
                        bias=False))
                bns_2.append(
                    build_norm_layer(self.norm_cfg, path*width, postfix=i + 1)[1])
            self.convs_2 = nn.ModuleList(convs_2)
            self.bns_2 = nn.ModuleList(bns_2)

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width * scales,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.stage_type = stage_type
        self.scales = scales
        self.width = width
        self.path = path
        self.branch = branch
        delattr(self, 'conv2')
        delattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            #print('out = ',out.shape)
            x_tmp = torch.split(out, self.path*self.width, 1)
            x1=x_tmp[0]
            x2=x_tmp[1]
            #print('input x:',x1.shape,x2.shape)

            spx1 = torch.split(x1, self.width, 1)
            #print(len(spx1))
            sp1 = self.convs[0](spx1[0].contiguous())
            sp1 = self.relu(self.bns[0](sp1))
            out1 = sp1
            for i in range(1, self.path - 1):
                if self.stage_type == 'stage':
                    sp1 = spx1[i]
                else:
                    sp1 = sp1 + spx1[i]
                sp1 = self.convs[i](sp1.contiguous())
                sp1 = self.relu(self.bns[i](sp1))
                out1 = torch.cat((out1, sp1), 1)

            #print('output y:',out1.shape,spx1[-1].shape)
            if self.stage_type == 'normal' or self.conv2_stride == 1:
                out1 = torch.cat((out1, spx1[-1]), 1)
            elif self.stage_type == 'stage':
                out1 = torch.cat((out1, self.pool(spx1[-1])), 1)

            spx2 = torch.split(x2, self.width, 1)
            sp2 = self.convs_1[0](spx2[0].contiguous())
            sp2 = self.relu(self.bns_1[0](sp2))
            out2 = sp2
            for i in range(1, self.path - 1):
                if self.stage_type == 'stage':
                    sp2 = spx2[i]
                else:
                    sp2 = sp2 + spx2[i]
                sp2 = self.convs_1[i](sp2.contiguous())
                sp2 = self.relu(self.bns_1[i](sp2))
                out2 = torch.cat((out2, sp2), 1)

            if self.stage_type == 'normal' or self.conv2_stride == 1:
                out2 = torch.cat((out2, spx2[-1]), 1)
            elif self.stage_type == 'stage':
                out2 = torch.cat((out2, self.pool(spx2[-1])), 1)



            out2 = out1 + out2
            out2 = self.convs_2[0](out2)
            out2 = self.relu(self.bns_2[0](out2))
            #print('output y:', out1.shape, out2.shape)
            out = torch.cat((out1,out2),1)
            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class GALayer(nn.Sequential):
    """Res2Layer to build Res2Net style backbone.
    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottle2neck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        scales (int): Scales used in Res2Net. Default: 4
        base_width (int): Basic width of each scale. Default: 26
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 scales=4,
                 base_width=8,
                 branch=2,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False),
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1],
            )

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                scales=scales,
                branch=branch,
                base_width=base_width,
                stage_type='stage',
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    scales=scales,
                    base_width=base_width,
                    branch=branch,
                    **kwargs))
        super(GALayer, self).__init__(*layers)


@BACKBONES.register_module()
class GANet(ResNet):
    """Res2Net backbone.
    Args:
        scales (int): Scales used in Res2Net. Default: 4
        base_width (int): Basic width of each scale. Default: 26
        depth (int): Depth of res2net, from {50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Res2net stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottle2neck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:
            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    Example:
        >>> from mmdet.models import Res2Net
        >>> import torch
        >>> self = Res2Net(depth=50, scales=4, base_width=26)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 256, 8, 8)
        (1, 512, 4, 4)
        (1, 1024, 2, 2)
        (1, 2048, 1, 1)
    """

    arch_settings = {
        50: (GABottleneck, (3, 4, 6, 3))
    }

    def __init__(self,
                 scales=4,
                 base_width=8,
                 branch=2,
                 style='pytorch',
                 deep_stem=True,
                 avg_down=True,
                 **kwargs):
        self.scales = scales
        self.base_width = base_width
        self.branch = branch
        super(GANet, self).__init__(
            style='pytorch', deep_stem=True, avg_down=True, **kwargs)

    def make_res_layer(self, **kwargs):
        return GALayer(
            scales=self.scales,
            base_width=self.base_width,
            branch=self.branch,
            base_channels=self.base_channels,
            **kwargs)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, GABottleneck):
                        # dcn in Res2Net bottle2neck is in ModuleList
                        for n in m.convs:
                            if hasattr(n, 'conv_offset'):
                                constant_init(n.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, GABottleneck):
                        constant_init(m.norm3, 0)
        else:
            raise TypeError('pretrained must be a str or None')