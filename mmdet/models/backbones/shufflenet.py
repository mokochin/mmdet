import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ShuffleUnit(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 padding=1,
                 bias=True,
                 groups=3,
                 dilation=1,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 grouped_conv=True,
                 norm_cfg=dict(type='BN',),
                 combine='add'):
        super(ShuffleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.dilation = dilation
        self.combine = combine

        #定义norm
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.bottleneck_channels, postfix=1)  #norm1由conv1 compress使用
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, self.bottleneck_channels, postfix=2)  #norm2由conv2 dw_conv使用
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, self.out_channels, postfix=3)  #norm3由conv3 expand使用

        #定义bottleneck的尺寸
        self.bottleneck_channels = in_channels//4
        self.first_1x1_groups = self.groups if grouped_conv else 1

        #compress:compress1x1、bn、relu
        self.conv1 = build_conv_layer(
            conv_cfg, in_channels, self.bottleneck_channels,
            kernel_size=1, stride=stride, padding=padding, bias=bias, groups=self.groups)
        self.add_module(self.norm1_name, norm1) #nn里面添加层的方法 net1.add_module('batchnorm', nn.BatchNorm2d(3))

        #bottleneck:dw_conv、bn
        self.conv2 = build_conv_layer(
            conv_cfg, self.bottleneck_channels, self.bottleneck_channels,
            kernel_size=3, stride=self.depthwise_stride, groups=self.bottleneck_channels)
        self.add_module(self.norm2_name, norm2)

        #expand:expand1x1、bn
        self.conv3 = build_conv_layer(
            conv_cfg, self.bottleneck_channels, out_channels,
            kernel_size=1, group=self.groups)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)

        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 2
            self._combine_func = self._concat

            # ensure output of concat has the same channels as
            # original output channels.
            self.out_channels -= self.in_channels
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(self.combine))

    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)
        # define the type of ShuffleUnit

    @property
    def norm1(self):
        return getattr(self, self.norm1_name) # getattr返回一个对象的属性值

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        out = self._combine_func(residual, out)

        return out

def make_shuffle_layer(block, in_channels, out_channels, blocks,
                   stride=1, dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN')):

    layers = []
    layers.append(
        block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            ))

    in_channels = out_channels * block.expansion

    for i in range(1, blocks):
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                ))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ShuffleNet(nn.Module):

    arch_settings = {
        1: (ShuffleUnit, (144, 288, 576)),
        2: (ShuffleUnit, (200, 400, 800)),
        3: (ShuffleUnit, (240, 480, 960)),
        4: (ShuffleUnit, (272, 544, 1088)),
        8: (ShuffleUnit, (384, 768, 1536))
    }

    blocks_settings = (3, 7, 3)

    #可能没用
    stride_settings = {
        0: (2, 1, 1, 1),
        1: (2, 1, 1, 1, 1, 1, 1, 1),
        2: (2, 1, 1, 1),
    }

    plane_settings = {
        0: (2, 1, 1, 1),
        1: (2, 1, 1, 1, 1, 1, 1, 1),
        2: (2, 1, 1, 1),
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=True):
        super(ShuffleNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = make_shuffle_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                )
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1) #更改layer name 好用
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    def _make_stem_layer(self): #这个是最前面的层
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            24,
            kernel_size=3,
            stride=2,
            bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, ShuffleUnit):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(ShuffleNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

net=ShuffleNet(28)
net.eval()
print(net)