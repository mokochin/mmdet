import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmdet.core import auto_fp16, force_fp32, mask_target
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class KeyPointHead(nn.Module):

    # 选用fpn_backbone的mask分支

    def __init__(self,
                 num_convs=4,  #初始选用
                 roi_feat_size=14,  #初始roi尺寸
                 in_channels=256,  #head的第一层channel
                 out_channels=256,
                 conv_kernel_size=3,  #常规卷积核尺寸
                 upsample_method='deconv',  #初始4层之后是上采样层
                 upsample_ratio=2,  #上采样层设置
                 num_classes=18,
                 final_out_channel=17,  # 最终层的通道数， fcn_mask中是80，这里我只需要17个keypoint点
                 conv_final_kernel_size=1,  # 最后一层卷积层卷积核的尺寸
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss',use_mask=True,loss_weight=1.0,)): #loss

        super(KeyPointHead,self).__init__()

        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))

        self.num_convs=num_convs
        self.roi_feat_size=_pair(roi_feat_size)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conv_kernel_size=conv_kernel_size
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.loss_mask=build_loss(loss_mask)
        self.num_classes=num_classes
        self.num_keypoints = final_out_channel

        #4层卷积层
        self.convs=nn.ModuleList()
        for i in range(self.num_convs):
            in_channels=(
                self.in_channels if i==0 else self.out_channels
            )
            padding=(self.conv_kernel_size-1)//2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg
                )
            )

        #上采样层
        upsample_in_channel=self.out_channels
        self.upsample=nn.ConvTranspose2d(
            upsample_in_channel,
            self.out_channels,
            self.upsample_ratio,
            stride=self.upsample_ratio
        )

        #最后的one-hot mask层
        logits_in_channel = self.out_channels
        logits_out_channel=self.num_keypoints
        self.conv_logits=nn.Conv2d(
            logits_in_channel,logits_out_channel,1
        )
        self.relu=nn.ReLU(inplace=True)

class PersonKeypoints(Keypoints):
    NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]

#把身上的17个点连接起来
def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
        ]
    return kp_lines
PersonKeypoints.CONNECTIONS = kp_connections(PersonKeypoints.NAMES)

#keypoint的one-hot map
def keypoints_to_heat_map(keypoints, rois, heatmap_size):
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid
