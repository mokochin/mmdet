import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms           #nms在roi proposal的时候就做了
from ..registry import HEADS
from .anchor_head import AnchorHead  #继承自AnchorHead


@HEADS.register_module
class RPNHead(AnchorHead): #nn.Module的子类的子类

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(2, in_channels, **kwargs) #2是rpn的2分类，从AnchorHead中继承不定数量的属性

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(   #rpn的三个网络层，其中feat.channels是256d
            self.in_channels, self.feat_channels, 3, padding=1) #n=3
        # class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # 二维卷积层, 输入的尺度是(N, C_in,H,W)，输出尺度（N,C_out,H_out,W_out）
        # N=batch_size;C=featur map dimension;H=high of feature map;W=wide of feature map.
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1) #sigmoid通道数每个anchor*2，卷积核大小为1，特征图的w,h不变
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1) #对应每个anchor4个量的回归。左上角点/中心点和长宽，卷积核1X1
        #num_anchors=9
        #问题：每个slide_window对应的9个anchors大小不一样，是怎样在同一个网络里计算的。预测应该是9个anchors组成一个样本输入。

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)  #mmcv.cnn中利用nn初始化权值，偏差的方法

        # def normal_init(module, mean=0, std=1, bias=0):
        #     nn.init.normal_(module.weight, mean, std)
        #     if hasattr(module, 'bias') and module.bias is not None:   #hasattr是python的内置属性，用于判断对象是否有对应的属性
        #         nn.init.constant_(module.bias, bias)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)   #inplace会替代原始的variable
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred #rpn的前向传播

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(RPNHead, self).loss( #继承AnchorHead父类的loss方法
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def get_bboxes_single(self,
                          cls_scores,  #scores如果是sigmoid就是输出尺度4维，通道为1，如果不是就是classes,结合AnchorHead看，中间有个-1的操作。
                          bbox_preds,  #pred有4列 对应回归的四个参数
                          mlvl_anchors, #结合fpn的多层anchors
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = [] #多层的proposal
        for idx in range(len(cls_scores)): #对一个batch中的每个图像进行循环
            rpn_cls_score = cls_scores[idx] #例：图像1的多通道cls_score
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:] #确认特征图大小一致
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0) #这里为什么要换一下维度呢，这里rpn_cls_score只有3维，0是通道也是anchors的数量，1、2分别是hw
            #这里换维度的原因是按照分类组合数据，结合后面reshape
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1) #size=[h*w*1]
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4) #同分类
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre: #nms-pre=1000
                _, topk_inds = scores.topk(cfg.nms_pre) #这里指定topk的个数为1000
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :] #nms的过程，top_k个anchors是nms之后保留的
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape) #根据回归的值矫正bbox，在特征图上完成proposal
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze() #nonzero制作索引剔除不符合大小的anchors
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr) #nms
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr) #fpn对应的nms方案
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]  #非fpn
            num = min(cfg.max_num, proposals.shape[0]) #保留proposal的个数不超过max_num
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals
