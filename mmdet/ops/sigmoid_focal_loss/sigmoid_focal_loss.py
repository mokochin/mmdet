import torch.nn as nn
from torch.autograd import Function #继承
from torch.autograd.function import once_differentiable

from . import sigmoid_focal_loss_cuda

#alpha、gamma调整简单样本的权值 详情见https://www.cnblogs.com/king-lps/p/9497836.html
#ctx is a context object that can be used to stash information for backward computation

class SigmoidFocalLossFunction(Function): #自定义的支持反向求导的函数，属于扩展autograd，自定义的函数需要继承autograd.Function
#扩展autograd 详情见https://blog.csdn.net/tsq292978891/article/details/79364140
    @staticmethod #静态函数 类或实例均可调用
    def forward(ctx, input, target, gamma=2.0, alpha=0.25): #ctx相当于self，其中gamma=2和alpha=0.25是论文里说的最优值
        ctx.save_for_backward(input, target)
        num_classes = input.shape[1]  #input的列数就是类别数
        ctx.num_classes = num_classes  #写到ctx里
        ctx.gamma = gamma    #focal_loss里面的gamma
        ctx.alpha = alpha    #alpha

        loss = sigmoid_focal_loss_cuda.forward(input, target, num_classes,
                                               gamma, alpha) #由这个函数计算
        return loss #loss就是output

    @staticmethod
    @once_differentiable #只能求导一次
    def backward(ctx, d_loss):
        input, target = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous() # Variable的连续化
        d_input = sigmoid_focal_loss_cuda.backward(input, target, d_loss,
                                                   num_classes, gamma, alpha)
        return d_input, None, None, None, None #其他四个参数不需要求导


sigmoid_focal_loss = SigmoidFocalLossFunction.apply #调用实现的Function


# TODO: remove this module
class SigmoidFocalLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        assert logits.is_cuda
        loss = sigmoid_focal_loss(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(gamma={}, alpha={})'.format(
            self.gamma, self.alpha)
        return tmpstr
