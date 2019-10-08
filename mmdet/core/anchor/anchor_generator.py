import torch


class AnchorGenerator(object):

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):  #ctr定义的是anchor的中心
        self.base_size = base_size  #anchors基础大小
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property #@装饰器给函数动态加上功能 把这个方法变成属性调用，没有setter，只读属性
    def num_base_anchors(self):  #查看有多少基础的anchors
        return self.base_anchors.size(0) #size（0）查看个数

    def gen_base_anchors(self): #生成基础的anchors
        w = self.base_size  #基础的anchors的w.h应该是一致的
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1) #应该是中心点的偏移量
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = torch.sqrt(self.ratios) #ratio的使用方法，并不是h*ratio，要保持anchors的面积不变，长宽变
        w_ratios = 1 / h_ratios  #如果是开根号，那么anchors的长度不会刚好是一个像素，那么这样怎么划anchors的区域呢
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1) #组成一个3X3的w矩阵,再把矩阵重组成一个size为9的Tensor
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1) #h对应w
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        # yapf: disable
        base_anchors = torch.stack( #base_anchors指的是没有经过nms的总共的anchors
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), #左上角
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)  #右下角
            ],
            dim=-1).round() #anchors的长度不是刚好一个像素，所以这里使用了round取整 这里是没有经过reg的anchors也不再feature map上，
        # 只是一些anchor的形状，只有9个
        # yapf: enable

        return base_anchors

    def _meshgrid(self, x, y, row_major=True): #生成栅格
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size #特征图的大小
        shift_x = torch.arange(0, feat_w, device=device) * stride #这几行是根据特征图的大小生成相应的栅格，左上角为0.0，栅格的步长为16
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y) #repeat这么多次生成的坐标岂不是很大
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1) #为什么弄成四列的，只是坐标两列不就行了：可能因为一个anchor有四个点。
        # feat_w*feat*h个坐标点
        shifts = shifts.type_as(base_anchors) #统一Tesnor的类型
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid
