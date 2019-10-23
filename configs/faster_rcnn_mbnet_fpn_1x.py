# model settings
# configs中保存了一些基础的模型配置文件 利用dict()设置一些键值对
model = dict(
    type='FasterRCNN', #model类型
    pretrained='torchvision://shufflenet_v2_x1_0', # 使用的预训练的模型
    # 这里设置网络的backbone
    backbone=dict(
        type='ShuffleNet', # Resnet50 backbone的类型
        g_groups=3,  #Resnet50的网络层数
        num_stages=3,  # Resnet50中有4个stage.每个stage包含了多个残差block 每个block的卷积层中第三维有着不同的维数
        out_indices=(0, 1, 2), #输出的stage序号
        frozen_stages=-1, #冻结的stage数量，表示这个stage不参与参数更新，-1表示全部都不冻结
        style='pytorch'), #设置网络风格为pytorch
    # 设置neck fpn这一步是在backbone提取出了图片的特征图之后做的，做出多种尺度的特征图。用于rpn等等。所以是网络的neck部分。
    neck=dict(
        type='FPN', #neck网络类型FPN
        in_channels=[256, 512, 1024, 2048], #输入的各个stage的通道数
        out_channels=256,  #输出的特征层的通道数
        num_outs=5), #输出的特征层的数量
    rpn_head=dict(
        type='RPNHead', #RPN的网络类型
        in_channels=256, #RPN的输入通道数
        feat_channels=256, #特征层的通道数 跟in_channels一致 一致为什么设置两个参数
        anchor_scales=[8], #anchor的大小，baselen=sqrt(w*h) 8*8的anchors
        anchor_ratios=[0.5, 1.0, 2.0], #每种scales对应的anchor的三种比例
        anchor_strides=[4, 8, 16, 32, 64], #在每个特征层上的anchor的步长，对应原图。neck中设置了5个特征层的数量，这里就有5种步长
        target_means=[.0, .0, .0, .0], #什么的均值，target是什么 按高斯分布归一化吗
        target_stds=[1.0, 1.0, 1.0, 1.0], #方差
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), #RPN的分类loss 是否使用sigmoid分类，sigmoid二进制分类，这里对应
        #论文中分类前景背景。对应的数值是前景背景的得分。False使用softmax分类。
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)), #RPN的回归loss回归bbox
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',   #ROI Extractor的类型
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2), #ROIlayer的类型，这里使用的是ROIAlign，输出尺寸为7，sample数为2
        #为什么会有sample数
        out_channels=256, #输出的通道数
        featmap_strides=[4, 8, 16, 32]), #特征图的步长 为什么特征图的步长只有4个，不应该跟neck里面一样是5个吗
    bbox_head=dict( #这里这个bbox_head应该指的是faster_rcnn的最后一部分
        type='SharedFCBBoxHead',  #全连接层的类型 FC BBOX HEAD  shared指什么
        num_fcs=2, # 两层全连接层
        in_channels=256, #输入通道数
        fc_out_channels=1024, #fc层的输出通道数
        roi_feat_size=7, #ROI特征层的尺寸
        num_classes=81, #分类器的数目+1，多了一个背景类别
        target_means=[0., 0., 0., 0.], #均值
        target_stds=[0.1, 0.1, 0.2, 0.2], #方差
        reg_class_agnostic=False, #是否采用class_agnostic的方法来预测，class_agnostic表示输出bbox时只考虑前景背景，后续分类的时候再根据
        # 该bbox在网络中的类别得分来分类，也就是说一个框可以对应多个类别
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),#这里涉及到了多分类因此use_sigmoid设置成了False，使用softmax
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))) #bbox回归依然使用SmoothL1
# model training and testing settings
train_cfg = dict( #网络训练相关的设置
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner', #RPN网络的正负样本划分
            pos_iou_thr=0.7,   #正样本的IOU阈值
            neg_iou_thr=0.3,   #负样本的IOU阈值
            min_pos_iou=0.3,   #正样本的IOU最小值，如果assign给gt的anchors中最大的IOU低于0.3，则忽略所有anchors，否则保留最大IOU的anchor
            ignore_iof_thr=-1), #忽略bbox的阈值，当gt中包含需要忽略的bbox时使用，-1表示不忽略
        sampler=dict(
            type='RandomSampler', #正负样本提取器的类型
            num=256, #需要提取的正负样本的数量
            pos_fraction=0.5, #正样本的比例
            neg_pos_ub=-1, #最大负样本比例，大于该比例的负样本忽略，-1表示不忽略。论文中提及负样本数目过多的问题，以及如何解决。
            add_gt_as_proposals=False), #把gt加入proposal中，这里没有添加
        allowed_border=0,  #允许在bbox周围扩展一定的像素
        pos_weight=-1,  #正样本的权重，-1表示保持原有的权重
        debug=False), #debug模式
    rpn_proposal=dict(  #rpn_proposal
        nms_across_levels=False, #在不同的fpn level上面nms
        nms_pre=2000, #nms之前2000个样本
        nms_post=2000,  #nms之后保留2000个样本
        max_num=2000, #最多2000个样本
        nms_thr=0.7, #与当前得分最高的框的IOU阈值
        min_bbox_size=0), #最小bbox的尺寸
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner', #正负样本划分
            pos_iou_thr=0.5, #正样本IOU阈值
            neg_iou_thr=0.5, #负样本IOU阈值
            min_pos_iou=0.5, #最小正样本阈值
            ignore_iof_thr=-1), #不忽略IOU阈值
        sampler=dict(
            type='RandomSampler', #正负样本提取器的类型
            num=512, #需要提取的正负样本数量
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(                  #推断时的rpn参数，train时调用了train和test两个cfg
    rpn=dict(
        nms_across_levels=False,  #所有的fpn层内nms
        nms_pre=1000,   #nms之前保留的得分最高的1000个proposal
        nms_post=1000,  #nms之后保留的得分最高的1000个proposal
        max_num=1000,   #处理后保留的proposal的数量
        nms_thr=0.7,    #nms的iou阈值
        min_bbox_size=0), #最小的bbox的大小
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100) #max_per_img为每张图片上的最多的bbox的数量
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05) #soft-nms参数
)
# dataset settings 有关数据集类型的设置
dataset_type = 'CocoDataset' #数据集的类型
data_root = 'data/coco/' #训练数据的文件根目录。这里我使用了ln -s链接
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) #输入图像初始化，减去mean，除以std。to_rgb表示bgr转为rgb
data = dict(
    imgs_per_gpu=1, #每个gpu计算的图像数量
    workers_per_gpu=1, #每个gpu分配的线程数
    train=dict(
        type=dataset_type,  #数据集的类型
        ann_file=data_root + 'annotations/instances_train2017.json', #数据集的annotation路径
        img_prefix=data_root + 'train2017/', #图片路径
        img_scale=(1333, 800), #输入图像尺寸
        img_norm_cfg=img_norm_cfg, #图像初始化参数
        size_divisor=32,    #对图像进行resize时的最小单位，32表示所有图像被resize成32的倍数
        flip_ratio=0.5,    #图像随机左右翻转的概率
        with_mask=False,   #训练时附带mask
        with_crowd=True,   #训练时附带困难样本
        with_label=True),  #训练时附带label
    val=dict(                 #验证阶段的配置
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    test=dict(                #推理阶段的配置
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001) #优化器，lr学习率，momentum为动量因子，weight_decay为权重衰减因子
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)) #梯度均衡参数
# learning policy
lr_config = dict(
    policy='step', #优化策略
    warmup='linear', #学习率增加的方法
    warmup_iters=500, #在初始的500次中学习率增加
    warmup_ratio=1.0 / 3, #起始的学习率
    step=[8, 11]) #在第8和11个循环的时候降低学习率
checkpoint_config = dict(interval=1)  #每1个epoch存储一次模型
# yapf:disable
log_config = dict(
    interval=50, #50个循环输出一次信息
    hooks=[
        dict(type='TextLoggerHook'), #控制台输出信息的风格
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12  # 最大的epoch数目
dist_params = dict(backend='nccl') #分布式参数
log_level = 'INFO' #输出信息的完整级别
work_dir = './work_dirs/faster_rcnn_r50_fpn_1x' #log文件和模型文件的存储路径
load_from = None #加载模型的路径，None表示从预训练的模型加载
resume_from = None  #恢复训练模型的路径
workflow = [('train', 1)]  #当前工作区的名称
