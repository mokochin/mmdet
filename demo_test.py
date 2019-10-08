from mmdet.apis import init_detector, inference_detector, show_result

# 配置文件
config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
# 模型文件
checkpoint_file = 'checkpoint/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

# 初始化模型
model = init_detector(config_file, checkpoint_file)

# 测试一张图片
img = 'test.jpg'
result = inference_detector(model, img)
show_result(img, result, model.CLASSES)

# # 测试一系列图片
# imgs = ['test1.jpg', 'test2.jpg']
# for i, result in enumerate(inference_detector(model, imgs, device='cuda:0')):
#     show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))