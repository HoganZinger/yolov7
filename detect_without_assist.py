"""
@author : hogan
time:
function:
"""
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from experimental import attempt_load
from datasets import LoadStreams, LoadImages, letterbox
from general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from plots import plot_one_box
from torch_utils import load_classifier, time_synchronized, TracedModel

import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import random

def draw_res(img, infos, rowLabels):
    colLabels = ['名称', '相似度']

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig, ax = plt.subplots(1, 2)

    # 展示图片
    ax[0].axis('off')
    ax[0].imshow(img)

    # 展示列表信息
    ax[1].axis('off')
    if len(rowLabels) == 0:
        ax[1].text(0.25, 0.5, '没有检测出目标')
        return plt
    ax[1].table(cellText=infos,
                colLabels=colLabels,
                rowLabels=rowLabels,
                cellLoc='center',
                rowLoc='center',
                loc="center")

    # 保存结果
    return plt

def detect(bar, window, device, source, target, name, weights, classes=None, save_conf=False, agnostic_nms=False, img_size=224, conf_thres=0.6, iou_thres=0.45, view_img=False, save_txt=False, trace=True, save_img=True, exist_ok=False, augment=False):
    source = source
    weights = weights
    view_img = view_img
    imgsz = img_size
    save_img = save_img and not source.endswith('.txt')
    save_path = target + name

    # Initialize
    set_logging()

    # Load model
    model = torch.load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    model.eval()
    model.to(device)

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    nums = len(dataset)
    bar.place(relx=0.27, width=300, height=10, rely=0.72)
    bar['maximum'] = nums
    bar['value'] = 0

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for path, img, im0s, vid_cap in dataset:
        img0 = im0s
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        plt = process(img0, model, imgsz, stride, device, augment, conf_thres, iou_thres, classes, agnostic_nms, save_img, view_img)

        # Save results (image with detections)
        filepath = save_path + '/' + path.split('\\')[-1]
        if save_img:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')

        bar['value'] += 1
        window.update()

def detect_one(img0, device, weights, classes=None, save_conf=False, agnostic_nms=False, img_size=224, conf_thres=0.6, iou_thres=0.45, view_img=False, save_txt=False, trace=True, save_img=True, exist_ok=False, augment=False):
    weights = weights
    view_img = view_img
    save_txt = save_txt
    imgsz = img_size
    trace = trace

    # Initialize
    set_logging()

    # Load model
    model = torch.load(weights, map_location=device)  # load FP32 model

    # stride = int(model.stride.max())  # model stride
    strides = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            strides.append(module.stride[0])
    stride = max(strides)

    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    model.eval()
    model.to(device)

    plt = process(img0, model, imgsz, stride, device, augment, conf_thres, iou_thres, classes, agnostic_nms, save_img, view_img)

    return plt

def process(img0, model, imgsz, stride, device, augment, conf_thres, iou_thres, classes, agnostic_nms, save_img, view_img):
    # split image into 1024 * 1024
    patches = []
    (width, length, depth) = img0.shape
    if width - 2000 < 500 and length - 2000 < 500:
        cut_width = 2048
        cut_length = 2048
    else:
        cut_width = 1024
        cut_length = 1024
    num_width = int(width / cut_width)
    num_length = int(length / cut_length)
    if cut_width * num_width < width:
        num_width += 1
    if cut_length * num_length < length:
        num_length += 1
    for i in range(0, num_width):
        for j in range(0, num_length):
            temp = img0[i * cut_width: min(width, (i + 1) * cut_width),
                   j * cut_length: min(length, (j + 1) * cut_length), :]
            patches.append(temp)

    infos = []
    rowLabels = []
    res_patches = []
    cnt = 0
    for img0 in patches:
        # Padded resize
        img = letterbox(img0, imgsz, stride=stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Get names and colors
        # names = model.module.names if hasattr(model, 'module') else model.names
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        names = model.fc.weight  # 获取模型的类别名称，以张量形式存储
        names = [str(name) for name in names]
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        ori_img = img0
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', ori_img

            new_det = []
            if len(det):
                for i in range(0, len(det)):
                    # Rescale boxes from img_size to im0 size
                    pos = scale_coords(img.shape[2:], det[i, :4], img.shape[2:]).round()
                    xl = int(pos[0])
                    xr = int(pos[2])
                    yl = int(pos[1])
                    yr = int(pos[3])
                    if (xr - xl) < 224 * 0.2 or (yr - yl) < 224 * 0.2:
                        continue

                    if det[i, 4] < 0.7:
                        pos = scale_coords(img.shape[2:], det[i, :4], im0.shape).round()
                        xl = int(pos[0])
                        xr = int(pos[2])
                        yl = int(pos[1])
                        yr = int(pos[3])

                        new_img = torch.tensor(ori_img[yl:yr, xl:xr, :]).detach().cpu().numpy().astype(np.uint8)
                        new_img = cv2.resize(new_img, (224, 224))
                        new_img = torch.FloatTensor(np.array([new_img])).permute(0, 3, 1, 2)
                        new_img = new_img.to(device)

                        out = model(new_img)  # Perform classification using the model

                        out = F.softmax(out, dim=1)
                        maxx, pre = torch.max(out.data, 1)
                        if maxx < 0.7:
                            continue
                        else:
                            if maxx > det[i, 4]:
                                det[i, 5] = pre
                            det[i, 4] = max(maxx, det[i, 4])
                            new_det.append(det[i].detach().cpu().numpy())
                    else:
                        det[i, :4] = scale_coords(img.shape[2:], det[i, :4], im0.shape).round()
                        new_det.append(det[i].detach().cpu().numpy())
                if len(new_det) > 0:
                    new_det = torch.tensor(np.array(new_det))
                    det = new_det
                    det[:, :4] = det[:, :4].round()
                    for *xyxy, conf, cls in reversed(det):
                        if xyxy[2] - xyxy[0] < 10 or xyxy[3] - xyxy[1] < 10:
                            continue

                        if save_img or view_img:
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, cnt, label=label, color=colors[int(cls)],
                                         line_thickness=10)
                            conf = label[-4:]
                            if 'side-complete' in label:
                                name = '南方菟丝子-侧面观-完整'
                            elif 'side-incomplete' in label:
                                name = '南方菟丝子-侧面观-不完整'
                            else:
                                name = '南方菟丝子-表面观'
                            infos.append([name, conf])
                            rowLabels.append(cnt)
                            cnt += 1
            im0 = cv2.cvtColor(np.asarray(im0), cv2.COLOR_RGB2BGR)
            res_patches.append(im0)

    img0 = np.zeros((width, length, depth), dtype=np.uint8)
    index = 0
    for i in range(0, num_width):
        for j in range(0, num_length):
            img0[i * cut_width: min(width, (i + 1) * cut_width), j * cut_length: min(length, (j + 1) * cut_length), :] = \
                res_patches[index]
            index += 1
    plt = draw_res(img0, infos, rowLabels)

    return plt

