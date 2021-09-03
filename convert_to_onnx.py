import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import *
from data import config
import numpy as np
import cv2
import tools
import time
from thop import profile

parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-v', '--version', default='yolov3',
                    help='yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny')
parser.add_argument('-d', '--dataset', default='coco-val',
                    help='voc, coco-val.')
parser.add_argument('-size', '--input_size', default=608, type=int,
                    help='input_size')
parser.add_argument('--trained_model', default='weights/yolov3_36.0_57.6.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS threshold')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use cuda.')

args = parser.parse_args()


if __name__ == '__main__':
    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # input size
    input_size = args.input_size

    # dataset
    if args.dataset == 'voc':
        print('test on voc ...')
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = 20

    elif args.dataset == 'coco-val':
        print('test on coco-val ...')
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = 80

    class_colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in
                    range(num_classes)]

    # model
    model_name = args.version
    print('Model: ', model_name)

    # load model and config file
    if model_name == 'yolov2_d19':
        from models.yolov2_d19 import YOLOv2D19 as yolo_net

        cfg = config.yolov2_d19_cfg

    elif model_name == 'yolov2_r50':
        from models.yolov2_r50 import YOLOv2R50 as yolo_net

        cfg = config.yolov2_r50_cfg

    elif model_name == 'yolov2_slim':
        from models.yolov2_slim import YOLOv2Slim as yolo_net

        cfg = config.yolov2_slim_cfg

    elif model_name == 'yolov3':
        from models.yolov3 import YOLOv3 as yolo_net

        cfg = config.yolov3_d53_cfg

    elif model_name == 'yolov3_spp':
        from models.yolov3_spp import YOLOv3Spp as yolo_net

        cfg = config.yolov3_d53_cfg

    elif model_name == 'yolov3_tiny':
        from models.yolov3_tiny import YOLOv3tiny as yolo_net

        cfg = config.yolov3tiny_cfg
    else:
        print('Unknown model name...')
        exit(0)

    # build model
    anchor_size = cfg['anchor_size_voc'] if args.dataset == 'voc' else cfg['anchor_size_coco']
    net = yolo_net(device=device,
                   input_size=(608, 608),
                   num_classes=num_classes,
                   trainable=False,
                   anchor_size=anchor_size)

    # load weight
    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.eval()

    model_path = "onnx/%s_608x608.onnx" % model_name

    dummy_input = torch.randn(1, 3, 608, 608).to("cpu")
    flops, _ = profile(net, inputs=(dummy_input,))
    print(flops)
    torch.onnx.export(net, dummy_input, model_path, opset_version=11, verbose=False, input_names=['input'],
                      output_names=['output'])
