import threading
from typing import Tuple

from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import QThread

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
# https://github.com/pytorch/pytorch/issues/3678
import sys
import numpy as np
sys.path.insert(0, './yolov5')


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img, (x1, y1),(x2,y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


# Class / Thread for creating a new project
class ProjectCreateThread(threading.Thread):
    def __init__(self, opt):
        super(ProjectCreateThread, self).__init__()
        self.opt = opt
        self.setDaemon(True)
        self.exitcode = 0
        self.message = ''
        self.video_path = self.opt['source']
        self.progress_bar = [0, 0]
        if self.opt['project_name'] in os.listdir(self.opt['project_path']):
            self.exitcode = -1
            self.message = '该文件夹下已存在此项目'
        else:
            os.makedirs(self.opt['output'])
            with open(self.opt['output'] + '/' + self.opt['project_name'] + '.vdep', 'w') as f:
                f.close()

    def run(self) -> Tuple[int, str]:
        if self.exitcode != 0:
            return self.exitcode, self.message
        with torch.no_grad():
            self.detect()

    def detect(self):
        progress = 0
        out, source, weights, view_img, save_txt, imgsz = \
            self.opt['output'], self.opt['source'], self.opt['weights'], self.opt['view_img'], self.opt['save_txt'], self.opt['img_size']
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(self.opt['config_deepsort'])
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize
        device = select_device(self.opt['device'])
        # if os.path.exists(out):
        #     shutil.rmtree(out)  # delete output folder
        # os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        # google_utils.attempt_download(weights)
        model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
        # model = torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
        # model.fuse()
        model.to(device).eval()
        if half:
            model.half()  # to FP16

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            # view_img = True
            save_img = True
            dataset = LoadImages(source, self.opt['frame_range'], img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        save_path = str(Path(out)) + '/video.mkv'
        txt_path = str(Path(out)) + '/track_results.txt'
        detection_path = str(Path(out)) + '/detection_results.txt'

        for frame_idx, (path, img, im0s, vid_cap, caption) in enumerate(dataset):
            self.progress_bar = dataset.frame_progress
            # print(self.progress_bar)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=self.opt['augment'])[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt['conf_thres'], self.opt['iou_thres'], classes=self.opt['classes'],
                                       agnostic=self.opt['agnostic_nms'])
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    bbox_xywh = []
                    confs = []

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        img_h, img_w, _ = im0.shape
                        x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])
                        with open(detection_path, 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (frame_idx, x_c, y_c, bbox_w, bbox_h))

                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    # Pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, im0)

                    # draw boxes for visualization
                    # if len(outputs) > 0:
                    #     bbox_xyxy = outputs[:, :4]
                    #     identities = outputs[:, -1]
                    #     draw_boxes(im0, bbox_xyxy, identities)

                    # Write MOT compliant results to file
                    if save_txt and len(outputs) != 0:
                        for j, output in enumerate(outputs):
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2]
                            bbox_h = output[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1,
                                                               -1))  # label format

                # Print time (inference + NMS)
                print('%s %sDone. (%.3fs)' % (caption, s, t2 - t1))

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path,
                                                     cv2.VideoWriter_fourcc(*self.opt['fourcc']), fps, (w, h))
                    vid_writer.write(im0)

        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()
        print('Done. (%.3fs)' % (time.time() - t0))
