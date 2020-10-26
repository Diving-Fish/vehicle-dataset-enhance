import os
from math import ceil

import cv2

from lib.create_project_window import Ui_Dialog
from PyQt5.QtWidgets import QDialog, QFileDialog
from lib.call_dialog import Dialog
from lib.call_progress_dialog import ProgressDialog


class CreateProjectDialog(QDialog, Ui_Dialog):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setupUi(self)
        self.chooseDeepsortButton.clicked.connect(lambda: self.select_path(self.deepsortPath))
        self.chooseVideoPathButton.clicked.connect(lambda: self.select_path(self.videoPath))
        self.chooseProjectPathButton.clicked.connect(lambda: self.select_directory(self.projectPath))
        self.chooseWeightsButton.clicked.connect(lambda: self.select_path(self.weightsPath))
        self.confirmButton.clicked.connect(self.create_progress)
        self.cancelButton.clicked.connect(self.close)
        self.videoPath.textChanged.connect(self.pre_read_video)
        self.startTime.setEnabled(False)
        self.endTime.setEnabled(False)
        self.weightsPath.setText('./yolov5/weights/yolov5x.pt')
        self.deepsortPath.setText('./deep_sort/configs/deep_sort.yaml')
        self.iouThres.setText('0.5')
        self.confThres.setText('0.4')
        self.fourcc.setText('mp4v')
        self.imgSize.setText('640')

        self.show()

    def pre_read_video(self):
        try:
            cap = cv2.VideoCapture(self.videoPath.text())
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.startTime.setEnabled(True)
            self.endTime.setEnabled(True)
            self.startTime.setText("0:00")
            sec = int(ceil(self.nframes / self.fps))
            minute = int(sec / 60)
            self.endTime.setText(f"{minute}:{sec % 60}")
            cap.release()
        except Exception as e:
            self.dialog = Dialog("请选择正确的视频文件")

    def select_path(self, target):
        path, file_type = QFileDialog.getOpenFileName(self, '选择文件', os.getcwd(), "所有文件(*)")
        target.setText(path)

    def select_directory(self, target):
        path = QFileDialog.getExistingDirectory(self, '选择文件夹', os.getcwd())
        target.setText(path)

    def create_progress(self):
        try:
            st = self.startTime.text().split(':')
            ed = self.endTime.text().split(':')
            start_f = self.fps * (int(st[0]) * 60 + int(st[1]))
            end_f = self.fps * (int(ed[0]) * 60 + int(ed[1]))
            opt = {
                'project_path': self.projectPath.text(),
                'project_name': self.projectName.text(),
                'output': self.projectPath.text() + '/' + self.projectName.text(),
                'source': self.videoPath.text(),
                'weights': self.weightsPath.text(),
                'view_img': False,
                'save_txt': True,
                'img_size': int(self.imgSize.text()),
                'conf_thres': float(self.confThres.text()),
                'iou_thres': float(self.iouThres.text()),
                'fourcc': self.fourcc.text(),
                'device': '',
                'classes': [2, 5, 7],
                'agnostic_nms': False,
                'augment': False,
                'config_deepsort': self.deepsortPath.text(),
                'frame_range': (start_f, end_f)
            }
            self.progress = ProgressDialog(opt, self)
        except Exception as e:
            self.dialog = Dialog("请正确填写参数")

    def finish_create(self):
        self.parent.finish_create(self.projectPath.text() + '/' + self.projectName.text(),
                                  self.projectPath.text() + '/' + self.projectName.text() + '/' + self.projectName.text() + '.vdep')
        self.close()

