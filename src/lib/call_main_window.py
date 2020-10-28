import threading
import time
from collections import defaultdict

from PyQt5.QtWidgets import QFileDialog, QMainWindow, QAction
from PyQt5.QtGui import QImage, QPixmap

from lib.call_dialog import Dialog
from lib.main_window import Ui_MainWindow
from lib.call_create_project_window import CreateProjectDialog
from lib.call_setting_dialog import SettingDialog
import os
from lib.setting_manager import SettingManager
from cv2 import cv2


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.createProjectButton.triggered.connect(self.new_project)
        self.loadProjectButton.triggered.connect(self.load_project)
        self.settingButton.triggered.connect(self.setting)
        self.exportData.setEnabled(False)
        self.saveProjectButton.setEnabled(False)
        self.groupBox.setHidden(True)
        self.groupBox_2.setHidden(True)
        self.show()

    def setting(self):
        self.setting_dialog = SettingDialog()

    def finish_create(self, path, project_path):
        self.load(path, project_path)

    def new_project(self):
        self.createProjectDialog = CreateProjectDialog(self)

    def load_project(self):
        self.openDialog = QFileDialog()
        self.openDialog.setFileMode(QFileDialog.ExistingFile)
        path, file_type = self.openDialog.getOpenFileName(self, 'Choose a File', os.getcwd(),
                                                          "Vehicle Dataset Enhance Project(*.vdep)")
        if path == "":
            return
        self.load(path[0:path.rindex('/')], path)

    def load(self, project_dir, path):
        try:
            # Init mark dict
            self.mark_map = {}
            cap = cv2.VideoCapture(project_dir + '/video.mkv')
            fd = open(project_dir + '/track_results.txt', 'r')
            fd2 = open(project_dir + '/detection_results.txt', 'r')
            detections = defaultdict(list)
            # Load detections
            for line in fd2:
                arr = line.strip().split(' ')
                detections[int(arr[0])].append([float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4])])
            # Load tracks
            track_results = defaultdict(list)
            identities = set([])
            for line in fd:
                arr = line.strip().split(' ')
                track_results[int(arr[0])].append([int(arr[2]), int(arr[3]), int(arr[4]), int(arr[5]), int(arr[1])])
                identities.add(int(arr[1]))
            # Load progress
            with open(path, 'r') as fd3:
                for line in fd3:
                    arr = line.strip().split(' ')
                    if len(arr) == 0:
                        continue
                    self.markmap[int(arr[0])] = (int(arr[1]), int(arr[2]))
        except Exception as e:
            print(e)
            self.dialog = Dialog("请选择正确的工程文件")
            return
        self.init_video(cap, track_results, detections, identities)

    def init_video(self, cap, track_results, detections, identities):
        # Show Group Box
        self.groupBox.setHidden(False)
        self.groupBox_2.setHidden(False)

        self.identity = 0
        self.identities = identities
        self.track_results = track_results
        self.detections = detections
        self.cap = VideoThread(cap, self)
        self.cap.start()
        # Init Slider
        self.videoSlider.setMinimum(0)
        self.videoSlider.setMaximum(self.cap.frame_count - 1)
        # Init button functions
        self.playButton.clicked.connect(self.play)
        self.prevFrameButton.clicked.connect(lambda: self.change_frame(-1))
        self.nextFrameButton.clicked.connect(lambda: self.change_frame(1))
        self.jumpFrameButton.clicked.connect(self.jump_to_next_unmarked_vehicle)
        self.markButton.clicked.connect(self.mark)
        self.ignoreButton.clicked.connect(self.ignore)
        self.videoSlider.sliderMoved.connect(self.change_video_slider)
        self.exportData.triggered.connect(self.export)
        self.exportData.setEnabled(True)
        self.saveProjectButton.setEnabled(True)

    def update_identity(self):
        for detections in self.track_results[self.cap.frame_no]:
            if detections[4] not in self.mark_map:
                self.identity = detections[4]
                self.identityLabel.setText(f"Vehicle #{detections[4]}")
                self.widget.setHidden(False)
                return
        self.widget.setHidden(True)
        if len(self.identities) == len(self.mark_map):
            self.identityLabel.setText("所有车辆均已标记")
        else:
            self.identityLabel.setText("该帧的所有车辆已标记完毕")

    def refresh_status_bar(self):
        self.statusBar().showMessage(f"{self.cap.w}x{self.cap.h}, {self.cap.fps} fps, "
                                     f"{self.cap.frame_no + 1}/{self.cap.frame_count} frames")

    def change_video_slider(self):
        self.cap.playing = False
        self.cap.frame_no = self.videoSlider.value()

    def change_frame(self, value):
        self.cap.playing = False
        self.cap.frame_no += value
        if self.cap.frame_no < 0:
            self.cap.frame_no = 0
        elif self.cap.frame_no == self.cap.frame_count:
            self.cap.frame_no = self.cap.frame_count - 1
        # self.videoSlider.setValue(self.cap.frame_no)

    def jump_to_next_unmarked_vehicle(self):
        for i in range(self.cap.frame_no, self.cap.frame_count):
            for detections in self.track_results[i]:
                if detections[4] not in self.mark_map:
                    self.change_frame(i - self.cap.frame_no)
                    return

    def play(self):
        self.cap.playing = not self.cap.playing

    def mark(self):
        self.mark_map[self.identity] = (self.typeComboBox.currentIndex(), 13 + self.colorComboBox.currentIndex())
        self.update_identity()

    def ignore(self):
        self.mark_map[self.identity] = (-1, -1)
        self.update_identity()

    def export(self):
        path, file_type = QFileDialog.getSaveFileName(self, 'Save as...', os.getcwd(), ".txt", ".txt")
        if path == '':
            return False
        w, h = self.cap.w, self.cap.h
        fw = open(path + '.txt' if path[-4:] != '.txt' else '', 'w')
        for frame in self.track_results:
            detections = self.cap.get_boxes(frame)
            for detection in detections:
                if detection[4] in self.mark_map:
                    tup = self.mark_map[detection[4]]
                    if tup[0] == -1:
                        continue
                    fw.write(f'{frame} {tup[0]} {tup[1]} {(detection[0] + detection[2]) / 2 / w} '
                             f'{(detection[1] + detection[3]) / 2 / w} '
                             f'{(detection[2] - detection[0]) / w} '
                             f'{(detection[3] - detection[1]) / w}\n')
        fw.close()


class VideoThread(threading.Thread):
    def __init__(self, cap, window: MainWindow):
        super(VideoThread, self).__init__()
        self.last_refresh = 0
        self.playing = False
        self.frame = None
        self.frame_no = 0
        # detect if changed
        self.prev_frame_no = -1
        self.cap = cap
        self.window = window
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.setDaemon(True)

    def run(self):
        while self.cap.isOpened():
            if self.frame_no == self.frame_count:
                continue
            self.refresh_video_capture()
            if not self.playing:
                continue
            self.frame_no += 1

    def refresh_video_capture(self):
        if self.prev_frame_no != self.frame_no:
            self.window.videoSlider.setValue(self.frame_no)
            self.prev_frame_no = self.frame_no
            self.window.update_identity()
        self.cap.set(1, self.frame_no)
        _, self.frame = self.cap.read()
        self.draw_box()
        scale = 1000 / self.w
        frame = cv2.resize(self.frame, (int(self.w * scale), int(self.h * scale)))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = QImage(frame.data, self.w * scale, self.h * scale, QImage.Format_RGB888)
        wait_time = max(self.last_refresh + 1.0 / self.fps - time.time(), 0)
        cv2.waitKey(int(wait_time * 1000))
        self.last_refresh = time.time()
        self.window.videoLabel.setPixmap(QPixmap.fromImage(img))
        self.window.refresh_status_bar()

    def draw_box(self):
        boxes = self.get_boxes(self.frame_no)
        for box in boxes:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            id = box[4]
            label = '{}{:d}'.format("", id)
            color = (0, 0, 255) if id not in self.window.mark_map else (0, 255, 0)
            # print(color)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(self.frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(self.frame, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    def get_boxes(self, frame_no):
        # TODO: Fix that if box has been erased, program doesn't consider all boxes has been marked
        boxes = self.window.track_results[frame_no]
        if SettingManager.config['replace_track_box']:
            detections = self.window.detections[frame_no]
            for box in boxes:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                for detection in detections:
                    if abs(cx - detection[0]) < self.w / 50 and abs(cy - detection[1]) < self.h / 50:
                        box[0] = int(detection[0] - detection[2] / 2)
                        box[1] = int(detection[1] - detection[3] / 2)
                        box[2] = int(detection[0] + detection[2] / 2)
                        box[3] = int(detection[1] + detection[3] / 2)
                        break
        if SettingManager.config['filter_box_on_the_edge']:
            for i in range(len(boxes) - 1, -1, -1):
                box = boxes[i]
                if box[0] < self.w / 50 or box[1] < self.h / 50 or box[2] > self.w * 0.98 or box[3] > self.h * 0.98:
                    del boxes[i]
        return boxes
