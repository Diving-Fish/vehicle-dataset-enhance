import threading
import time
from collections import defaultdict

from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QAction, QApplication, QShortcut
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import *

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
        self.names = ['Sedan',
                      'Hatchback',
                      'Suv',
                      'Taxi',
                      'Bus',
                      'Van',
                      'MiniVan',
                      'Truck-Box-Large',
                      'Truck-Box-Med',
                      'Truck-Util',
                      'Truck-Flatbed',
                      'Truck-Pickup',
                      'Police',
                      'Red',
                      'Orange',
                      'Yellow',
                      'Green',
                      'Blue',
                      'Beige',
                      'Brown',
                      'White',
                      'Silver',
                      'Gray',
                      'Black',
                      'Multi']
        self.setupUi(self)
        self.mark_map = None
        self.cap = None
        self.createProjectButton.triggered.connect(self.new_project)
        self.loadProjectButton.triggered.connect(self.load_project)
        self.settingButton.triggered.connect(self.setting)
        self.saveProjectButton.triggered.connect(self.save)
        self.playButton.clicked.connect(self.play)
        self.prevFrameButton.clicked.connect(lambda: self.change_frame(-1))
        self.nextFrameButton.clicked.connect(lambda: self.change_frame(1))
        self.jumpFrameButton.clicked.connect(self.jump_to_next_unmarked_vehicle)
        self.identityComboBox.currentIndexChanged.connect(lambda: self.change_identity(self.identityComboBox.currentIndex()))
        self.markButton.clicked.connect(self.mark)
        self.ignoreButton.clicked.connect(self.ignore)
        self.videoSlider.sliderMoved.connect(self.change_video_slider)
        self.exportData.triggered.connect(lambda: self.export(False))
        self.exportDataWithoutUnmarked.triggered.connect(lambda: self.export(True))
        self.exportData.setEnabled(False)
        self.exportDataWithoutUnmarked.setEnabled(False)
        self.saveProjectButton.setEnabled(False)
        self.groupBox.setHidden(True)
        self.groupBox_2.setHidden(True)
        self.show()
        QShortcut(QKeySequence.Undo, self, self.undo)
        QShortcut(QKeySequence.Save, self, self.save)
        QShortcut(Qt.Key_Space, self, self.jump_to_next_unmarked_vehicle)

    def wheelEvent(self, event) -> None:
        angle = event.angleDelta().y()
        self.change_frame(int(-angle / 60))

    def save(self):
        if not self.mark_map:
            return
        with open(self.path, 'w') as fd3:
            for key in self.mark_map:
                fd3.write(f'{key} {self.mark_map[key][0]} {self.mark_map[key][1]}\n')
        fd3.close()

    def undo(self):
        if self.mark_map:
            self.mark_map.popitem()

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
            self.path = path
            cap = cv2.VideoCapture(project_dir + '/video.mkv')
            fd = open(project_dir + '/track_results.txt', 'r')
            fd2 = open(project_dir + '/detection_results.txt', 'r')
            detections = defaultdict(list)
            # Load detections
            for line in fd2:
                arr = line.strip().split(' ')
                detections[int(arr[0])].append([float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4])])
            fd2.close()
            # Load tracks
            track_results = defaultdict(list)
            identities = set([])
            for line in fd:
                arr = line.strip().split(' ')
                track_results[int(arr[0])].append([int(arr[2]), int(arr[3]), int(arr[4]), int(arr[5]), int(arr[1])])
                identities.add(int(arr[1]))
            fd.close()
            # Load progress
            with open(path, 'r') as fd3:
                for line in fd3:
                    arr = line.strip().split(' ')
                    if len(arr) == 0:
                        continue
                    self.mark_map[int(arr[0])] = (int(arr[1]), int(arr[2]))
                fd3.close()
        except Exception as e:
            print(e)
            self.dialog = Dialog("请选择正确的工程文件")
            return
        self.init_video(cap, track_results, detections, identities)

    def init_video(self, cap, track_results, detections, identities):
        if self.cap:
            self.cap.cap.release()
        # Show Group Box
        self.groupBox.setHidden(False)
        self.groupBox_2.setHidden(False)

        self.identity = 0
        self.current_identities = []
        self.identities = identities
        self.track_results = track_results
        self.detections = detections
        self.cap = VideoThread(cap, self)
        self.cap.start()
        # Init Slider
        self.videoSlider.setMinimum(0)
        self.videoSlider.setMaximum(self.cap.frame_count - 1)
        # Init button functions
        self.exportData.setEnabled(True)
        self.exportDataWithoutUnmarked.setEnabled(True)
        self.saveProjectButton.setEnabled(True)

    def update_identity(self, index=-1):
        flag = False

        self.current_identities = []
        for detections in self.track_results[self.cap.frame_no]:
            if detections[4] not in self.mark_map:
                flag = True
                self.current_identities.append(detections[4])

        if flag:
            self.identityComboBox.clear()
            self.widget.setHidden(False)
            self.identityLabel.setText("该帧仍有未标记的车辆")
            # if self.identity in self.current_identities:
            #     self.identity = self.current_identities[self.current_identities.index(self.identity) + 1]
            # else:
            if index + 1 < len(self.current_identities):
                self.identity = self.current_identities[index+1]
                self.identityComboBox.setCurrentIndex(index+1)
            else:
                if self.identity in self.current_identities:
                    self.identityComboBox.setCurrentIndex(self.current_identities.index(self.identity))
                else:
                    self.identity = self.current_identities[0]
                    self.identityComboBox.setCurrentIndex(0)
            for elem in self.current_identities:
                self.identityComboBox.addItem(str(elem))
            return
        self.widget.setHidden(True)
        if len(self.identities) == len(self.mark_map):
            self.identityLabel.setText("所有车辆均已标记")
        elif not flag:
            self.identityLabel.setText("该帧的所有车辆已标记完毕")

    def refresh_status_bar(self):
        self.statusBar().showMessage(f"{self.cap.w}x{self.cap.h}, {self.cap.fps} fps, "
                                     f"{self.cap.frame_no + 1}/{self.cap.frame_count} frames")

    def change_video_slider(self):
        self.cap.playing = False
        self.cap.frame_no = self.videoSlider.value()

    def change_identity(self, index):
        self.identity = self.current_identities[index]

    def change_frame(self, value):
        if not self.cap:
            return
        self.cap.playing = False
        self.cap.frame_no += value
        if self.cap.frame_no < 0:
            self.cap.frame_no = 0
        elif self.cap.frame_no == self.cap.frame_count:
            self.cap.frame_no = self.cap.frame_count - 1
        # self.videoSlider.setValue(self.cap.frame_no)

    def jump_to_next_unmarked_vehicle(self):
        if not self.cap:
            return
        for i in range(self.cap.frame_no, self.cap.frame_count):
            for detections in self.track_results[i]:
                if detections[4] not in self.mark_map:
                    self.change_frame(i - self.cap.frame_no)
                    return

    def play(self):
        self.cap.playing = not self.cap.playing

    def mark(self):
        self.mark_map[self.identity] = (self.typeComboBox.currentIndex(), 13 + self.colorComboBox.currentIndex())
        self.update_identity(self.current_identities.index(self.identity))

    def ignore(self):
        self.mark_map[self.identity] = (-1, -1)
        self.update_identity(self.current_identities.index(self.identity))

    def export(self, except_unmarked):
        path, file_type = QFileDialog.getSaveFileName(self, 'Save as...', os.getcwd(), ".txt", ".txt")
        if path == '':
            return False
        w, h = self.cap.w, self.cap.h
        fw = open(path + '.txt' if path[-4:] != '.txt' else '', 'w')
        for frame in self.track_results:
            detections = self.cap.get_boxes(frame)
            buff = ''
            for detection in detections:
                if detection[4] in self.mark_map:
                    tup = self.mark_map[detection[4]]
                    if tup[0] == -1:
                        continue
                    buff += (f'{frame} {tup[0]} {tup[1]} {(detection[0] + detection[2]) / 2 / w} '
                             f'{(detection[1] + detection[3]) / 2 / w} '
                             f'{(detection[2] - detection[0]) / w} '
                             f'{(detection[3] - detection[1]) / w}\n')
                elif except_unmarked:
                    buff = ''
                    break
            fw.write(buff)
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
            label = f"{id}"
            if id in self.window.mark_map:
                color = (0, 255, 0)
                t = self.window.mark_map[id]
                if t[0] != -1:
                    label += f" {self.window.names[t[0]]}({self.window.names[t[1]]})"
            elif self.window.identity == id:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
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
                    if abs(cx - detection[0]) < self.w / 10 and abs(cy - detection[1]) < self.h / 50:
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
