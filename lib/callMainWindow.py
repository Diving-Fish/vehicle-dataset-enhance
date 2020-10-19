import threading
import time
from collections import defaultdict

from PyQt5.QtWidgets import QWidget, QFileDialog, QDialog, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from lib.mainWindow import Ui_MainWindow
import os
from cv2 import cv2


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, file_name, cap, frame_detections, identities):
        super().__init__()
        self.file_name = file_name
        self.identity = 0
        self.identities = identities
        # Init mark dict
        self.mark_map = {}
        # Init video
        self.frame_detections = frame_detections
        self.setupUi(self)
        self.cap = VideoThread(cap, self)
        self.cap.start()
        self.show()
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
        self.exportButton.clicked.connect(self.export)

    def __del__(self):
        print('Saving unsaved data...')
        if len(self.mark_map) != 0:
            if 'data' not in os.listdir(os.getcwd()):
                os.mkdir('./data', 666)
            with open(f'./data/{self.file_name}_cpt.txt', 'w') as fw:
                for key in self.mark_map:
                    fw.write(f'{key} {self.mark_map[key]}\n')

    def update_identity(self):
        for detections in self.frame_detections[self.cap.frame_no]:
            if detections[4] not in self.mark_map:
                self.identity = detections[4]
                self.identityLabel.setText(f"Vehicle #{detections[4]}")
                self.widget.setHidden(False)
                return
        self.widget.setHidden(True)
        if len(self.identities) == len(self.mark_map):
            self.identityLabel.setText("All vehicles have been marked.")
        else:
            self.identityLabel.setText("No unmarked vehicles in this frame.")

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
            for detections in self.frame_detections[i]:
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
        for frame in self.frame_detections:
            detections = self.frame_detections[frame]
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
        boxes = self.window.frame_detections[self.frame_no]
        for box in boxes:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            id = box[4]
            label = '{}{:d}'.format("", id)
            color = (0, 0, 255) if id not in self.window.mark_map else (0, 255, 0)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(self.frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(self.frame, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
