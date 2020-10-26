from collections import defaultdict

from PyQt5.QtWidgets import QWidget, QFileDialog
from lib.prev_window import Ui_Form
from lib.call_dialog import Dialog
from lib.call_main_window import MainWindow
import os
from cv2 import cv2


class PrevWindow(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.chooseVideoFileButton.clicked.connect(lambda: self.choose_file(self.videoFilePath))
        self.chooseDetectFileButton.clicked.connect(lambda: self.choose_file(self.detectFilePath))
        self.Launch.clicked.connect(self.launch)
        self.show()

    def choose_file(self, target):
        path, file_type = QFileDialog.getOpenFileName(self, 'Choose a File', os.getcwd(), "All Files(*)")
        target.setText(path)

    def launch(self):
        video_path = self.videoFilePath.text()
        detect_path = self.detectFilePath.text()
        try:
            cap = cv2.VideoCapture(video_path)
            fd = open(detect_path, 'r')
            frame_detections = defaultdict(list)
            identities = set([])
            for line in fd:
                arr = line.strip().split(' ')
                frame_detections[int(arr[0])].append([int(arr[2]), int(arr[3]), int(arr[4]), int(arr[5]), int(arr[1])])
                identities.add(int(arr[1]))
        except Exception:
            self.dialog = Dialog("Please choose right file")
            return
        self.main = MainWindow(video_path.split('\\')[-1].split('/')[-1], cap, frame_detections, identities)
        self.close()
