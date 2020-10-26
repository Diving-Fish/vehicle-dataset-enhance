from time import sleep

from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QDialog

from lib.progress_dialog import Ui_CreateProgressDialog
from create_project import ProjectCreateThread


class ProgressDialog(QDialog, Ui_CreateProgressDialog):
    def __init__(self, opt, parent):
        super().__init__()
        self.parent = parent
        self.setupUi(self)
        self.opt = opt
        self.t = ProjectCreateThread(self)
        self.show()
        self.t.start()
        self.t.join()
        self.close()
        self.parent.finish_create()
