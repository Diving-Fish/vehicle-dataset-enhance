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
        self.t = ProjectCreateThread(self.opt, self.progressBar, self.pushButton)
        self.show()
        self.t.start()
        self.pushButton.clicked.connect(self.finish_create)
        self.pushButton.setEnabled(False)

    def finish_create(self):
        self.close()
        self.parent.finish_create()
