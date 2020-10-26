from PyQt5.QtWidgets import QDialog
from lib.dialog import Ui_Dialog


class Dialog(QDialog, Ui_Dialog):
    def __init__(self, message):
        super().__init__()
        self.setupUi(self)
        self.label.setText(message)
        self.show()
