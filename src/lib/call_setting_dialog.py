from PyQt5.QtWidgets import QDialog

from lib.setting_dialog import Ui_Dialog


class SettingDialog(QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()
        self.pushButton.clicked.connect(self.confirm)

    def confirm(self):
        # store file to settings.json
        self.close()
