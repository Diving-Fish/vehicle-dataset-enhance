from PyQt5.QtWidgets import QDialog

from lib.setting_dialog import Ui_Dialog
from lib.setting_manager import SettingManager


class SettingDialog(QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()
        self.pushButton.clicked.connect(self.confirm)
        self.checkBox.setCheckState(2 if SettingManager.config['filter_box_on_the_edge'] else 0)
        self.checkBox_2.setCheckState(2 if SettingManager.config['replace_track_box'] else 0)

    def confirm(self):
        # store file to settings.json
        SettingManager.modify_config({
            "replace_track_box": self.checkBox_2.checkState() == 2,
            "filter_box_on_the_edge": self.checkBox.checkState() == 2,
        })
        self.close()
