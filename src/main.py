import sys
from PyQt5.QtWidgets import QApplication
from lib.call_main_window import MainWindow
from lib.setting_manager import SettingManager


def main():
    app = QApplication(sys.argv)

    SettingManager.load_config()

    main_window = MainWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
