import sys
from PyQt5.QtWidgets import QApplication, QWidget
from lib.callPrevWindow import PrevWindow


def main():
    app = QApplication(sys.argv)

    pw = PrevWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
