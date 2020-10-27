# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Github\dataset-enhance\src\ui\main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(955, 609)
        MainWindow.setToolTip("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.videoLabel = QtWidgets.QLabel(self.groupBox_2)
        self.videoLabel.setText("")
        self.videoLabel.setObjectName("videoLabel")
        self.verticalLayout.addWidget(self.videoLabel)
        self.videoSlider = QtWidgets.QSlider(self.groupBox_2)
        self.videoSlider.setOrientation(QtCore.Qt.Horizontal)
        self.videoSlider.setObjectName("videoSlider")
        self.verticalLayout.addWidget(self.videoSlider)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.playButton = QtWidgets.QPushButton(self.groupBox_2)
        self.playButton.setObjectName("playButton")
        self.horizontalLayout.addWidget(self.playButton)
        self.prevFrameButton = QtWidgets.QPushButton(self.groupBox_2)
        self.prevFrameButton.setObjectName("prevFrameButton")
        self.horizontalLayout.addWidget(self.prevFrameButton)
        self.nextFrameButton = QtWidgets.QPushButton(self.groupBox_2)
        self.nextFrameButton.setObjectName("nextFrameButton")
        self.horizontalLayout.addWidget(self.nextFrameButton)
        self.jumpFrameButton = QtWidgets.QPushButton(self.groupBox_2)
        self.jumpFrameButton.setObjectName("jumpFrameButton")
        self.horizontalLayout.addWidget(self.jumpFrameButton)
        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 2)
        self.horizontalLayout.setStretch(2, 2)
        self.horizontalLayout.setStretch(3, 3)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.setStretch(0, 1)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox_2, 0, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMinimumSize(QtCore.QSize(250, 0))
        self.groupBox.setMaximumSize(QtCore.QSize(250, 16777215))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.identityLabel = QtWidgets.QLabel(self.groupBox)
        self.identityLabel.setGeometry(QtCore.QRect(20, 30, 221, 21))
        self.identityLabel.setObjectName("identityLabel")
        self.widget = QtWidgets.QWidget(self.groupBox)
        self.widget.setGeometry(QtCore.QRect(20, 60, 211, 151))
        self.widget.setObjectName("widget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.typeComboBox = QtWidgets.QComboBox(self.widget)
        self.typeComboBox.setMinimumSize(QtCore.QSize(125, 0))
        self.typeComboBox.setObjectName("typeComboBox")
        self.typeComboBox.addItem("")
        self.typeComboBox.addItem("")
        self.typeComboBox.addItem("")
        self.typeComboBox.addItem("")
        self.typeComboBox.addItem("")
        self.typeComboBox.addItem("")
        self.typeComboBox.addItem("")
        self.typeComboBox.addItem("")
        self.typeComboBox.addItem("")
        self.typeComboBox.addItem("")
        self.typeComboBox.addItem("")
        self.typeComboBox.addItem("")
        self.typeComboBox.addItem("")
        self.horizontalLayout_3.addWidget(self.typeComboBox)
        self.gridLayout_4.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        self.markButton = QtWidgets.QPushButton(self.widget)
        self.markButton.setObjectName("markButton")
        self.gridLayout_4.addWidget(self.markButton, 2, 0, 1, 1)
        self.ignoreButton = QtWidgets.QPushButton(self.widget)
        self.ignoreButton.setObjectName("ignoreButton")
        self.gridLayout_4.addWidget(self.ignoreButton, 3, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.colorComboBox = QtWidgets.QComboBox(self.widget)
        self.colorComboBox.setMinimumSize(QtCore.QSize(125, 0))
        self.colorComboBox.setObjectName("colorComboBox")
        self.colorComboBox.addItem("")
        self.colorComboBox.addItem("")
        self.colorComboBox.addItem("")
        self.colorComboBox.addItem("")
        self.colorComboBox.addItem("")
        self.colorComboBox.addItem("")
        self.colorComboBox.addItem("")
        self.colorComboBox.addItem("")
        self.colorComboBox.addItem("")
        self.colorComboBox.addItem("")
        self.colorComboBox.addItem("")
        self.colorComboBox.addItem("")
        self.horizontalLayout_2.addWidget(self.colorComboBox)
        self.gridLayout_4.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox, 0, 2, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 955, 23))
        self.menuBar.setObjectName("menuBar")
        self.fileMenu = QtWidgets.QMenu(self.menuBar)
        self.fileMenu.setObjectName("fileMenu")
        self.settingMenu = QtWidgets.QMenu(self.menuBar)
        self.settingMenu.setObjectName("settingMenu")
        MainWindow.setMenuBar(self.menuBar)
        self.createProjectButton = QtWidgets.QAction(MainWindow)
        self.createProjectButton.setObjectName("createProjectButton")
        self.loadProjectButton = QtWidgets.QAction(MainWindow)
        self.loadProjectButton.setObjectName("loadProjectButton")
        self.exportData = QtWidgets.QAction(MainWindow)
        self.exportData.setObjectName("exportData")
        self.saveProjectButton = QtWidgets.QAction(MainWindow)
        self.saveProjectButton.setObjectName("saveProjectButton")
        self.settingButton = QtWidgets.QAction(MainWindow)
        self.settingButton.setObjectName("settingButton")
        self.fileMenu.addAction(self.createProjectButton)
        self.fileMenu.addAction(self.loadProjectButton)
        self.fileMenu.addAction(self.saveProjectButton)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exportData)
        self.settingMenu.addAction(self.settingButton)
        self.menuBar.addAction(self.fileMenu.menuAction())
        self.menuBar.addAction(self.settingMenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Vehicle Detection Dataset Enhance"))
        self.groupBox_2.setTitle(_translate("MainWindow", "视频流"))
        self.playButton.setText(_translate("MainWindow", "播放/暂停"))
        self.prevFrameButton.setText(_translate("MainWindow", "上一帧"))
        self.nextFrameButton.setText(_translate("MainWindow", "下一帧"))
        self.jumpFrameButton.setText(_translate("MainWindow", "跳转到下一个未标记的车辆所在帧"))
        self.groupBox.setTitle(_translate("MainWindow", "检测"))
        self.identityLabel.setText(_translate("MainWindow", "Vehicle #1"))
        self.label.setText(_translate("MainWindow", "种类"))
        self.typeComboBox.setItemText(0, _translate("MainWindow", "Sedan"))
        self.typeComboBox.setItemText(1, _translate("MainWindow", "Hatchback"))
        self.typeComboBox.setItemText(2, _translate("MainWindow", "Suv"))
        self.typeComboBox.setItemText(3, _translate("MainWindow", "Taxi"))
        self.typeComboBox.setItemText(4, _translate("MainWindow", "Bus"))
        self.typeComboBox.setItemText(5, _translate("MainWindow", "Van"))
        self.typeComboBox.setItemText(6, _translate("MainWindow", "MiniVan"))
        self.typeComboBox.setItemText(7, _translate("MainWindow", "Truck-Box-Large"))
        self.typeComboBox.setItemText(8, _translate("MainWindow", "Truck-Box-Med"))
        self.typeComboBox.setItemText(9, _translate("MainWindow", "Truck-Util"))
        self.typeComboBox.setItemText(10, _translate("MainWindow", "Truck-Flatbed"))
        self.typeComboBox.setItemText(11, _translate("MainWindow", "Truck-Pickup"))
        self.typeComboBox.setItemText(12, _translate("MainWindow", "Police"))
        self.markButton.setText(_translate("MainWindow", "标记"))
        self.ignoreButton.setText(_translate("MainWindow", "忽略"))
        self.label_2.setText(_translate("MainWindow", "颜色"))
        self.colorComboBox.setItemText(0, _translate("MainWindow", "Red"))
        self.colorComboBox.setItemText(1, _translate("MainWindow", "Orange"))
        self.colorComboBox.setItemText(2, _translate("MainWindow", "Yellow"))
        self.colorComboBox.setItemText(3, _translate("MainWindow", "Green"))
        self.colorComboBox.setItemText(4, _translate("MainWindow", "Blue"))
        self.colorComboBox.setItemText(5, _translate("MainWindow", "Beige"))
        self.colorComboBox.setItemText(6, _translate("MainWindow", "Brown"))
        self.colorComboBox.setItemText(7, _translate("MainWindow", "White"))
        self.colorComboBox.setItemText(8, _translate("MainWindow", "Silver"))
        self.colorComboBox.setItemText(9, _translate("MainWindow", "Gray"))
        self.colorComboBox.setItemText(10, _translate("MainWindow", "Black"))
        self.colorComboBox.setItemText(11, _translate("MainWindow", "Multi"))
        self.fileMenu.setTitle(_translate("MainWindow", "文件"))
        self.settingMenu.setTitle(_translate("MainWindow", "设置"))
        self.createProjectButton.setText(_translate("MainWindow", "从视频创建新项目"))
        self.loadProjectButton.setText(_translate("MainWindow", "打开现有项目"))
        self.exportData.setText(_translate("MainWindow", "导出数据"))
        self.saveProjectButton.setText(_translate("MainWindow", "保存当前项目"))
        self.settingButton.setText(_translate("MainWindow", "设置"))
