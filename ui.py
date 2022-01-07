# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from ImageViewer import ImageViewer

class Ui_Home(object):
    def setupUi(self, Home):
        Home.setObjectName("Home")
        Home.resize(900, 630)
        Home.setMinimumSize(QtCore.QSize(900, 630))
        Home.setStyleSheet("#Home{\n"
"    /*background-image: url(ui/background.png);*/\n"
"}\n"
"\n"
"#imgViewer{\n"
"    border: 2px solid transparent;\n"
"    border-radius: 8px;\n"
"    background-color: rgba(60, 63, 65, 75%);\n"
"}\n"
"\n"
"\n"
".QPushButton{\n"
"    border: 2px solid #5d6165;\n"
"    border-radius: 5px;\n"
"    background-color: transparent;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
".QPushButton:hover\n"
"{\n"
"    color: white;\n"
"    border-width: 1px;\n"
"}\n"
"\n"
".QPushButton:pressed\n"
"{ \n"
"    border-style: inset; \n"
"    border-width: 3px;\n"
"}")
        self.gridLayout = QtWidgets.QGridLayout(Home)
        self.gridLayout.setObjectName("gridLayout")
        self.imgViewer = ImageViewer(Home)
        self.imgViewer.setMinimumSize(QtCore.QSize(820, 500))
        self.imgViewer.setStyleSheet("")
        self.imgViewer.setObjectName("imgViewer")
        self.gridLayout.addWidget(self.imgViewer, 0, 0, 1, 5)
        spacerItem = QtWidgets.QSpacerItem(200, 20, QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        self.uploadButton = QtWidgets.QPushButton(Home)
        self.uploadButton.setMinimumSize(QtCore.QSize(120, 50))
        self.uploadButton.setMaximumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.uploadButton.setFont(font)
        self.uploadButton.setObjectName("uploadButton")
        self.gridLayout.addWidget(self.uploadButton, 1, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(200, 20, QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 2, 1, 1)
        self.downloadButton = QtWidgets.QPushButton(Home)
        self.downloadButton.setMinimumSize(QtCore.QSize(120, 50))
        self.downloadButton.setMaximumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.downloadButton.setFont(font)
        self.downloadButton.setObjectName("downloadButton")
        self.gridLayout.addWidget(self.downloadButton, 1, 3, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(200, 20, QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 1, 4, 1, 1)

        self.retranslateUi(Home)
        QtCore.QMetaObject.connectSlotsByName(Home)

    def retranslateUi(self, Home):
        _translate = QtCore.QCoreApplication.translate
        Home.setWindowTitle(_translate("Home", "Form"))
        self.uploadButton.setText(_translate("Home", "上 传"))
        self.downloadButton.setText(_translate("Home", "下 载"))
