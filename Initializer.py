from PyQt5.QtWidgets import QColorDialog, QFileDialog
import os
import cv2

class Initializer:
    def initialize(self, Home):
        self.initWidget()
        self.eventBond(Home)

    def initWidget(self):
        """初始未上传图片时下载按钮不可用"""
        self.downloadButton.setDisabled(True)

    def eventBond(self, Home):
        """关联按钮点击事件"""
        self.uploadButton.clicked.connect(lambda: self.uploadImg(Home))
        self.downloadButton.clicked.connect(lambda: self.downloadImg(Home))

    def uploadImg(self, Home):
        """上传图片到画板"""
        fpath, _ = QFileDialog.getOpenFileName(
            Home, "选择图片",
            os.path.join(os.path.expanduser('~'), "Desktop"),
            "所有文件(*.jpg *.png *.jpeg);;(*.jpg);;(*.png);;(*.jepg)"
        )
        if fpath == '':
            return

        self.img_bgr = self.colorizer.colorizeImg(fpath)
        self.imgViewer.newItem(self.img_bgr.copy())

        # 第一次上传图片后将画笔等按钮激活
        if self.downloadButton.isEnabled() == False:
            self.downloadButton.setEnabled(True)

    def downloadImg(self, Home):
        """将AI上色后的图片下载到本地"""
        fpath, _ = QFileDialog.getSaveFileName(
            Home, "选择保存目录",
            os.path.join(os.path.expanduser('~'), "Desktop"),
            "PNG(*.png);;JPG(*.jpg);;JEPG(*.jepg)"
        )
        if fpath != '':
            pass

        cv2.imwrite(fpath, self.img_bgr)
