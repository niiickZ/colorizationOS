from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF, QLineF
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsView
import cv2

class ImageScene(QGraphicsScene):
    wheelSignal = pyqtSignal(float, QPointF)
    dragSignal = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.prePos = QPointF(0, 0)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.prePos = event.scenePos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            mouseMove = event.scenePos() - self.prePos  # 鼠标当前位置-先前位置=单次偏移量
            self.prePos = event.scenePos()
            self.dragSignal.emit(mouseMove)

    def wheelEvent(self, event):
        angle = event.delta() / 8
        self.wheelSignal.emit(angle, event.scenePos())

class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        # self.setStyleSheet("padding: 0px; border: 0px;")  # 内边距和边界去除
        self.setStyleSheet("background-color: pink")

        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # 改变对齐方式

        self.scene = ImageScene(self)
        self.setSceneRect(0, 0, self.width(), self.height())  # 设置图形场景大小和图形视图大小一致
        self.scene.wheelSignal.connect(self.zoom)
        self.scene.dragSignal.connect(self.drag)
        self.setScene(self.scene)

        self.ratio = 1.0  # 缩放初始比例
        self.zoom_step = 0.05  # 缩放步长
        self.zoom_max = 2  # 缩放最大值
        self.zoom_min = 0.05  # 缩放最小值
        self.pixmapItem = None

    def newItem(self, img):
        """添加新图元"""
        self.scene.clear()  # 清除当前图元

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Qimg = QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        self.pixmap = QPixmap(Qimg)

        if self.pixmap.width() > self.width() - 20 or self.pixmap.height() > self.height() - 20:
            self.ratio = min((self.width() - 20) / self.pixmap.width(),
                             (self.height() - 20) / self.pixmap.height())

        self.pixmapItem = self.scene.addPixmap(self.pixmap)
        self.pixmapItem.setScale(self.ratio)  # 缩放

        w = self.pixmap.width() * self.ratio
        h = self.pixmap.height() * self.ratio

        originX = (self.width() - w) / 2
        originY = (self.height() - h) / 2

        self.pixmapItem.setPos(originX, originY)

    def getItemPos(self):
        """获取图元左上角和右下角坐标"""
        w = self.pixmap.size().width() * self.ratio
        h = self.pixmap.size().height() * self.ratio

        # 左上角
        x1 = self.pixmapItem.scenePos().x()
        y1 = self.pixmapItem.scenePos().y()

        # 右下角
        x2 = x1 + w
        y2 = y1 + h

        return x1, y1, x2, y2

    def drag(self, mouseMove):
        """拖动图元"""
        self.pixmapItem.setPos(self.pixmapItem.scenePos() + mouseMove)

    def zoom(self, angle, scenePos):
        """鼠标滚轮缩放图片，当光标在图元之外，以图元中心为缩放原点；当光标在图元之中，以光标位置为缩放中心"""
        x1, y1, x2, y2 = self.getItemPos()

        fac = 1 if angle > 0 else -1
        self.ratio = self.ratio + fac * self.zoom_step

        if angle > 0 and self.ratio > self.zoom_max:
            self.ratio = self.zoom_max
        elif angle < 0 and self.ratio < self.zoom_min:
            self.ratio = self.zoom_min
        else:
            if x1 < scenePos.x() < x2 and y1 < scenePos.y() < y2:  # 判断鼠标悬停位置是否在图元中
                # print('在内部')
                self.pixmapItem.setScale(self.ratio)  # 缩放
                a1 = scenePos - self.pixmapItem.scenePos()  # 鼠标与图元左上角的差值
                a2 = self.ratio / (self.ratio - fac * self.zoom_step) - 1  # 对应比例
                delta = a1 * a2
                self.pixmapItem.setPos(self.pixmapItem.scenePos() - delta)
            else:
                # print('在外部')  # 以图元中心缩放
                self.pixmapItem.setScale(self.ratio)  # 缩放
                delta_x = (self.pixmap.size().width() * self.zoom_step) / 2  # 图元偏移量
                delta_y = (self.pixmap.size().height() * self.zoom_step) / 2
                self.pixmapItem.setPos(self.pixmapItem.scenePos().x() - fac * delta_x,
                                       self.pixmapItem.scenePos().y() - fac * delta_y)  # 图元偏移
