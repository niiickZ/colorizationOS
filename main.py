import sys
from ui import Ui_Home
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPixmap, QPainter
from Initializer import Initializer
from model.Colorizer import Colorizer
import os

class MyWindow(QWidget, Ui_Home, Initializer):
    def paintEvent(self, event):
        super().paintEvent(event)
        qp = QPainter(self)
        qp.drawPixmap(0, 0, self.width(), self.height(), self.backgroundImg)

    def __init__(self):
        super().__init__()

        self.backgroundImg = QPixmap('ui/background.png')
        self.setupUi(self)
        self.initialize(self)

        self.colorizer = Colorizer(os.path.abspath('model\\weights'))
        self.img_bgr = None

def main():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()