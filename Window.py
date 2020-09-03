from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

class Window:
    def __init__(self, title, width, height, monitorDim = None):
        self.width = width
        self.height = height
        self.title = title
        self.windowHandle = QWidget()
        self.windowHandle.resize(self.width, self.height)
        self.windowHandle.setWindowTitle(self.title)

        self.currentCapture = None
        self.videoFeed = QLabel(self.windowHandle)

        self.globalTimer = QTimer(self.windowHandle)

        if monitorDim is not None:
            self.windowHandle.move(monitorDim[0] / 2 - self.width / 2, monitorDim[1] / 2 - self.height / 2)

    def create(self):
        self.globalTimer.start(1000 // 60)
        self.videoFeed.show()
        self.windowHandle.show()

    def setCaptureFrame(self, frame):
        self.currentCapture = QPixmap.fromImage(QImage(frame.tobytes("raw", "BGR"), frame.size[0], frame.size[1], QImage.Format_RGB888))
        self.videoFeed.resize(frame.size[0], frame.size[1])
        self.videoFeed.move(self.width / 2 - frame.size[0] / 2, self.height / 2 - frame.size[1] / 2)
        self.videoFeed.setPixmap(self.currentCapture)

    def setUpdateFunc(self, func):
        self.globalTimer.timeout.connect(func)