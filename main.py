import numpy as np
from mss import mss
from PIL import Image
from PyQt5.QtWidgets import QApplication
import sys
from Window import Window


def grabScreenshot():
    viewport = {'top': 100, 'left': 100, 'width': 200, 'height': 200}
    recorder = mss()
    sourceImg = recorder.grab(viewport)
    img = Image.frombytes("RGB", sourceImg.size, sourceImg.rgb, "raw", "BGR")
    return img

def main():

    app = QApplication(sys.argv)
    window = Window("Mario AI Software", 800, 600)
    window.create()
    window.setUpdateFunc(lambda: window.setCaptureFrame(grabScreenshot()))
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()