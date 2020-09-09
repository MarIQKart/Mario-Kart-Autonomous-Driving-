from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

from mss import mss
from PIL import Image

class Window:

    '''
        :param title:
            The title of the window that will be shown in the
            top of the screen.

        :param width:
            The width of the window in pixels.

        :param height:
            The height of the window in pixels.
    '''
    def __init__(self, title, width, height):
        self.width = width # Width of the window in pixels
        self.height = height # Height pf the window in pixels
        self.title = title # The title of the window
        self.windowHandle = QWidget() # A handle to our window object
        self.windowHandle.resize(self.width, self.height) # Resizes the window the width and height specified
        self.windowHandle.setWindowTitle(self.title) # Sets the title to the string specified

        self.screenRecorderObj = mss() # A handle to the screen recording object
        self.recordingViewport = None # The viewport that the screen will record from
        self.recordingRate = 60 # The number of times per second that we will grab a new frame from the screen

        self.currentCapture = None # The current frame that was just captured
        self.videoFeed = QLabel(self.windowHandle) # The video feed object drawn to the window
        self.globalTimer = QTimer(self.windowHandle) # A global timer to keep track of when to grab a screenshot

    '''
        :desc:
            Creates the window when the function is called.
            It starts the global timer for updating the window
            and it also shows the video feed and renders it
            to the screen.
    '''
    def create(self):
        self.globalTimer.start(1000 // self.recordingRate)
        self.videoFeed.show()
        self.windowHandle.show()

    '''
        :desc:
            This takes in an image and sets our
            video feed equal to the image that's passed in.
            
        :param frame:
            This is the current frame of the screen
            capture that the user grabs
    '''
    def setCaptureFrame(self, frame):
        self.currentCapture = QPixmap.fromImage(QImage(frame.tobytes("raw", "BGR"), frame.size[0], frame.size[1], QImage.Format_RGB888))
        self.videoFeed.resize(frame.size[0], frame.size[1])
        self.videoFeed.move(self.width / 2 - frame.size[0] / 2, self.height / 2 - frame.size[1] / 2)
        self.videoFeed.setPixmap(self.currentCapture)

    '''
        :desc:
            This sets the rate (in frames per second)
            that the window will regrab a screenshot. In other
            words, you can set this to the FPS of the
            emulator to get a 1-1 frame-rate capture.
        
        :param timeInFPS:
            This is the fps that you want to record at.
    '''
    def setRecordRate(self, timeInFPS):
        self.recordingRate = timeInFPS

    '''
        :desc:
            This is the function that the user
            wants to be called every single time the
            window updates.
            
        :param func:
            This is the function that will be called
            every frame the window updates.
    '''
    def setUpdateFunc(self, func):
        self.globalTimer.timeout.connect(func)

    '''
        :desc:
            This sets the viewport (rectangle) of the screen
            that will be grabbed for recording.
        
        :param left:
            The number of pixels from the left of your
            screen the screen recording rectangle will start
            from.
            
        :param top:
            The number of pixels from the top of your
            screen the screen recording rectangle will start
            from.
            
        :param width:
            The width in pixels of the viewport rectangle
            
        :param height:
            The height in pixels of the viewport rectangle
    '''
    def setRecordingViewport(self, left, top, width, height):
        self.recordingViewport = {"left": left, "top": top, "width": width, "height": height}

    '''
        :desc:
            This actually grabs the screenshot from the window
            using the screen recording object we created earlier.
    '''
    def grabScreenshot(self):
        sourceImg = self.screenRecorderObj.grab(self.recordingViewport) # Grabs the source from the desktop
        img = Image.frombytes("RGB", sourceImg.size, sourceImg.rgb, "raw", "BGR") # Converts it to a better format
        return img
