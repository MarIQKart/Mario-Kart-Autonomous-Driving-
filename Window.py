from PyQt5.QtWidgets import QLabel, QMainWindow, QScrollArea, QAction, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from RLAgent import RLAgent
import cv2

from mss import mss
from PIL import Image

class Window (QMainWindow):

    '''
        :param title:
            The title of the window that will be shown in the
            top of the screen.

        :param width:
            The width of the window in pixels.

        :param height:
            The height of the window in pixels.
    '''
    def __init__(self, title, x, y, width, height):
        super().__init__()

        '''
            Menu bar code for getting the menu bar to
            have file menus
        '''
        self.statusBar().showMessage("Waiting to start...")
        self.menuBarObject = self.menuBar()
        self.actionMenu = self.menuBarObject.addMenu("&File")

        '''
            We add an about menu so that you can find all
            of the keyboard shortcuts in case you forget.
        '''
        self.aboutMenu = self.menuBarObject.addMenu("&About")
        self.helpAction = QAction("&Help")
        self.helpAction.setShortcut("ctrl+h")
        self.helpAction.triggered.connect(self.helpFunc)
        self.aboutMenu.addAction(self.helpAction)

        '''
            We have a save mode action from the file
            menu
        '''
        self.saveModel = QAction("&Save Model")
        self.saveModel.setShortcut("ctrl+s")
        self.saveModel.triggered.connect(self.saveFunc)
        self.actionMenu.addAction(self.saveModel)

        '''
            We also have a save as action inside the
            file menu
        '''
        self.saveModelAs = QAction("&Save Model As")
        self.saveModelAs.setShortcut("ctrl+alt+s")
        self.saveModelAs.triggered.connect(self.saveAsFunc)
        self.actionMenu.addAction(self.saveModelAs)

        '''
            Adds an option menu to the status bar
        '''
        self.optionsMenu = self.menuBarObject.addMenu("&Options")


        '''
            We can toggle if the current AI is paused or not.
        '''
        self.togglePause = QAction("&Toggle Pause")
        self.togglePause.setShortcut("ctrl+p")
        self.togglePause.triggered.connect(self.pauseFunc)

        '''
            We can toggle between training and debug mode
            to see the current screen to examine the output
            that the AI will see.
        '''
        self.toggleTraining = QAction("&Toggle Training")
        self.toggleTraining.setShortcut("ctrl+t")
        self.toggleTraining.triggered.connect(self.toggleTrainingFunc)

        '''
            We can load an existing model as well from
            the file menu
        '''
        self.loadModel = QAction("&Load Model")
        self.loadModel.setShortcut("ctrl+k")
        self.loadModel.triggered.connect(self.loadModelFunc)

        '''
            This is a label to let us know what the current
            state of the AI program is.
            
            The first parameter is the paused status. Either
            PAUSED or UNPAUSED
            
            The second parameter is the current state,
            either TRAINING or DEBUG
        '''
        self.aiStateLabel = QLabel(self)
        self.aiStateText = "Current AI State: "
        self.aiStateLabel.setGeometry(0, 20, self.width(), 50)
        self.aiStateLabel.setText(self.aiStateText)
        self.aiStateLabel.show()

        '''
            This is a label that will tell us which model
            is currently loaded so we know what we're saving to.
            This is the name that gets saved to when you do
            a save command.
            
            When a new file is loaded this text is updated to reflect the
            current file that's open.
        '''
        self.currentAIModelInUse = QLabel(self)
        self.currentAIModelInUseText = "Current Loaded Model: "
        self.currentAIModelInUse.setGeometry(0, 80, self.width(), 20)
        self.currentAIModelInUse.setText(self.currentAIModelInUseText + "model.txt")
        self.currentAIModelInUse.show()

        '''
            This code just adds the submenus (actions) to the
            file menu.
        '''
        self.optionsMenu.addAction(self.togglePause)
        self.optionsMenu.addAction(self.toggleTraining)
        self.optionsMenu.addAction(self.loadModel)

        self.width = width # Width of the window in pixels
        self.height = height # Height pf the window in pixels
        self.title = title # The title of the window
        self.setGeometry(x, y, self.width, self.height)
        self.setWindowTitle(self.title) # Sets the title to the string specified

        '''
            It's hard to see the entire QTable if we don't put it
            inside of a QScrollArea. It will allow you to see
            the entirety of the QTable this way.
        '''
        self.qTableLabel = QLabel(self)
        self.qTableScroller = QScrollArea(self)
        self.qTableScroller.setGeometry(0, 100, self.width, self.height // 2)

        self.screenRecorderObj = mss() # A handle to the screen recording object
        self.recordingViewport = None # The viewport that the screen will record from
        self.recordingRate = 60 # The number of times per second that we will grab a new frame from the screen

        self.currentCapture = None # The current frame that was just captured
        self.videoFeed = QLabel(self) # The video feed object drawn to the window
        self.globalTimer = QTimer(self) # A global timer to keep track of when to grab a screenshot

        self.isPaused = True # Keeps track of if the AI is paused or not currently
        self.isTraining = True # Keeps track of if we are in debug mode or not
        self.currentAgent = None # Keeps track of the current agent parameters
        self.currentModelFile = None # Keeps track of the current model we have loaded

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
        self.show()

    '''
        :desc:
            This takes in an image and sets our
            video feed equal to the image that's passed in.
            
        :param frame:
            This is the current frame of the screen
            capture that the user grabs
            
        :param currentAction:
            This is the current action that the agent just
            took. It gets passed to this function in order to render it in the status bar.
            
        :param agent:
            This is the agent with the most updated parameters. Once again,
            it really just gets passed here so that we can take the QTable data
            from it in order to display it.
    '''
    def setCaptureFrame(self, currentEpisode = None, currentAction = None, agent = None):
        stateText = self.aiStateText # Stores a temporary state text variable
        self.currentModelFile = agent.model_file # Sets the current model file to whatever the agent is using

        '''
            If the program is paused, then update the
            state text to reflect it.
        '''
        if self.isPaused:
            stateText += "PAUSED\t"
        else: # If we aren't paused, then reflect it in the state text
            stateText += "UNPAUSED\t"

        '''
            If the current program is not in training mode,
            then we want to show the current AI debug video feed.
            This is just so that we can see exactly what our
            image processing has done to the source frame so that
            we can make sure that it did it correctly.
            
            This was mostly used when we were working with template matching,
            but it's still a good feature to have around so it stayed.
        '''
        if not self.isTraining:
            stateText += "DEBUG\t" # We're currently in debug mode, so reflect it in the state text

            '''
                Below we are just setting the current video feed and
                converting the parameters to a better format so that
                it can fit on the entire screen
            '''
            if agent is not None:
                if agent.processedImage is not None:
                    frame = agent.processedImage
                    frame = cv2.resize(frame, (int(frame.shape[1] * 3), int(frame.shape[0] * 3)))
                    self.currentCapture = QPixmap.fromImage(QImage(frame.tobytes(), frame.shape[1], frame.shape[0], QImage.Format_Grayscale8))
                    self.videoFeed.resize(frame.shape[1], frame.shape[0])
                    self.videoFeed.move(self.width / 2 - frame.shape[1] / 2, self.height / 2 - frame.shape[0] / 2)
                    self.videoFeed.setPixmap(self.currentCapture) # Lastly, we show the video feed by setting the pixmap
        else: # If we're in training mode
            if agent is not None: # As long as the agent has been initialized we can train

                '''
                    Update the stored version of the agent so that we can make
                    sure that we have the most up to date Q-Table and other variables
                '''
                self.currentAgent = agent

                stateText += "TRAINING" # Reflect that we're training in the state text

                qTable = agent.q_table # Gets a copy of the Q-Table from the agent

                '''
                    Here we add a header to the QScrollArea that labels it as the
                    Q-Table (in case it wasn't clear)
                '''
                outputText = ("-" * 68) + "QTable" + ("-" * 68) + "\n\n"

                '''
                    The Q-Table is stored as a dictionary, so we loop through
                    all the keys and value pairs so that we can write them in the
                    QScrollArea.
                '''
                for key, value in qTable.items():
                    outputText += "\tKey: {}\tValue: {}\n".format(str(key), str(value))

                '''
                    Here we store the text we created from the Q-Table and
                    change the size of the label that we use to write it to
                    the ScrollArea with.
                '''
                self.qTableLabel.setText(outputText)

                self.qTableLabel.setGeometry(0, 0, self.qTableScroller.width(), self.height // 2) # Sets the label size
                self.qTableScroller.setWidget(self.qTableLabel) # Adds the label to the scroll area
                self.qTableScroller.hide() # Hides the old scroller
                self.qTableScroller.show() # Updates it by re-showing it

        '''
            If the current episode is set equal to some value, then we need
            to update the status bar (at the very bottom of the screen) to
            reflect the current episode that we're on, as well as the last
            action that was taken.
        '''
        if currentEpisode is not None:
            self.statusBar().showMessage("Current Episode: " + str(currentEpisode) + "\tCurrent Action: " +
                                         currentAction)
        self.aiStateLabel.setText(stateText) # Sets the state text we gathered from this method
        self.update() # Updates the entire main window widget

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
        
        :returns:
            Returns the current image that was in the screenshot
    '''
    def grabScreenshot(self):
        sourceImg = self.screenRecorderObj.grab(self.recordingViewport) # Grabs the source from the desktop
        img = Image.frombytes("RGB", sourceImg.size, sourceImg.rgb, "raw", "BGR") # Converts it to a better format
        return img

    '''
        :desc:
            This function will toggle the current
            paused state when it gets called.
    '''
    def pauseFunc(self):
        self.isPaused = not self.isPaused

    '''
        :desc:
            This function will toggle between training mode
            and debug mode. It will hide the appropriate widgets that
            don't get used for this mode, and it will show the ones
            that need to be seen.
    '''
    def toggleTrainingFunc(self):
        self.isTraining = not self.isTraining # Toggles the training state

        '''
            If we are in training mode after the swap
        '''
        if self.isTraining:
            self.videoFeed.hide() # Hide the video feed
            self.qTableScroller.show() # Show the scroller
            self.qTableLabel.show() # SHow the label
        else: # If we're in debug mode
            self.videoFeed.show() # Show the video feed
            self.qTableScroller.hide() # Hides the scroller
            self.qTableLabel.hide() # Hides the label

    '''
        :desc:
            This is the Save-As function. It gets the user input
            and saves the Q-Table as whatever the user wants to save
            it as.
    '''
    def saveAsFunc(self):

        '''
            Gets the user input in a file dialog
        '''
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Model")

        '''
            As long as the user actually entered a file name
            (and didn't hit cancel), then grab the file name
            they entered and use it.
        '''
        if fileName is not None and len(fileName) > 0:
            print("Trying to save: {}".format(fileName))
            self.currentAgent.save_model(fileName)

    '''
        :desc:
            This function saves the current file to whatever
            the open file is.
    '''
    def saveFunc(self):

        '''
            If the user hasn't entered a custom file name,
            then use the default.
        '''
        if self.currentModelFile is not None:
            self.currentAgent.save_model() # Saves the model with the default name
        else:
            '''
                 If the user has entered another file name,
                 either by loading a new file or clicking save-as,
                 then use it. 
             '''
            self.currentAgent.save_model(self.currentModelFile) # Saves the file with the custom name

    '''
        :desc:
            This function will get the user input for a file to load to be
            used as the model. It will then reflect the changes in the Q-Table scroller
    '''
    def loadModelFunc(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Load Model")
        if fileName is not None and len(fileName) > 0:
            print("Trying to load: {}".format(fileName))
            self.currentModelFile = fileName
            self.currentAgent.load_model(fileName)
            self.currentAIModelInUse.setText(self.currentAIModelInUseText + fileName[fileName.rfind("/") + 1:])

    '''
        :desc:
            This function will open up a popup
            box whenever the user needs a reference to the
            shortcuts.
    '''
    def helpFunc(self):
        msgBox = QMessageBox()
        msgBox.setInformativeText("Shortcuts Help: ctrl + h\nSave: ctrl + s\nSave-As: ctrl + alt + s\n"
                                  "Load Model: ctrl + k\nPause: ctrl + p\nToggle training/debug: ctrl + t")

        msgBox.setWindowTitle("Shortcuts Reference Menu")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.resize(500, 300)
        returnValue = msgBox.exec()
        msgBox.show()