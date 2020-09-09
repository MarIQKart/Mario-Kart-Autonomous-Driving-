from PyQt5.QtWidgets import QApplication
import sys
from Window import Window
from EmulatorInterface import EmulatorInterface

'''
    :desc:
        This is the update function that can be
        used each frame to take control of the emulator.
        Below are some examples in the comments where you can
        use an emulator object to map key presses
        to the 
'''
def onUpdate(window, emulator):
    window.setCaptureFrame(window.grabScreenshot()) # Sets the current screenshot to the video feed in the program

    '''
        TODO:
            Here we can simulate the AI pressing the buttons.
            Examples:
            
            emulator.emulatePress("throttle") # Emulates driving forward
            emulator.emulatePress("right") # Emulates steering to the right
            emulator.emulatePress("left") # Emulates steering to the left
    '''

def main():
    app = QApplication(sys.argv) # Create the application
    window = Window("Mario AI Software", 900, 683) # This is the default size of the emulator when it opens
    window.setRecordingViewport(0, 110, 900, 683) # This is the default size of the emulator when it opens
    window.setRecordRate(30) # Tells the window to record at 30fps

    emu = EmulatorInterface("Mupen 64") # Creates a Mupen 64 emulator object
    print(emu.getPossibleEmulations(mappings = True)) # Prints the emulatable key presses and their mappings

    window.setUpdateFunc(lambda: onUpdate(window, emu)) # Tells the update function to grab a screenshot 30fps
    window.create() # Creates the window given the parameters we've already set
    sys.exit(app.exec_()) # Tells the program to check for exit

if __name__ == "__main__":
    main()
