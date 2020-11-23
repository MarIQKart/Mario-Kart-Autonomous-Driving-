from PyQt5.QtWidgets import QApplication
from Window import Window
from EmulatorInterface import EmulatorInterface
from RLAgent import RLAgent

import sys


'''
    Creates a global Agent object in order
    to train the model
'''
agent = RLAgent()


'''
    :desc:
        This is the update function that can be
        used each frame to take control of the emulator.
        Below are some examples in the comments where you can
        use an emulator object to map key presses
        to the 
'''
def onUpdate(window, emulator):
    src = window.grabScreenshot()
    global agent

    '''
        TODO:
            Here we can simulate the AI pressing the buttons.
            Examples:
            
            emulator.emulatePress("throttle") # Emulates driving forward
            emulator.emulatePress("right") # Emulates steering to the right
            emulator.emulatePress("left") # Emulates steering to the left
    '''

    actionTaken = "Paused ---> No action"

    if not window.isPaused:
        '''
             The agent will take an action based on the current source
             frame that it is given as an input. It will then
             return the action that it took (L or R)
         '''
        actionTaken = agent.act(src)

        '''
            The emulator will get the action taken, and it will
            use that action to map a key press to it.
        '''
        emulator.emulatePresses([actionTaken])

        '''
            After we press them we have to clear the even hooks
            so that the presses get released between frames.
    
        '''
        emulator.emulateReleasePresses([actionTaken])

    '''
        Updates the current window capture frame with the source image,
        the currentEpisode (see status bar), the current action (see status bar),
        as well as if we are debugging or training at the current moment in time.
        
        If we're in debug mode, then then a direct recording of the game is shown
        in the emulator. However, if it isn't then the emulator doesn't render
        the direct recording of the game and it will display the current
        actions being taken in a visual format instead.
        
        If you're in training mode, the program will display the Q-Table as well in a
        visual way so that we can see exactly what's going on with the data.
    '''

    window.setCaptureFrame(currentEpisode = agent.episode, currentAction = actionTaken, agent = agent)


def main():
    app = QApplication(sys.argv) # Create the application
    window = Window("Mario AI Software", 1000, 50, 900, 1200) # This is the default size of the emulator when it opens
    window.setRecordingViewport(0, 110, 900, 683) # This is the default size of the emulator when it opens
    window.setRecordRate(30) # Tells the window to record at 30fps

    emu = EmulatorInterface("Mupen 64", "mario kart") # Creates a Mupen 64 emulator object for mario kart

    window.setUpdateFunc(lambda: onUpdate(window, emu)) # Tells the update function to grab a screenshot 30fps
    window.create() # Creates the window given the parameters we've already set
    sys.exit(app.exec_()) # Tells the program to check for exit

if __name__ == "__main__":
    main()
