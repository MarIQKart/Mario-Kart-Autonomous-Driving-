import time
import keyboard

class EmulatorInterface:

    '''
        :param emulator:
            This is a string that represents the name
            of the emulator that you're going to be simulating
            the key presses for. It is not case sensitive
    '''
    def __init__(self, emulator, game):
        emulator = emulator.lower() # Converts the string to all lower case
        game = game.lower() # Converts the string to all lower case
        '''
            This is the input mapping dictionary. The first
            parameter is the name of the emulator you're using.
            It's passed into the constructor of the EmulatorInterface
            object. It stores the names that we can use to modify
            the state of the driver in the emulator.
            
            throttle:
                The name of the state that makes the driver
                move forward
            
            left:
                The name of the state that makes the driver
                steer left
                
            right:
                The name of the state that makes the driver
                steer right
                
            up:
                The name of the state that makes the driver
                steer upwards
                
            down:
                The name of the state that makes the driver
                steer down
        '''
        self.inputMapping = \
            {
                "mupen 64":
                {
                    "mario kart":
                    {
                        "throttle": 'x',
                        "left": 'l',
                        "right": 'r',
                        "up": 'u',
                        "down": 'd'
                    }
                }
            }
        self.currentEmulatorMapping = self.inputMapping[emulator][game] # Here we keep track of the emulator inputs we need for a game

    '''
        :desc:
            This function simulates an input by emulating
            it with a keyboard mapping and sending it over to the
            emulator client.
    
        :param input:
            This is the input state that we wish to emulate
            with our interface.
    '''
    def emulatePress(self, input):
        if input in self.currentEmulatorMapping: # Checks to see if the input mapping exists
            inputMapping = self.currentEmulatorMapping[input] # Grabs the mapping
            self.holdKey(inputMapping, 0.15)
        else: # If there was no input mapping
            print("Not a valid input mapping") # Alert the user before quitting
            exit(-1) # Quits the program


    '''
        :desc:
            This function is used to hold a key
            for a certain amount of time. This is
            used because the emulator will have a certain amount
            of time that a key must be pressed for in order
            for the event to be registered by it.
            
        :param key:
            This is the key we want to have pressed down
            
        :param holdTime:
            This is the parameter that controls how
            long to hold the key down (the units are in seconds).
            
            For example, half a second: holdTime = 0.5
    '''
    def holdKey(self, key, holdTime):
        start = time.time() # gets the current time
        while (time.time() - start) < holdTime: # As long as the change in time is less than the holdTime
            keyboard.press(key) # Then keep pressing the key

    '''
        :desc:
            This function takes in an array, maps the inputs,
            and then presses all the keys in the array.
            
        :param inputs:
            This is an array of the inputs that can be entered
            to the function.
            
            Example:
                inputs = ["throttle", 'left', 'up']
                
                The emulator will press them in that order
    '''
    def emulatePresses(self, inputs):
        mappedInputs = [] # An array of the mapped inputs
        for input in inputs: # Loops through the inputs
            if input not in self.currentEmulatorMapping: # Checks to see if the input mapping exists
                print("Not a valid input mapping for: {}".format(input))
                exit(-1) # If it isn't a valid mapping, then fail
            inputMapping = self.currentEmulatorMapping[input] # Grabs the mapping
            mappedInputs.append(inputMapping) # Adds the valid mapping to the list

        '''
            Loops through the current mapped inputs and holds
            them down for 0.15 seconds (the minimum amount of time that
            the emulator will need in order for a key to be registered.)
        '''
        for key in mappedInputs:
            self.holdKey(key, 0.15) # Holds the current key

    '''
        :desc:
            This function takes in an array, maps the inputs,
            and then releases all the keys in the array.

        :param inputs:
            This is an array of the inputs that can be entered
            to the function.

            Example:
                inputs = ["throttle", 'left', 'up']

                The emulator will release them in that order
    '''
    def emulateReleasePresses(self, inputs):
        mappedInputs = []
        for input in inputs:
            if input not in self.currentEmulatorMapping:  # Checks to see if the input mapping exists
                print("Not a valid input mapping for: {}".format(input))
                exit(-1)
            inputMapping = self.currentEmulatorMapping[input]  # Grabs the mapping
            mappedInputs.append(inputMapping)

        for key in mappedInputs:
            keyboard.release(key)

    '''
        :desc:
            This function returns the dictionary
            of key presses that the emulator that you
            currently have selected is capable
            of emulating.
            
        :param mappings:
            A boolean that will let you display
            the current key mapping that corresponds to
            each emulatable key inside of the emulator.
            
        :returns:
            A dictionary of all the possible
            key presses (and their corresponding mappings)
            that the emulator currently has.
    '''
    def getPossibleKeyEmulations(self, mappings = False):
        if mappings: # If the user wants the mappings then return them
            return self.currentEmulatorMapping
        else: # Else, just give us the list of values the emulator can accept as inputs for presses
            return [x[0] for x in self.currentEmulatorMapping.items()]