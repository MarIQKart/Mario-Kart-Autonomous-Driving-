import keyboard # keyboard library for sending native WindowsOS key-presses to the emulators

class EmulatorInterface:

    '''
        :param emulator:
            This is a string that represents the name
            of the emulator that you're going to be simulating
            the key presses for. It is not case sensitive
    '''
    def __init__(self, emulator):
        emulator = emulator.lower() # Converts the string to all lower case

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
                    "throttle": "x",
                    "left": "left",
                    "right": "right",
                    "up": "up",
                    "down": "down"
                },
            }
        self.currentEmulatorMapping = self.inputMapping[emulator] # Here we keep track of the emulator inputs we need


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
            keyboard.press(inputMapping) # Presses the key corresponding to the mapping
        else: # If there was no input mapping
            print("Not a valid input mapping") # Alert the user before quitting
            exit(-1) # Quits the program

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
    def getPossibleEmulations(self, mappings = False):
        if mappings: # If the user wants the mappings then return them
            return self.currentEmulatorMapping
        else: # Else, just give us the list of values the emulator can accept as inputs for presses
            return [x[0] for x in self.currentEmulatorMapping.items()]