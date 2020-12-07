# MarI/O Kart: Autonomous Driving in Mario Kart

## Problem Statement 

The goal is to create a reinforcement agent that is capable of learning how to drive autonomously across different versions of Mario-Kart games. The desired agent will play the game in the Time Trials mode in order to simplify the problem. In Time Trials mode there is only one player on the track, and the objective is to get lap times down to as little as possible. The goal for our agent was to not only complete laps, but to be able to learn generically enough to learn regardless of Mario Kart game environment and translate any knowledge across the various environments. The agent will be trained on a Nintendo-64 version of the game, as well as a GameCube version of the game; after which it will be determined if either of the games were better for training the model for playing Mario-Kart. The agent was trained using console emulation on the PC, meaning that no physical Nintendo-64 or GameCube consoles are required in order to train the model or run the program.

## Circumstances of the Program:

The project has been tested to run successfully on Windows 10 for 64-bit architecture and x86 architecture. No testing has been done for other operating systems or architectures. 

Though the emulator may sometimes experience crashing, we expect that this issue is not due to our program, but due to issues within the emulator itself and its resource allocation over extended use (over 40mins of continuous use while screen recording software was running)

The program has been tested with the use of tensorflow-gpu with cudnn support for the training and implemntation of the CNN state-aggregation classifier. We have not tested the performance without the use of a GPU

## Files:

* **`CNN/`:** this directory contains all file files and information relevant to training the CNN classifier used in state-aggregation. For more information about the contents of this directory, see `CNN/NN_readme.md`.
* **`EmulatorInterface.py`:** class method used by the program for interfacing with the emulator window. This file is responsible for managing emulated keypresses and other interactions with the game window.
* **`Graphics.py`:** contains function definitions necessary for operating upon, transforming, and producing graphics.
* **`HitboxFinder.py`:** (deprecated) This file was used for locating Mario's hitbox within the captured frame using a Template Matching algorithm through open CV
* **`RLAgent.py`:** python class definition for the class which performs the reinforcement learning operations, including action decision, state aggregation, and maintenance of the Q-Table used for learning
* **`Window.py`:** contains class definitions used for the capture of the game window and gui display for our program's window.
* **`classifier.h5`:** (deprecated) this file contains the keras weights for the original classifier with the use of only 5 states
* **`classifier_v2.h5`:** (deprecated) this file contains the keras weights for an updated classifier which uses 7 states. 
* **`classifier_v3.h5`:** (deprecated) this file contains the keras weights for a further updated classifier which uses 9 states.
* **`classifier_v4.h5`:** this file contains the keras weights for the final version of the classifier, which utilizes both N64 and GameCube frames for training the 9 classes, rather than just the N64 frames used in the previous versions
* **`frame_array_to_state.py`:** (deprecated) this file contains a function definition for implementing Canny Edge detection for interpretation of image frames into feature vectors
* **`main.py`:** main driver of the program. This file should be run in order to run the program
* **`template.png`:** (deprecated) this file contains the template image used for the template matching algorithm

## Running the Program:

1. Ensure all dependencies are installed in your active python environment
2. Using your emulator of choice (Mumpen64 or Dolphin in our case), run the game's ROM file.
3. Run `main.py` in your active python environment. 
4. (optional) toggle debug mode in our program's GUI that displays after tensorflow finishes activation. From here, ensure that the rom's window is shown in the debug window
5. Navigate your ROM to the desired Mario Kart track (only Mario is supported for N64, and Mario+Luigi for GameCube). Only Luigi's Raceway has been tested; support for other characters/tracks is unknown.
6. Once the race begins, toggle our program back to "training mode" and click on the rom window. (some users may need to press the throttle key once in order to "instantiate focus" on the ROM program before emulated keypresses are recognized)
7. Behold the RL Agent learning to drive in Mario Kart.
