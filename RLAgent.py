# ================================================================================
# FILE: RLAgent.py
# ================================================================================
# DESCRIPTION:
# ================================================================================
#
# Q-Table implemntation of a reinforcement learning agend for the program. It
# Will learn the appropriate actions to take to remain near the center of the
# track.
#
# ================================================================================



# ================================================================================
# IMPORTS & INITIALIZATION
# ================================================================================
#
#   * tensorflow: used for the CNN classifier used in state space VFA from image 
#                 frames
#
#   * gpus: used for extracting the physical gpus
#
#   * experimental memory growth = True: tensorflow was finding errors with 
#                  "finding the convolution algorithm" otherwise. It is untested
#                  whether or not this works on machines without a GPU
#
# ================================================================================
import tensorflow as tf
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)



# ================================================================================
# CLASS: RLAgent
# ================================================================================
# ATTRIBUTES:
# ================================================================================
#
#   - model_file:
#        * String containing the name of the file which the model will be saved
#          to and loaded from.
#
#   - action_space:
#        * List of the possible actions to take. For now it's been hard-coded to
#          be throttle, left, and right. We may revise to take all keys from the
#          EmulatorInterface's input mapping.
#
#   - classifier_file:
#        * String containing the path to the keras model used for the frame state classifier
#
#   - classifier_model
#        * keras model resulting from loading the classifier file. Used for interpreting
#          the input frame as a state.
#
#   - classifier_image_shape
#        * dimensions needed for the image that the classifier expects
#
#   - classifier_input_shape
#        * dimensions that that the model's input layer expects
#
#   - frame_classes
#        * dictionary mapping the indices of the output layer to specific classes
#             > 'center': mario is in the center of the screen
#             > 'near_left': mario is near the left of the track
#             > 'near_right': mario is near the right of the track
#             > 'off_left': mario is off the left side of the track
#             > 'off_right': mario is off the right side of the track
#             > 'wall_left': mario is facing a wall on the left side of the track
#             > 'wall_right': mario is facing a wall on the right side of the track
#             > 'tunnel_left': mario is facing the track with the tunnel opening on his left
#             > 'tunnel_right': mario is facing the track with the tunnel opening on his right
#
#   - q_table:
#        * the model stored in the form of a dictionary of lists. The keys of the
#          dictionary are the states, and the values are lists of the corresponding
#          (value,number) pairs for each action in the action space.
#             - Value = the current average learned reward for that action with that state
#             - Number = number of times the state-action pair has been rewarded (for average calcs)
#
#   - is_training:
#        * Boolean flag to designate whether or not model is in training mode
#             - True  = Perform learning
#             - False = Demo mode. No learning, no exploring.
#
#   - reward_discount:
#        * The rate at which reward decays when back-propagated
#
#   - episode:
#        * current number of episodes trained
#
#   - max_episodes:
#        * maximum number of training episodes before switching from training mode to demo
#
#   - explore_chance:
#        * Probability of exploration rather than using learned values
#
#   - explore_decay:
#        * Rate at which the chance of exploration decays after each episode
#
#   - explore_min:
#        * minimum exploration probability during training
#
#   - history:
#        * record of past state-action pairs for the current training episode
#
#   - propagate_interval:
#        * maximum length of a training episode
#
# ================================================================================
# CONSTRUCTOR:
# ================================================================================
#
# Input:
#   - use_existing_model (optional):
#        * Default = True
#        * Boolean flag to specify whether to load saved model or start from scratch
#   - is_training (optional):
#        * Default = True
#        * Boolean flag. True = training mode, False = Demo mode
#   - propagate_interval (optional):
#        * Default = 10
#        * The exact fixed length of each training episode
#   - max_episodes (optional):
#        * Default = 10000
#        * Number of training episodes before switching to demo mode
#
# Output:
#   - N/A
#
# Task:
#   - Initialize all attributes appropriately
#
# ================================================================================
# MEMBER FUNCTION: RLAgent.frame_to_state( frame )
# ================================================================================
#
# Input:
#   - frame:
#        * image representation of the current game's frame
#
# Output:
#   - string representing one of the 8 classes defined in self.frame_classes, representing
#     the model's classification of the input frame
#
# Task:
#   - Convert the given frame to grayscale
#   - Resize the frame to 80px wide by 64px tall
#   - use the classifier_model to predict the class of the frame
#        * classifier was trained to have an accuracy of 0.9980 on a 35,000 image
#          dataset (~26,000 training and ~9,000 validation)
#   - return the name of the class with the greatest prediction value
#
# ================================================================================
# MEMBER FUNCTION: RLAgent.update_explore_chance( )
# ================================================================================
#
# Input:
#   - N/A
#
# Output:
#   - no return
#
# Task:
#   - update the explore_chance attribute
#   - multiply by the explore decay
#   - set it to the maximum of that computed value and the minimum explore chance
#
# Note:
#   - We expect this function to be called internally as a private function, and
#     it is expected to only be called once at the end of each training episode.
#
# ================================================================================
# MEMBER FUNCTION: RLAgent.get_q_value( state , action_idx )
# ================================================================================
#
# Input:
#   - state:
#        * vector/tuple representation of the state being queried
#   - action_idx=None:
#        * None or the index of the specific action being queried
#        * if None: return the entire row (all values for all actions)
#
# Output:
#   - If an action_idx was specified, return a tuple of the form (V,N)
#        * V = learned Q value
#        * N = Number of times learned (for incremental average)
#   - If no action_idx specified, return a list of tuples of the form (V,N)
#        * one for each of the possible actions in self.action_space
#
# Task:
#   - Return the result of a QTable lookup for the specified state (and action),
#     with the potential for a default value if no value has been learned yet
#
# ================================================================================
# MEMBER FUNCTION: RLAgent.select_action( state )
# ================================================================================
#
# Input:
#   - state:
#        * tuple of the form computed by self.frame_to_state representing the current
#          state
#
# Output:
#   - The index of self.action_space corresponding to the chosen action
#   - Exact value depends on whether exploration or exploitation was done
#
# Task:
#   - If the agent is not in training mode, exploration chance becomes 0
#   - explore with a probability of exploration chance, otherwise exploit
#   - if exploring, select a random index to return
#   - if exploit, select the index which has the highest learned reward in the QTable
#
# ================================================================================
# MEMBER FUNCTION: RLAgent.calculate_reward( next_state )
# ================================================================================
#
# Input:
#   - next_state:
#        * The state which resulted from the previous action taken, used for computing
#          the reward value
#
# Output:
#   - Reward between 0 and 100*len(next_state) based on a custom reward function
#   - The more 0s, the higher the reward.
#
# Task:
#   - compute reward based on condition of the next state.
#
# ================================================================================
# MEMBER FUNCTION: RLAgent.apply_reward( state , action , reward )
# ================================================================================
#
# Input:
#   - state:
#        * The state to receive the reward
#   - action:
#        * the action to receive the reward
#   - reward:
#        * the reward being combined to the current learned reward
#
# Output:
#   - No return
#
# Task:
#   - perform an incremental average to include the given reward for the
#     state-action pair in the QTable.
#
# ================================================================================
# MEMBER FUNCTION: RLAgent.new_reward( state , action )
# ================================================================================
#
# Input:
#   - state:
#        * The state to receive the reward
#   - action:
#        * the action to receive the reward
#
# Output:
#   - value representing the reward for the action taken in the given state
#
# Task:
#   - lookup the base-reward table to determine reward values
#   - Return the reward value
#
# ================================================================================
# MEMBER FUNCTION: RLAgent.propagate_reward( )
# ================================================================================
#
# Input:
#   - N/A
#
# Output:
#   - No return
#
# Task:
#   - compute reward based next state
#   - iterate backwards through the recorded history of state-action pairs, applying
#     the reward to that state-action pair. 
#   - clear all states in the history that were rewarded (exclude the most recent
#     since it we don't apply reward to it)
#
# ================================================================================
# MEMBER FUNCTION: RLAgent.load_model( use_existing_model )
# ================================================================================
#
# Input:
#   - use_existing_model:
#        * flag specifying whether to load an existing model or create a new one
#
# Output:
#   - dictionary representing the model
#        * empty dict() if use_existing_model was false or if there was an error
#          loading self.model_file
#        * otherwise, keys are state vectors and values are list of (V,N) tuples
#             - V = learned value
#             - N = number of rewards applied (for incremental average)
#
# Task:
#   - if the given flag is true, try to load the model, parse to the appropriate
#     data types, and interpret values to form a QTable dictionary of the appropriate
#     structure
#   - if the given flag is false or the attemtp to load failed, return an empty dictionary
#
# ================================================================================
# MEMBER FUNCTION: RLAgent.save_model( )
# ================================================================================
#
# Input:
#   - N/A
#
# Output:
#   - No return
#   - File saved or overwritten
#
# Task:
#   - Open/Create self.model_file for writing
#   - Iterate through the key-value pairs in the QTable
#   - save each key value pair as a line in the file with the form:
#        * "key:value"
#
# ================================================================================
# MEMBER FUNCTION: RLAgent.act( frame )
# ================================================================================
#
# Input:
#   - frame:
#        * image representation/screenshot from the game's current display
#
# Output:
#   - a chosen action to take based on the given frame and learned knowledge
#        * one of the values within self.action_space
#
# Task:
#   - convert the frame to state with the use of value function approximation with
#     state-aggregation
#   - determine the action to take from the given state
#   - if training, update the history with the state-action p air
#   - if history is long enough for training episode to end, propagate reward
#   - return the selected action
#
# ================================================================================
class RLAgent:

    # ============================================================================
    # Constructor:
    # ============================================================================
    def __init__(self,
                 use_existing_model=True,
                 is_training=True,
                 episode_length=10,
                 max_episodes=10000):

        # === Save/Load Housekeeping === #
        self.model_file = 'model.txt'

        # === Action Space Housekeeping === #
        #self.action_space = [ 'left' , 'right' , 'throttle' , 'up' , 'down' ]
        self.action_space = [ 'left' , 'right' , 'throttle' ]
        
        # === State Space Classification Housekeeping === #
        self.classifier_file        = 'classifier_v4.h5'
        self.classifier_model       = tf.keras.models.load_model( self.classifier_file )
        self.classifier_image_shape = ( 80 , 64 )
        self.classifier_input_shape = (  1 , 80 , 64 , 1 )
        self.frame_classes          = {
            0:'center',
            1:'near_left',
            2:'near_right',
            3:'off_left',
            4:'off_right',
            5:'wall_left',
            6:'wall_right',
            7:'tunnel_left',
            8:'tunnel_right'
        }

        # === Initialize Model === #
        self.q_table = self.load_model(use_existing_model)

        # === Training Housekeeping === #
        self.is_training = is_training
        self.reward_discount = 0.9
        self.episode = 0
        self.max_episodes = max_episodes

        # === Explore/Exploit Housekeeping === #
        self.explore_chance = 0.50#0.99
        self.explore_decay = 0.90#0.99
        self.explore_min = 0.01

        # === History Housekeeping === #
        self.history = list()  # maybe give default if we want |history| > 1
        self.episode_length = episode_length

        # === Image data === #
        self.processedImage = None

        return  # __init__

    # ============================================================================
    # RLAgent.frame_to_state
    # ============================================================================
    def frame_to_state( self , frame ):

        # === Necessary Imports === #
        import numpy as np
        
        # === Process the Frame === #
        processed = frame.convert( 'L' ).resize( ( 80 , 64 ) )
        img_arr   = np.asarray( processed )
        self.processedImage = img_arr
        img_arr   = img_arr.reshape( ( 1 , img_arr.shape[0] , img_arr.shape[1] , 1) )
        result    = self.classifier_model.predict( img_arr )
        state     = np.argmax( result )
        
        # === Return the Frame's Class as the State === #
        return self.frame_classes[state]

    # ============================================================================
    # RLAgent.update_explore_chance
    # ============================================================================
    def update_explore_chance(self):

        # === If the Explore Chance can be Reduced === #
        if self.explore_chance > self.explore_min:
            # === Reduce it === #
            self.explore_chance *= self.explore_decay
            self.explore_chance = max(self.explore_chance, self.explore_min)

        return  # update_explore_chance

    # ============================================================================
    # RLAgent.get_q_value
    # ============================================================================
    def get_q_value(self, state, action_idx=None):

        # === If No Action Given, Give Full List === #
        if action_idx is None:
            return self.q_table.get(state, [(0, 0) for _ in range(len(self.action_space))])

        # === If Given Action, Return the Specific Value === #
        else:
            return self.q_table.get(state, [(0, 0) for _ in range(len(self.action_space))])[action_idx]

    # ============================================================================
    # RLAgent.select_action
    # ============================================================================
    def select_action(self, state):

        # === Import Numpy for Random Choice === #
        import numpy as np

        # === If Training, Use Stored Explore Probability === #
        if self.is_training:
            explore_chance = self.explore_chance

        # === If Demo, Never Explore === #
        else:
            explore_chance = 0

        # === If Explore, Choose Random Action === #
        if np.random.rand() < explore_chance:
            action_idx = np.random.choice(range(len(self.action_space)))

        # === If Exploit, Use Learned Action with Highest Reward === #
        else:
            q_value = self.get_q_value(state)
            max_value = max(q_value, key=lambda x: x[0])
            action_idx = q_value.index(max_value)

        # === Return the Chosen Action === #
        return action_idx  # select_action

    # ============================================================================
    # RLAgent.calculate_reward
    # ============================================================================
    def calculate_reward(self, next_state):
        # === Necessary Imports === #
        import numpy as np

        if next_state == 'center':
            reward = 0
        elif next_state in ['near_left', 'near_right']:
            reward = -50
        elif next_state in ['off_left','off_right','wall_left','wall_right']:
            reward = -100
        return reward

        # === Reward Function === #
        reward = 100 * len(next_state) - 100 * abs(np.sum(next_state))

        # === Return Calculated Reward === #
        return reward  # calculate_reward

    # ============================================================================
    # RLAgent.apply_reward
    # ============================================================================
    def apply_reward(self, state, action, reward):

        # === Get Working Values from Q Table === #
        q_value = self.get_q_value(state)
        V, N = q_value[action]

        # === Incremental Average Formula === #
        V = (V * N + reward) / (N + 1)
        N = N + 1

        # === Update "Table" === #
        q_value[action] = (V, N)
        self.q_table[state] = q_value

        return  # apply_reward
        
        
    # ============================================================================
    # RLAgent.new_reward
    # ============================================================================
    def new_reward( self , state , action ):
    
        # === Lookup Table for Base Reward Values === #
        reward_table = {
            'center':{
                'left':0,
                'right':0,
                'throttle':100
            },
            'near_left':{
                'left':-100,
                'right':100,
                'throttle':100
            },
            'near_right':{
                'left':100,
                'right':-100,
                'throttle':100
            },
            'off_left':{
                'left':-100,
                'right':100,
                'throttle':100
            },
            'off_right':{
                'left':100,
                'right':-100,
                'throttle':100
            },
            'wall_left':{
                'left':-100,
                'right':100,
                'throttle':0
            },
            'wall_right':{
                'left':100,
                'right':-100,
                'throttle':0
            },
            'tunnel_left':{
                'left':100,
                'right':-100,
                'throttle':100
            },
            'tunnel_right':{
                'left':-100,
                'right':100,
                'throttle':100
            }
        }
        
        # === Return the Reward === #
        return reward_table[state][self.action_space[action]]

    # ============================================================================
    # RLAgent.propagate_reward
    # ============================================================================
    def propagate_reward(self):
    
        # === Iteratively Reward Each State-Action Pair in the History === #
        for state,action in self.history:
            reward = self.new_reward( state , action )
            self.apply_reward( state , action , reward )
            
        # === Housekeeping for End of Episode === #
        self.update_explore_chance()
        self.episode += 1
        if self.episode >= self.max_episodes:
            self.is_training = False
            self.save_model()
        return

    # ============================================================================
    # RLAgent.load_model
    # ============================================================================
    def load_model(self, use_existing_model):

        # === Necessary Imports === #
        from ast import literal_eval

        # === If Told to Use Saved Model === #
        if use_existing_model:

            # === Try to Load the Model === #
            try:

                # === Load Model === #
                model = dict()
                with open(self.model_file, 'r') as infile:
                    model_text = infile.read().split('\n')

                # === Interpret Model Line-by-Line === #
                for line in model_text:
                    if line:
                        key, value = line.split(':')
                        key = literal_eval(key)
                        value = literal_eval(value)
                        model[key] = value

                # === If Successfully Loaded, Return the Model === #
                return model

            # === If Loading Fails, Return Empty Model Instead === #
            except:
                return dict()

        # === If Told Not to Use Saved Model === #
        return dict()

    # ============================================================================
    # RLAgent.save_model
    #
    #   @Update:
    #       Added optional fileName for a save-as feature
    # ============================================================================
    def save_model(self, fileName = None):
        # === Open Model File to Save === #
        if fileName is None:
            with open(self.model_file, 'w') as outfile:
                # === Iteratively Write "Key:Value" to File === #
                for state in self.q_table:
                    outfile.write('{}:{}\n'.format(state, self.q_table[state]))
        else:
            with open(fileName, 'w') as outfile:
                # === Iteratively Write "Key:Value" to File === #
                for state in self.q_table:
                    outfile.write('{}:{}\n'.format(state, self.q_table[state]))

        return  # save_model

    # ============================================================================
    # RLAgent.act
    # ============================================================================
    def act(self, frame):

        # === Convert Frame to State (VFA with State Aggregation) === #
        state = self.frame_to_state(frame)

        # === Select Action Given the State === #
        action_idx = self.select_action(state)

        # === If Training, Update History === #
        if self.is_training:
            self.history.append((state, action_idx))

        # === Propagate Reward When History is Desired Length === #
        if len(self.history) > self.episode_length:
            self.propagate_reward()

        print('DEBUG: Action = {}'.format(self.action_space[action_idx]))
        #print('DEBUG: Reward = {}\n'.format(self.reward(self.get_q_value(state,action)))

        # === Return Action Taken === #
        return self.action_space[action_idx]
