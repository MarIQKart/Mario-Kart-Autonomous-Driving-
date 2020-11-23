# ================================================================================
# FILE: HitboxFinder.py
# ================================================================================
# DESCRIPTION:
# ================================================================================
#
# This file contains a class definition for a HitboxFinder, which determines the
# hitbox of the vehicle in the given frame using a Template Matching algorithm
# with the use of opencv (opencv-python) for Template Matching, numpy for
# representation of pixel arrays, Python Image Library (PIL) for use reading the
# template image file, and matplotlib.pyplot for debugging purposes to draw the
# image with hitbox as a pyplot
#
# ================================================================================
# REQUIREMENTS
# ================================================================================
#
#   * numpy:
#        `pip install numpy`
#
#   * PIL (Python Image Library)
#        `pip install Pillow`
#
#   * cv2 (opencv for python, version 2)
#        `pip install opencv-python`
#
#   * matplotlib
#        `pip install matplotlib`
#
# ================================================================================


# ================================================================================
# CLASS: HitboxFinder
# ================================================================================
# ATTRIBUTES:
# ================================================================================
#
#   - template_file:
#        * string containing the file path to the template file being used
#
#   - template:
#        * numpy array containing the pixel representation of the template file
#
#   - x_offset:
#        * How far (in pixels) to shift the x dimension of the hitbox.
#        * Positive = rightward
#        * Negative = leftward
#
#   - y_offset:
#        * How far (in pixels) to shift the y dimension of the hitbox
#        * Positive = Down
#        * Negative = Up
#
#   - width:
#        * Width (in pixels) of the hitbox
#
#   - height:
#        * Height (in pixels) of the hitbox
#
# ================================================================================
# CONSTRUCTOR:
# ================================================================================
#
# Input:
#   - template_file:
#        * string to be assigned as the template_file attribute
#        * file will be read as an image to store as template attribute
#        * valid image filename/filepath assumed, no exception handling for bad values
#        * it is assumed the width and height of the tempalte are smaller than the frame
#   - x_offset (optional):
#        * Default = 0
#        * Number of pixels to shift the x dimension of the hitbox
#        * Positive = right, negative = left
#        * It is assumed that since the driver is near-center of the screen, this
#          offset can be relatively large, and therefore, no error checking is present
#          for when the offset would place the hitbox offscreen
#   - y_offset (optional):
#        * Default = 0
#        * Number of pixels to shift the x dimension of the hitbox
#        * Positive = down, negative = up
#        * It is assumed that since the driver is near-center of the screen, this
#          offset can be relatively large, and therefore, no error checking is present
#          for when the offset would place the hitbox offscreen
#   - width (optional):
#        * Default = 0
#        * width in pixels for the hitbox
#        * it is assumed that the width is smaller than the width of the frame
#        * it is assumed the width will be small enough to fit the hitbox entirely within
#          the frame when placed approximately center
#        * if given a value <= 0, reassign to the width of the template
#   - height (optional):
#        * Default = 0
#        * height in pixels for the hitbox
#        * it is assumed that the height is smaller than the width of the frame
#        * it is assumed the height will be small enough to fit the hitbox entirely within
#          the frame when placed approximately lower-center of the frame
#
# Output:
#   - N/A
#
# Task:
#   - Initialize all attributes appropriately
#
# ================================================================================
# OVERLOAD: __repr__ ( string representation of class )
# ================================================================================
#
# Input:
#   - N/A
#
# Output:
#   - formatted text contianing all the attributes except the template as a string
#
# Task:
#   - Create a string labelling each attribute and containing the attribute's value
#   - return the string
#
# Note:
#   - This function is essentially a "core dump" to be used for debugging purposes
#
# ================================================================================
# MEMBER FUNCTION: HitboxFinder.get_image( filename )
# ================================================================================
#
# Input:
#   - filename:
#        * name of the file to be extracted as a numpy array
#
# Output:
#   - numpy array containing the pixel representation of the image file
#
# Task:
#   - Use Python Image Library (PIL) to open the image file
#   - use numpy's asarray() to cast the image to a numpy array
#   - return the resulting array
#
# Note:
#   - There is no error handling present for bad files. It is assumed the given file is
#     valid
#   - The PIL and numpy packages are required
#
# ================================================================================
# MEMBER FUNCTION: HitboxFinder.show_template( )
# ================================================================================
#
# Input:
#   - N/A
#
# Output:
#   - No Return
#   - Matplotlib figure is shown containing the template image from the file given attribute
#     construction
#
# Task:
#   - Use matplotlib.pyplot.imshow() to show the template image file
#
# Note:
#   - This function is intended for use in debugging only, as depending on IDE,
#     program may hang until the shown image is closed
#
# ================================================================================
# MEMBER FUNCTION: HitboxFinder.get_hitbox_corners( frame )
# ================================================================================
#
# Input:
#   - frame:
#        * numpy array representing the pixels (rgb) of the game's current frame
#          to find the hitbox in
#
# Output:
#   - tuple of the form (top_left, top_right, bottom_left, bottom_right) where
#        * each item in the tuple corresponds to a corner of the hitbox
#        * each item in the tuple itself is a tuple of the form (x,y) containing the
#          x and y coordinates of the given frame at which the hitbox corner is
#          present
#
# Task:
#   - Use opencv's (cv2) matchTemplate function to find the pixel with the lowest
#     square difference to the template
#   - That pixel's location is the top left
#   - Calculate appropriate x and y offset from the corresponding attributes
#   - compute the remaining corners from the top left corner using the width and
#     height attributes
#   - return the comptued corners in the appropriate order
#
# ================================================================================
# MEMBER FUNCTION: HitboxFinder.draw_hitbox( frame , hitbox )
# ================================================================================
#
# Input:
#   - frame:
#        * current game frame for which to draw the hitbox on
#        * represented as a numpy array of pixels (rgb)
#   - hitbox=None:
#        * if a value is given, it is expected to be a tuple of the form produced
#          by HitboxFinder.get_hitbox_corners(), and be corresponding to the given
#          frame
#        * computing template matching can be expensive O(n) where n is # of pixels
#          in the frame (900x683 for n64), so save on computation whenever possible
#        * if nothing is given, self.get_hitbox_corners() will be called on the given
#          frame to compute
#
# Output:
#   - no return
#   - image with the hitbox (given or computed) drawn is shown through
#     matplotlib.pyplot.imshow()
#
# Task:
#   - compute the hitbox if none is given
#   - use opencv to draw a rectangle from the top_left to bottom_right corners
#     of the hitbox onto a copy of the given figure
#   - use matplotlib to show the copy with the hitbox drawn on it
#
# Note:
#   - This function uses matplotlib to show the image, which, IDE dependent, may
#     cause the program to hang until the shown figure is closed
#   - Because of this, it is intended that this function be used for debugging
#
# ================================================================================
# TODO
# ================================================================================
#
# 1. (Maybe) Rewrite the draw_hitbox function to overlay the hitbox on the emulator
#    instead of use in matplotlib. Or write a new function to do so. It may not
#    be necessary to draw the hitbox on the overlay for real-time performance
#
# 2. (Maybe) integrate this into the (to be created) class for state interpretation.
#    the hitbox location is critical for determining the current state, but we may
#    not require state interpretation if we decide a neural network will work better,
#    taking the frame image as input, and outputting the control directly. This depends
#    on whether the course covers neural nets in the next few lectures
#
# 3. (Maybe) further optimize the template matching speed (since 30fps may be hard
#    with such an expensive algorithm). This can be done by checking a cropped version
#    of the frame and applying transformations on the hitbox coordinates to correct
#    for the crop. (say mario stays in the lower 2/3 of the frame, we could crop the
#    top ~230 pixels, and if we know he stays 200px from the left/right of the screen,
#    thats another ~400px to crop, and say we know he stays above (<) 600, that's another
#    ~83 pixels to crop, leaving a 400x470=188,000 instead of 900x683=614700.
#       * "Maybe" for two reasons:
#            1. We lose generality by doing this, there may be edge-cases where
#               mario('s hat) is barely out of bounds of the crop, and the hitbox is
#               wrong
#            2. Doing this is again reliant on whether we use states at all
#
# ================================================================================
import cv2
import numpy as np
import Graphics as gfx # Imports the custom drawing module I created

class HitboxFinder:

    # ============================================================================
    # Constructor
    # ============================================================================
    def __init__(self, template_file, x_offset=0, y_offset=0, width=0, height=0):
        self.template_file = template_file
        self.template = self.get_image(template_file)
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.width = width if width > 0 else self.template.shape[0]
        self.height = height if height > 0 else self.template.shape[1]
        return

    # ============================================================================
    # String Representation Overload
    # ============================================================================
    def __repr__(self):
        output = ''
        output += 'Template File Path: {}\n'.format(self.template_file)
        output += 'Template Image Dimensions: {}\n'.format(self.template.shape)
        output += 'Hitbox X Offset (positive=right): {}\n'.format(self.x_offset)
        output += 'Hitbox Y Offset (positive=down): {}\n'.format(self.y_offset)
        output += 'Hitbox Width: {}\n'.format(self.width)
        output += 'Hitbox Height: {}\n'.format(self.height)
        return output

    # ============================================================================
    # HitboxFinder.get_image( filename )
    # ============================================================================
    def get_image(self, filename):
        # === Necessary Imports === #
        # === Open, Cast to Numpy, and Return === #
        imgData = cv2.imread(filename)
        return np.asarray(imgData)

    # ============================================================================
    # HitboxFinder.show_template() -- intended use: debugging
    # ============================================================================
    def show_template(self):
        # === Necessary Import === #
        cv2.imshow("IMG", self.template)

    # ============================================================================
    # HitboxFinder.get_hitbox_corners( frame )
    # ============================================================================
    def get_hitbox_corners(self, frame):
        # === Import === #

        # === Template Matching === #
        convertedFrame = np.asarray(frame)
        match_heatmap = cv2.matchTemplate(convertedFrame, self.template, cv2.TM_SQDIFF)

        # === Get the Appropriate Coordinate for Hitbox === #
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_heatmap)

        # === Compute Appropriate Transformations and Dimensions of Hitbox === #
        top_left = (min_loc[0] + self.x_offset, min_loc[1] + self.y_offset)
        top_right = (top_left[0] + self.width, top_left[1])
        bottom_left = (top_left[0], top_left[1] + self.height)
        bottom_right = (top_left[0] + self.width, top_left[1] + self.height)

        # === Return Ordered Tuple === #
        return (top_left, top_right, bottom_left, bottom_right)

    # ============================================================================
    # HitboxFinder.draw_hitbox( frame , hitbox=None ) -- intended use: debugging
    #
    # CHANGES:
    #   1) Adjusted to check the bounds of the image under the event that mario wasn't
    #   detected in this frame (which happens on occasion)
    #
    #   2) Uses the Graphics library I made to make it easy to add a rectangle to the
    #      image
    # ============================================================================
    def draw_hitbox(self, frame, hitbox=None):

        # === Compute Hitbox if Not Given === #
        if hitbox is None:
            hitbox = self.get_hitbox_corners(frame)

        # === Draw the Hitbox === #
        top_left = hitbox[0]
        top_right = hitbox[1]
        bottom_left = hitbox[2]
        bottom_right = hitbox[3]

        '''
            @update:
                Modified the code to display the current
                hitbox so that it could be integrated with
                the rest of our drawing code.
        '''
        x = top_left[0]
        y = top_left[1]
        w = top_right[0] - top_left[0] # Difference in X
        h = bottom_right[1] - top_right[1] # Difference in Y

        imageWidth, imageHeight = frame.size # Gets the image size
        if (0 < x < imageWidth) and (0 < y < imageHeight): # Checks if X and Y are in bounds
            if (0 < x + w < imageWidth) and (0 < y + h < imageHeight): # Checks if x + w and y + h are in bounds

                '''
                    @update:
                        gfx class will write the bounding box
                        onto the current frame as it's passed through
                        our rendering pipeline.
                '''
                gfx.addRect(frame, x, y, w, h, 255, 0, 0) # If the rect it found was in bounds, draw it