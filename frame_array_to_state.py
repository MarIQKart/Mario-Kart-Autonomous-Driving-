# ================================================================================
# Function: frame_to_state
# ================================================================================
#
# Input:
#   * frame_img:     numpy array representing the rgb pixel-array of the frame 
#                    (assumed minimap is toggled off the road)
#   * n_features:    The desired length of the state vector (size of state space
#                    = 3^n_features. I would recommend no more than 7)
#   * center_margin: acceptable margin for the average canny edge to be considered "center"
#
# Output:
#   - Tuple of length n_features containing values {0,-1,1}
#        * 0: the average detected canny edge over the sample slice is within center_margin
#             left or right of the center
#        * 1: the average detected canny edge over the sample slice is center_margin right of
#             the center
#        * -1: the average detected canny edge over the sample slice is center_margin left
#              the center.
#
# Task:
#   1. Convert the Frame to Grayscale
#   2. Resize the frame by the given scaling factor
#   3. Find the canny edges in the transformed image
#   4. Select a row to represent each feature (number of rows=desired number of 
#      features)
#   5. Find the average index of the canny edges in each representative row
#   6. Map the average to a representative feature value
#         * -1: the average is on the left side of the screen
#         *  0: the average is "close enough" to the center (within center_margin*width)
#         * +1: the average is on the right side of the screen
#   7. Return the representative feature values as an ordered tuple
#
# ================================================================================
def frame_to_state( frame_img , n_features=5 , center_margin=0.05 ):
    
    # === Necessary Imports === #
    from PIL import Image
    
    # === Housekeeping === #
    resize_scale = 5
    
    # === Image Preprocessing === #
    img   = Image.fromarray( np.uint8( frame_img ) )
    gray  = img.convert( 'L' )
    small = gray.resize( ( gray.size[0]//resize_scale , gray.size[1]//resize_scale ) )
    edges = cv2.Canny( np.asarray( small ) , 150 , 200 )
    
    # === Select Representative Rows === #
    rows = [ i for i in range( edges.shape[0]//2 , edges.shape[0]-int(edges.shape[0]*0.1) , (edges.shape[0]-int(edges.shape[0]*0.1))//2//n_features ) ]
    while len( rows ) > n_features:
        rows = rows[:-1]
        
    # === Determine Threshold for "Center" === #
    center = edges.shape[1]//2
    left   = center - edges.shape[1]*center_margin
    right  = center + edges.shape[1]*center_margin
    
    # === Generate the State "Vector" === #
    state = tuple( )
    for rowslice in edges[rows]:
        locs = np.where(rowslice==255)[0]
        avg  = locs.mean( ) if len( locs ) > 0 else center
        
        # === More Left Edges than Right === #
        if avg < left:
            state += (-1,)
            
        # === More Right Edges than Left
        elif avg > right:
            state += (1,)
            
        # === Edges Approximately Even on Left and Right === #
        else:
            state += (0,)
    
    # === Return the State === #
    return state
