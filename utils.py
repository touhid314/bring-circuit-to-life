'''
file to contain various utility functions and algorithms
'''
import numpy as np


def get_bounding_box_edge_pixels(xywh, img_w, img_h):
    """
    Returns separate arrays of (x, y) coordinates representing the top, left, right, 
    and bottom edge pixels of a bounding box in non-normalized pixel values.

    Args:
    - xywh: list or tuple containing [x, y, w, h] where:
        - x, y: center coordinates of the bounding box as fractions (0 to 1).
        - w, h: full width and full height of the bounding box as fractions (0 to 1).
    - img_w: Width of the image in pixels.
    - img_h: Height of the image in pixels.

    Returns:
    - Four NumPy arrays of (x, y) pixel coordinates for each edge:
      (top_edge, left_edge, right_edge, bottom_edge)
    """
    x, y, w, h = xywh
    
    # Calculate half-width and half-height in pixels
    half_w = int((w * img_w) / 2)
    half_h = int((h * img_h) / 2)
    
    # Calculate the bounding box corners in absolute pixel coordinates
    center_x = int(x * img_w)
    center_y = int(y * img_h)
    
    left = center_x - half_w
    right = center_x + half_w
    top = center_y - half_h
    bottom = center_y + half_h

    # Collect edge pixels without normalization
    top_edge = np.array([(i, top) for i in range(left, right + 1)], dtype=np.int32)
    bottom_edge = np.array([(i, bottom) for i in range(left, right + 1)], dtype=np.int32)
    left_edge = np.array([(left, j) for j in range(top, bottom + 1)], dtype=np.int32)
    right_edge = np.array([(right, j) for j in range(top, bottom + 1)], dtype=np.int32)

    return top_edge, left_edge, right_edge, bottom_edge

# # Example usage
# # Assuming `skeleton_ckt` is a 2D NumPy array representing the original image
# image = skeleton_ckt.copy()  # Assuming `skeleton_ckt` is a predefined numpy array of the image
# img_h, img_w = image.shape

# # Define the bounding box as normalized coordinates (example values)
# xywh = [0.228522, 0.496144, 0.0996564, 0.267352]
# top_edge, left_edge, right_edge, bottom_edge = get_bounding_box_edge_pixels(xywh, img_w, img_h)


def get_COMPONENTS(skeleton_ckt, comp_bbox):
    '''
    args: 


    return:
    the COMPONENT array
    '''
    # in pil: pixel => (x, y) 
    # in numpy: pixel => (y, x) y= row value, x = column value

    from simulate import NON_ELECTRICAL_COMPS

    # define all data structrues
    COMPONENTS = []


    NODE_MAP = np.full(skeleton_ckt.shape, np.nan)
    node_count = 0 # this is basically the assigned node number


    for comp_bbox_row in comp_bbox:
    # comp_bbox_row =  [13.0, 0.228522, 0.496144, 0.0996564, 0.267352, 1]

        comp_class = comp_bbox_row[0]
        if(comp_class  in NON_ELECTRICAL_COMPS): continue
        
        comp_orient = comp_bbox_row[5]
        bbox =  comp_bbox_row[1:5]

        list_to_append = [] # a row of the component array
        list_to_append.append(comp_class)
        nodes_to_append = [] # a list containing the nodes of a component

        img_h, img_w = skeleton_ckt.shape
        top_edge, left_edge, right_edge, bottom_edge = get_bounding_box_edge_pixels(bbox, img_w, img_h)

        for edge_index, edge in enumerate([top_edge, right_edge, bottom_edge, left_edge]):
            # edge_index = 0 => top edge, edge_index = 1 => right edge and so on .. 
            for pixel in edge:
                if(edge_index == 0):
                    # top_edge
                    next_row = pixel[1] - 1
                    next_col = pixel[0] 
                    next_pixel_val = skeleton_ckt[next_row, next_col]
                elif(edge_index == 1):
                    #right edge
                    next_row = pixel[1] 
                    next_col = pixel[0] + 1
                    next_pixel_val = skeleton_ckt[next_row, next_col]
                elif(edge_index == 2):
                    # bottom edge
                    next_row = pixel[1] + 1 
                    next_col = pixel[0]
                    next_pixel_val = skeleton_ckt[next_row, next_col]
                else:
                    # left edge
                    next_row = pixel[1] 
                    next_col = pixel[0] - 1
                    next_pixel_val = skeleton_ckt[next_row, next_col]



                if(next_pixel_val == 1):
                    # add a new node to the component array
                    nodes_to_append.append(node_count)
                    
                    # add the node into the node map
                    print(next_row, next_col) # registering this pixel in the node map
                    NODE_MAP[next_row, next_col] = node_count # node_count is basically the node number assigned to a node

                    # increment the nodecount 
                    node_count = node_count + 1

        list_to_append.append(nodes_to_append) # adding the nodes list for the component into a row
        list_to_append.append(comp_orient)
        
        COMPONENTS.append(list_to_append) # appending the row for the component to the COMPONENTS matrix

    # for row in COMPONENTS:
    #     print(row)

    return COMPONENTS
    