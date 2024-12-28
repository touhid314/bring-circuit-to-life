'''
file to contain various utility functions and algorithms
'''
import numpy as np
import cv2, math
from PIL import Image



############### IMAGE PREPROCESSING FUNCTIONS #######################
# TODO: improve the image preprocessing
def img_preprocess(img, contrast_factor, sharpness_factor, show_enhanced_img:bool = False):
    '''
    args:
        img - a PIL ckt image
        show_enhanced_img - if True plot the enhanced image
    returns:
        img_enhanced - enhanced PIL Image

    '''
    from PIL import ImageEnhance

    # contrast_factor = 2
    # sharpness_factor = 1

    # Resize the image while maintaining the aspect ratio
    original_width, original_height = img.size
    aspect_ratio = original_height / original_width
    new_width = 600
    new_height = int(new_width * aspect_ratio)
    img = img.resize((new_width, new_height), Image.LANCZOS)


    # increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img_enhanced = enhancer.enhance(contrast_factor)

    # Increase sharpness 
    sharpness_enhancer = ImageEnhance.Sharpness(img_enhanced) 
    img_enhanced = sharpness_enhancer.enhance(sharpness_factor)    

    if show_enhanced_img:
        img_enhanced.show()

    return img_enhanced



############### SKELETONIZE IMAGE ###################################
def skeletonize_ckt(image: Image.Image, kernel_size:int, show_skeleton_ckt: bool) -> np.ndarray:
    '''
    arg: 
        image - a PIL image of the ckt in L mode

    returns:
        skeleton - skeleton image of the ckt in numpy format
    '''
    from skimage.morphology import skeletonize

    image_array = np.array(image)  # Convert to NumPy array

    # Apply Canny edge detection
    edges = cv2.Canny(image_array, 100, 200)  # Adjust thresholds as needed

    # Apply dilation and closing to fill small gaps
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Adjust the kernel size if needed
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)

    # Create a new RGBA image to store the final output
    segmented_image = Image.new('L', image.size, (0))

    # Loop through the processed edges and set the pixels to white where edges are detected
    for x in range(image.width):
        for y in range(image.height):
            if edges_closed[y, x] > 0:  # If there is an edge
                segmented_image.putpixel((x, y), (255))  # White with full opacity

    # Convert the PIL image to a NumPy array
    segmented_ckt = np.array(segmented_image)
    segmented_ckt = (segmented_ckt > 0).astype(np.uint8)

    # Ensure the image is binary (0 or 1 values)
    binary_image = (segmented_ckt > 0).astype(np.uint8)

    # Perform skeletonization
    skeleton = skeletonize(binary_image).astype(np.uint8)

    # Scale back to 0-255 for visualization
    skeleton = skeleton * 255
    skeleton = (skeleton > 0).astype(np.uint8)

    if show_skeleton_ckt:         
        pil_image = Image.fromarray(skeleton * 255) 
        pil_image.show()

    return skeleton



############## INITIAL NODE ASSIGNMENT TO COMPONENTS ################  
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

def get_COMPONENTS(skeleton_ckt, comp_bbox):
    '''
    algorithm to find out the connections between the components and assign nodes.
    this algorithm does not find the connections between components.
    
    args: 


    return:
        COMPONENTS -  
        NODE_MAP - 
    '''
    # in pil: pixel => (x, y) 
    # in numpy: pixel => (y, x) y= row value, x = column value

    from simulate import NON_ELECTRICAL_COMPS

    # define all data structrues
    COMPONENTS = []
    all_start_points = []


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
                    # print(next_row, next_col) # registering this pixel in the node map
                    NODE_MAP[next_row, next_col] = node_count # node_count is basically the node number assigned to a node
                    all_start_points.append([next_row, next_col])

                    # increment the nodecount 
                    node_count = node_count + 1

        list_to_append.append(nodes_to_append) # adding the nodes list for the component into a row
        list_to_append.append(comp_orient)
        
        COMPONENTS.append(list_to_append) # appending the row for the component to the COMPONENTS matrix

    # for row in COMPONENTS:
    #     print(row)

    return COMPONENTS, NODE_MAP, all_start_points



######## VIRAL SPREAD ALGORITHM - A RECURSIVE ALGORITHM TO FIND NODAL CONNECTION BETWEEN COMPONENTS IN A CKT IMAGE #######
# TODO: further optimize the algorithm
 
# helper functions
def remove_duplicates(array):
    for row in array:
        # if the duplicate 
        
        # Convert the second element of each row to a set to remove duplicates, then back to a list
        row[1] = list(set(row[1]))
        # Optionally sort the list to maintain order
        # row[1].sort(reverse=True)  # Use reverse=False for ascending order if needed
        # the order of nodes, cannot be changed, this will harm the component polarity detection
    return array

def modify_list(list1, list2):
    """
    Modifies list1 by removing subsets from the second element of its rows, 
    if the subsets are present as complete rows in list2.
    Also updates list2 by deleting the rows that caused modifications in list1.

    Args:
        list1 (list): The first 2D array-like list to be modified.
        list2 (list): The second 2D array-like list to compare against.

    Returns:
        tuple: Modified version of list1 and list2.
    """
    # Convert list2 elements to sets for easier comparison
    list2_sets = [set(map(float, row)) for row in list2]

    # Rows to remove from list2
    rows_to_remove = []

    # Process list1
    for row in list1:
        second_element = row[1]  # The list we need to modify
        to_remove = []  # Pairs to remove

        # Check all pairs in the second element
        for i in range(len(second_element) - 1):
            for j in range(i + 1, len(second_element)):
                pair = set(second_element[i:j + 1])  # Form a subset
                if pair in list2_sets:  # Check if it's a row in list2
                    to_remove.extend(pair)
                    # Track the row index in list2
                    rows_to_remove.append(list2_sets.index(pair))

        # Remove the identified elements from list1
        row[1] = [val for val in second_element if val not in to_remove]

    # Remove the corresponding rows from list2
    rows_to_remove = sorted(set(rows_to_remove), reverse=True)  # Remove duplicates and reverse for safe deletion
    for idx in rows_to_remove:
        list2.pop(idx)

    return list1, list2

def update_nodes(all_connected_nodes: list, NODE_MAP:np.ndarray, COMPONENTS:np.ndarray) -> None:
    flattened_list = [element for row in all_connected_nodes for element in row]
    row_numbers = [row_index for row_index, row in enumerate(all_connected_nodes) for _ in row]

    #  modify the componenets matrix
    for i, row in enumerate(COMPONENTS):
        tmp = row[1] # the 2nd element of the row holds the list of nodes comp is connected to
        for j, node in enumerate(tmp):
            # going through each node
            if node in flattened_list: 
                for row_index, connected_nodes in enumerate(all_connected_nodes):
                    if node in connected_nodes:
                        COMPONENTS[i][1][j] = row_index
                        break
            else:
                COMPONENTS[i][1][j] = None
        
        # # now remove the None nodes
        COMPONENTS[i][1] = [x for x in COMPONENTS[i][1] if x is not None]


    remove_duplicates(COMPONENTS) # removes duplicate nodes in the node list, [1, [2,2,1]] => [1, [2,1]]

    # # modify the NODE_MAP matrix

    # # modify the NODE_MAP matrix
    # import math
    # for i in range(NODE_MAP.shape[0]):
    #     for j in range(NODE_MAP.shape[1]):
    #         # updating the NODE_MAP matrix for position [i,j]
    #         if (math.isnan(NODE_MAP[i, j] == False) and (NODE_MAP[i,j] != -1)):
    #             if NODE_MAP[i, j] in flattened_list:
    #                 # for row_index, connected_nodes in enumerate(all_connected_nodes):
    #                 #     if NODE_MAP[i, j] in connected_nodes:
    #                 #         NODE_MAP[i, j] = row_index
    #                 a = NODE_MAP[i, j]
    #                 NODE_MAP[i, j] = row_numbers[flattened_list.index(a)]

    # Modify the NODE_MAP matrix
    import math
    for i in range(NODE_MAP.shape[0]):
        for j in range(NODE_MAP.shape[1]):
            # updating the NODE_MAP matrix for position [i,j]
            if not math.isnan(NODE_MAP[i, j]) and NODE_MAP[i, j] != -1:
                if NODE_MAP[i, j] in flattened_list:
                    a = NODE_MAP[i, j]
                    NODE_MAP[i, j] = row_numbers[flattened_list.index(a)]

# main algo
connected_nodes = []
def viral_spread(x: int, y: int, NODE_MAP: np.ndarray, binary_img: np.ndarray, all_connected_nodes: list):
    """
    Spreads the value of the main node to neighboring points based on specific conditions.

    Parameters:
        x (int): X-coordinate of the current node.
        y (int): Y-coordinate of the current node.
        NODE_MAP: 
        binary_img: 
        all_connected_nodes:

    returns:
        nothing, updates the NODE_MAP, all_connected_ndoes in place
    
    """


    viral_node_num = NODE_MAP[x, y].item()
    viral_pixel_val = binary_img[x, y].item()

    global connected_nodes
    if len(connected_nodes) == 0: connected_nodes.append(viral_node_num)

    if (viral_pixel_val != 0 and viral_node_num >= 0): # attack only if the viral point satisfies these conditions, otherwise do nothing
        for i in [-1, 0, 1]: # generating attack point coordinates
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue  # Skip the current node itself        
                
                attack_point_x = x + i
                attack_point_y = y + j

                attacked_pixel_val = binary_img[attack_point_x, attack_point_y].item()
                attacked_node_num = NODE_MAP[attack_point_x, attack_point_y].item()

                if (attacked_node_num == -1):
                    # attacked_pixel_val == 0 and attacked_node_num == -1
                    # attacked_pixel_val == 1 and attacked_node_num == -1
                    continue
                elif (attacked_pixel_val == 0):
                    # attacked_pixel_val == 0 and attacked_node_num == nan
                    # attacked_pixel_val == 0 and attacked_node_num == [0, inf)
                    NODE_MAP[attack_point_x, attack_point_y] = -1

                elif (attacked_pixel_val == 1 and math.isnan(attacked_node_num)):
                    # attacked_pixel_val == 1 and attacked_node_num == nan
                    NODE_MAP[attack_point_x, attack_point_y] = viral_node_num
                    viral_spread(attack_point_x, attack_point_y, NODE_MAP, binary_img, all_connected_nodes)
                
                elif (attacked_pixel_val == 1 and attacked_node_num >= 0):
                    # attacked_pixel_val == 1 and attacked_node_num == [0, inf)
                    if (viral_node_num < attacked_node_num): 
                        NODE_MAP[attack_point_x, attack_point_y] = viral_node_num
                        # this means a connection has been found
                        connected_nodes.append(attacked_node_num)

                        viral_spread(attack_point_x, attack_point_y, NODE_MAP, binary_img, all_connected_nodes)

                    else:
                        continue

def reduce_nodes(skeleton_ckt: np.ndarray, comp_bbox: list[list[float]], NODE_MAP: np.ndarray, COMPONENTS: np.ndarray, all_start_points: list):
    '''
    algorithm to find the connections between components and hence reduce the node counts in terms of connection.
    
    args: 

    returns: 
        none.
        modifies the NODE_MAP and COMPONENTS array in place as they are passed by reference by default
    '''
    from simulate import NON_ELECTRICAL_COMPS

    # remove all electrical COMPONENTS from the skeleton and find the contours

    # remove all elec COMPONENTS from skeleton
    skeleton_ckt_stripped = skeleton_ckt.copy() # a 2d numpy array which will have the elec COMPONENTS stripped off 

    # Loop through each bounding box
    for box in comp_bbox:

        comp_class = box[0]
        
        # Check if the component class is NOT in NON_ELECTRICAL_COMPS
        if comp_class not in NON_ELECTRICAL_COMPS:
            _, x, y, w, h, _ = box

            # Convert center-based bbox format (x, y, w, h) to pixel coordinates
            center_x = int(x * skeleton_ckt.shape[1])  # Convert normalized to pixel position
            center_y = int(y * skeleton_ckt.shape[0])
            width = int(w * skeleton_ckt.shape[1])  # Convert normalized width to pixel width
            height = int(h * skeleton_ckt.shape[0])

            # Calculate the top-left and bottom-right coordinates of the box
            x1 = max(0, center_x - width // 2)
            y1 = max(0, center_y - height // 2)
            x2 = min(skeleton_ckt.shape[1], center_x + width // 2)
            y2 = min(skeleton_ckt.shape[0], center_y + height // 2)

            # Mask the region with black pixels (value 0)
            skeleton_ckt_stripped[y1:y2, x1:x2] = 0

    # plt.imshow(skeleton_ckt_stripped) 
    # # find all contours on the strippped skeleton ckt
    # all_contours, _  = cv2.findContours(skeleton_ckt_stripped, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # import math, pprint

    # all_connected_nodes = []
    # node_num = 0

    # for contour in all_contours:
    #     connected_nodes = []

    #     for pixel in contour:
    #         pixel=pixel[0]
    #         pixel_node = NODE_MAP[pixel[1], pixel[0]] #what node does this pixel belong to
            
    #         if(not math.isnan(pixel_node)):
    #             # if inside this block, it means, the pixel belongs to a node, so the whole contour belongs to that node
    #             # print(pixel_node)
    #             if (pixel_node not in connected_nodes):
    #                 # if the node has not been added into the list, then add
    #                 connected_nodes.append(int(pixel_node.item()))
        
    #     if(len(connected_nodes) > 1):
    #         # if a contour does not connect to any node or only connects to one node reject that node
    #         all_connected_nodes.append(connected_nodes)

    #         node_num = node_num + 1
    #         # TODO: register the pixels of the contour with their new node number
            

    # modified code
    global connected_nodes
    all_connected_nodes = []

    import sys
    sys.setrecursionlimit(50000)

    # pil_image = Image.fromarray(skeleton_ckt_stripped * 255)  # Convert binary (0, 1) to grayscale (0, 255)
    # pil_image.show()

    for start_point in all_start_points:
        connected_nodes = []

        viral_spread(start_point[0], start_point[1], NODE_MAP, skeleton_ckt_stripped, all_connected_nodes)
        
        if (len(connected_nodes)>1): all_connected_nodes.append(connected_nodes)

    # reducing nodes
    modify_list(COMPONENTS, all_connected_nodes)
    update_nodes(all_connected_nodes, NODE_MAP, COMPONENTS)





############################ trash functions #######################
def is_same_wire(pix1: tuple, pix2: tuple, ckt_img) -> bool:
    '''
    input: 2 tuples representing 2 pixels
            ckt_img - grayscale PIL image of the ckt
    output: true - the two pixels are actually two sides of the same wire
    ''' 
    print(f'is_same_wire function: {pix1}, {pix2}')
    line_pixels = bresenham_line(pix1[0], pix1[1], pix2[0], pix2[1])
    print(f'line_pixels: {line_pixels} ')
    
    total_color_value = 0 
    for pixel in line_pixels:
        total_color_value = total_color_value + ckt_img.getpixel(pixel)

    avg_color_value = total_color_value/len(line_pixels)
    print(f"avg_color_value between the pixels: {avg_color_value}")
    

    img_arr = np.array(ckt_img)
    img_avg_color = img_arr.mean()
    print('img avg color: ', img_avg_color)
    # thinking that, avg color of the img = the bg color of the img
    # TODO: need to improve here in determining what is bg color
    
    threshold = 50 

    if(abs(img_avg_color - avg_color_value) > threshold):
        # the two pixels belong to the same wire
        return True
    else:
        return False
    
    # example use 
    # is_same_wire((20,30), (30,40), ckt_img_enhanced)

    # necessary functions


    # # Example usage
    # array = [
    #     [0, [2, 2, 2, 1]],
    #     [2, [2, 2, 1]],
    #     [2, [2, 0, 0]],
    #     [3, [1, 0, 0]]
    # ]

    # # Call the function
    # result = remove_duplicates(array)
    # print(result)





