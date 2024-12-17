'''
contains the main simulation functionality
'''

import cv2
import numpy as np
from skimage.morphology import skeletonize

def skeletonize_image(edge_image):
    """
    Computes the skeleton of a binary edge image.

    Args:
        edge_image (ndarray): Binary edge-detected image (output of Canny edge detection).

    Returns:
        ndarray: Skeletonized image with a single-pixel-wide skeleton.
    """
    # Ensure the image is binary (0 or 1 values)
    binary_image = (edge_image > 0).astype(np.uint8)

    # Perform skeletonization
    skeleton = skeletonize(binary_image).astype(np.uint8)

    # Scale back to 0-255 for visualization
    skeleton = skeleton * 255

    return skeleton


from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


def segment_high_contrast_portion(image: Image.Image):
    '''
    arg: 
        image - a PIL image of the ckt

    returns:
       segmented_ckt
        skeleton - skeleton image of the ckt in numpy format
    '''

    # Load the image and convert it to grayscale
    # image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_array = np.array(image)  # Convert to NumPy array

    # Apply Canny edge detection
    edges = cv2.Canny(image_array, 100, 200)  # Adjust thresholds as needed

    # Apply dilation and closing to fill small gaps
    kernel = np.ones((7, 7), np.uint8)  # Adjust the kernel size if needed
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

    # find skeleton
    skeleton = skeletonize_image(segmented_ckt)
    skeleton = (skeleton > 0).astype(np.uint8)

    return segmented_ckt, skeleton


import numpy as np
from PIL import Image, ImageDraw


def bresenham_line(x1: int, y1: int, x2: int, y2: int):
    """
    Generate the coordinates of pixels in a straight line between two points.
    
    Args:
    - x1, y1: Coordinates of the first pixel.
    - x2, y2: Coordinates of the second pixel.
    
    Returns:
    - A list of (x, y) tuples representing the coordinates of pixels on the line.
    """
    pixels = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        pixels.append((x1, y1))  # Add the current pixel to the list
        if x1 == x2 and y1 == y2:  # If reached the endpoint
            break
        err2 = err * 2
        if err2 > -dy:  # Move in x direction
            err -= dy
            x1 += sx
        if err2 < dx:  # Move in y direction
            err += dx
            y1 += sy
            
    return pixels

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
def remove_duplicates(array):
    for row in array:
        # Convert the second element of each row to a set to remove duplicates, then back to a list
        row[1] = list(set(row[1]))
        # Optionally sort the list to maintain order
        row[1].sort(reverse=True)  # Use reverse=False for ascending order if needed
    return array

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


def reduce_nodes(all_connected_nodes: list, NODE_MAP, COMPONENTS):
    #  modify the componenets matrix
    for i, row in enumerate(COMPONENTS):
        tmp = row[1] # the 2nd element of the row holds the list of nodes comp is connected to
        for j, node in enumerate(tmp):
            # going through each node
            for row_index, connected_nodes in enumerate(all_connected_nodes):
                if node in connected_nodes:
                    COMPONENTS[i][1][j] = row_index

    remove_duplicates(COMPONENTS) # removes duplicate nodes in the node list, [1, [2,2,1]] => [1, [2,1]]

    # modify the NODE_MAP matrix

    already_updated = []    
    for i in range(NODE_MAP.shape[0]):
        for j in range(NODE_MAP.shape[1]):
            for row_index, connected_nodes in enumerate(all_connected_nodes):
                if ([i,j] not in already_updated) and (NODE_MAP[i, j] in connected_nodes):
                        NODE_MAP[i, j] = row_index
                        already_updated.append([i, j])

    
def img_preprocess(ckt_img):
    '''
    args:
    ckt_img - a PIL ckt image
    
    returns:
    the skeleton of the ckt in numpy format 
    '''
    enhancer = ImageEnhance.Contrast(ckt_img)
    contrast_factor = 2
    ckt_img_enhanced = enhancer.enhance(contrast_factor)
    # ckt_img_enhanced.show()

    segmented_ckt, skeleton_ckt = segment_high_contrast_portion(ckt_img)
    
    # optionally plot the image
    # print(segmented_array)  # Print the NumPy array representation of the segmented image

    # Plot the segmented image
    # plt.imshow(segmented_ckt, cmap='gray')
    # plt.axis('off')  # Turn off axis labels
    # plt.show()

    # plt.figure(figsize=(50, 50))
    # plt.imshow(skeleton_ckt, cmap='gray')
    # plt.axis('off')
    # plt.show()

    return skeleton_ckt

#########################################################################################
from PIL import ImageEnhance
from get_comp_class_bbox_orient import get_comp_bbox_class_orient

# * GLOBAL VARIABLES
NON_ELECTRICAL_COMPS = [-1, -2] #classes for COMPONENTS like crossover, junction etc



def simulate_from_img(path: str):
    '''
    given the path to an image, it returns ??
    '''
    global NON_ELECTRICAL_COMPS

    # * image preprocessing
    ckt_img = Image.open(path).convert('L')  # Convert to grayscale
    
    skeleton_ckt = img_preprocess(ckt_img)

    # * get component bounding box
    comp_bbox = get_comp_bbox_class_orient(path)
    electrical_component_bbox = comp_bbox.copy()

    ## only keeping the electrical COMPONENTS in the ckt skeleton image
    for index, row in enumerate(electrical_component_bbox):
        if row[0] in NON_ELECTRICAL_COMPS:
            del electrical_component_bbox[index] 


    ############################################################# finding nodes
    # * finding nodes
    COMPONENTS = []
    NODE_MAP = np.full(skeleton_ckt.shape, np.nan)

    from utils import get_COMPONENTS
    COMPONENTS = get_COMPONENTS(skeleton_ckt, comp_bbox)
    print(COMPONENTS)

    ###################################################### reducing nodes
    # remove all electrical COMPONENTS from the skeleton and find the contours

    # remove all elec COMPONENTS from skeleton
    skeleton_ckt_stripped = skeleton_ckt.copy() # a 2d numpy array which will have the elec COMPONENTS stripped off 
    comp_bbox = comp_bbox # bboxes and comp class in normalized pixel values


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
    # find all contours on the strippped skeleton ckt
    all_contours, _  = cv2.findContours(skeleton_ckt_stripped, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    import math, pprint

    all_connected_nodes = []

    for contour in all_contours:
        connected_nodes = []

        for pixel in contour:
            pixel=pixel[0]
            pixel_node = NODE_MAP[pixel[1], pixel[0]] #what node does this pixel belong to
            
            if(not math.isnan(pixel_node)):
                # the pixel belongs to a node, so the whole contour belongs to that node
                # print(pixel_node)
                if (pixel_node not in connected_nodes):
                    # if the node has not been added into the list, then add
                    connected_nodes.append(int(pixel_node.item()))
        
        if(len(connected_nodes) > 1):
            # if a contour does not connect to any node or only connects to one node reject that node
            all_connected_nodes.append(connected_nodes)
        

            # register the pixels of the contour with their new node number
    print(f"all_connected_nodes: {pprint.pprint(all_connected_nodes)}")
    print("so, all nodes in row 3 of connected nodes will become node 3")

    # reducing nodes
    reduce_nodes(all_connected_nodes, NODE_MAP, COMPONENTS)
    
    ######################################################## make ckt and simulate
    from make_netlist import make_netlist
    circuit = make_netlist(COMPONENTS) # from the connection described in the COMPONENTS array, get the ckt netlist

    # TODO: this is where LLM comes into play after the circuit object is made
    from analyse import analyse
    comp_voltages = analyse(circuit, COMPONENTS)

    return electrical_component_bbox, comp_voltages


if __name__ == "__main__":
    path = r"ckt1.jpeg"
    bbox, volts = simulate_from_img(path)

    print("----------simulation result:", bbox, volts)
     