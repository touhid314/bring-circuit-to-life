'''
contains the main simulation functionality
'''

from PIL import Image
from pprint import pprint

# * GLOBAL VARIABLES
NON_ELECTRICAL_COMPS = [-1, -2] #classes for COMPONENTS like crossover, junction etc

from PIL import Image
import numpy as np

def process_and_show_node_map(node_map: np.ndarray, ckt_img: Image.Image):
    """
    Processes a 2D numpy array by replacing NaN with 0,
    assigns a specific color to non-negative values,
    applies a dilation kernel to make nodes fatter,
    overlays the node map on the circuit image,
    and displays the result as a combined image.

    Parameters:
        node_map (numpy.ndarray): Input 2D numpy array.
        ckt_img (PIL.Image.Image): Circuit image to overlay the node map onto.
    """
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    from scipy.ndimage import binary_dilation

    # Create a copy to avoid modifying the original array
    processed_map = np.nan_to_num(node_map, nan=-1)  # Replace NaN with -1

    # Create an RGB map
    height, width = processed_map.shape
    colored_map = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign green color to non-negative values
    non_negative_mask = processed_map >= 0  # Non-negative values
    dilation_kernel_size = 15
    dilated_mask = binary_dilation(non_negative_mask, structure=np.ones((dilation_kernel_size, dilation_kernel_size)))  # Apply dilation
    colored_map[dilated_mask] = (0, 255, 0)  # Green for non-negative values

    # Convert to a PIL image
    node_map_img = Image.fromarray(colored_map)

    # Annotate node names
    draw = ImageDraw.Draw(node_map_img)
    try:
        font = ImageFont.truetype("arial.ttf", size=70)  # Load a font
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if "arial.ttf" is not found

    unique_values = np.unique(processed_map)
    for value in unique_values:
        if value >= 0:
            y, x = np.argwhere(processed_map == value)[0]  # Get the first occurrence of the node
            draw.text((x+10, y+10), str(int(value)), fill="white", font=font)

    # Resize the node map to match the circuit image size
    node_map_img_resized = node_map_img.resize(ckt_img.size, resample=Image.BILINEAR)

    # Combine the two images
    combined_img = ckt_img.convert("RGBA").copy()
    node_map_img_resized = node_map_img_resized.convert("RGBA")

    # Blend the images (node map overlay is semi-transparent)
    combined_img = Image.blend(combined_img, node_map_img_resized, alpha=0.3)
    

    # Display the combined image
    # combined_img.show()

    return combined_img 



def simulate_from_img(path: str):
    '''
    given the path to an image, it returns ??
    '''
    global NON_ELECTRICAL_COMPS
    import time
    ekdom_start = time.time()

    start_time = time.time()  # Record the start time

    ckt_img = Image.open(path).convert('L')  # Convert to grayscale
    end_time = time.time()  # Record the end time
    print(f">>>>>> Execution time for image open : {end_time - start_time:.4f} seconds")

    # * skeletonize the ckt
    start_time = time.time()  # Record the start time

    from my_utils import img_preprocess, skeletonize_ckt
    ckt_img_enhanced = img_preprocess(ckt_img, contrast_factor=3, sharpness_factor=1, show_enhanced_img=False)
    skeleton_ckt = skeletonize_ckt(ckt_img_enhanced, kernel_size=7,show_skeleton_ckt=False)

    
    end_time = time.time()  # Record the end time
    print(f">>>>>> Execution time for skeletonize : {end_time - start_time:.4f} seconds")

    # * get component bounding box
    start_time = time.time()  # Record the start time
    from get_comp_class_bbox_orient import get_comp_bbox_class_orient
    comp_bbox = get_comp_bbox_class_orient(path)

    end_time = time.time()  # Record the end time
    print(f">>>>>> Execution time for all models: {end_time - start_time:.4f} seconds")

    ## only keeping the electrical COMPONENTS in the ckt skeleton image
    electrical_component_bbox = comp_bbox.copy()

    for index, row in enumerate(electrical_component_bbox):
        if row[0] in NON_ELECTRICAL_COMPS:
            electrical_component_bbox[index] = None

    electrical_component_bbox = [x for x in electrical_component_bbox if x!=None]
    


    # * assigning nodes to components
    start_time = time.time()  # Record the start time

    from my_utils import get_COMPONENTS
    COMPONENTS, NODE_MAP, all_start_points = get_COMPONENTS(skeleton_ckt, comp_bbox)


    # * finding connection between components and reducing node counts
    from my_utils import reduce_nodes
    reduce_nodes(skeleton_ckt, comp_bbox, NODE_MAP, COMPONENTS, all_start_points)
    
    end_time = time.time()  # Record the end time
    print(f">>>>>> Execution time for nodal algos: {end_time - start_time:.4f} seconds")

    # * make ckt and simulate
    start_time = time.time()  # Record the start time

    from make_netlist import make_netlist
    circuit = make_netlist(COMPONENTS) # from the connection described in the COMPONENTS array, get the circuit object

    # TODO: this is where LLM comes into play after the circuit object is made
    from analyse import Analyzer
    analyzer = Analyzer(circuit)
    comp_voltages = analyzer.get_comp_voltages(COMPONENTS)

    end_time = time.time()  # Record the end time
    print(f">>>>>> Execution time for simulation: {end_time - start_time:.4f} seconds")

    ekdom_sesh = time.time()
    print(f">>>>>> total execution time: {ekdom_sesh - ekdom_start:.4f} seconds")

    combined_img = process_and_show_node_map(NODE_MAP, ckt_img)

    return electrical_component_bbox, comp_voltages, NODE_MAP, combined_img


if __name__ == "__main__":
    # path = r"ckt5.jpg"
    # path = r"C:\Users\Touhid2\Desktop\50_jpg.rf.dfa9222529f42fb211b7fd65119dddf3.jpg"
    # path = r"ckt1.jpeg"
    path = r"ckt6.jpg"
    bbox, volts, _ , _ = simulate_from_img(path)

    print("----------simulation result:", bbox, volts)