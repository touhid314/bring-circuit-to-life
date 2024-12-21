'''
contains the main simulation functionality
'''

from PIL import Image
from pprint import pprint

# * GLOBAL VARIABLES
NON_ELECTRICAL_COMPS = [-1, -2] #classes for COMPONENTS like crossover, junction etc



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
    ckt_img_enhanced = img_preprocess(ckt_img, contrast_factor=3, sharpness_factor=1)
    skeleton_ckt = skeletonize_ckt(ckt_img_enhanced, kernel_size=5,show_skeleton_ckt=True)

    
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
    circuit = make_netlist(COMPONENTS) # from the connection described in the COMPONENTS array, get the ckt netlist

    # TODO: this is where LLM comes into play after the circuit object is made
    from analyse import analyse
    comp_voltages = analyse(circuit, COMPONENTS)

    end_time = time.time()  # Record the end time
    print(f">>>>>> Execution time for simulation: {end_time - start_time:.4f} seconds")

    ekdom_sesh = time.time()
    print(f">>>>>> total execution time: {ekdom_sesh - ekdom_start:.4f} seconds")

    return electrical_component_bbox, comp_voltages


if __name__ == "__main__":
    # path = r"ckt5.jpg"
    # path = r"C:\Users\Touhid2\Desktop\50_jpg.rf.dfa9222529f42fb211b7fd65119dddf3.jpg"
    # path = r"ckt1.jpeg"
    path = r"H:\NOTHING\#Projects\bring_ckt_to_life_project\code\test_imgs\20220729_111338_jpg.rf.2b39886379e2c93aa69a30966e0126e3.jpg"

    bbox, volts = simulate_from_img(path)

    print("----------simulation result:", bbox, volts)



     