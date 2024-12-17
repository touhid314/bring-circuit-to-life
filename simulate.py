'''
contains the main simulation functionality
'''

import numpy as np
from PIL import Image

# * GLOBAL VARIABLES
NON_ELECTRICAL_COMPS = [-1, -2] #classes for COMPONENTS like crossover, junction etc



def simulate_from_img(path: str):
    '''
    given the path to an image, it returns ??
    '''
    global NON_ELECTRICAL_COMPS

    ckt_img = Image.open(path).convert('L')  # Convert to grayscale

    # * skeletonize the ckt
    from utils import img_preprocess, skeletonize_ckt
    ckt_img_enhanced = img_preprocess(ckt_img)
    skeleton_ckt = skeletonize_ckt(ckt_img_enhanced)

    # * get component bounding box
    from get_comp_class_bbox_orient import get_comp_bbox_class_orient
    comp_bbox = get_comp_bbox_class_orient(path)
    electrical_component_bbox = comp_bbox.copy()

    ## only keeping the electrical COMPONENTS in the ckt skeleton image
    for index, row in enumerate(electrical_component_bbox):
        if row[0] in NON_ELECTRICAL_COMPS:
            del electrical_component_bbox[index] 


    # * assigning nodes to components
    from utils import get_COMPONENTS
    COMPONENTS, NODE_MAP = get_COMPONENTS(skeleton_ckt, comp_bbox)


    # * finding connection between components and reducing node counts
    from utils import reduce_nodes
    reduce_nodes(skeleton_ckt, comp_bbox, NODE_MAP, COMPONENTS)
    
    # * make ckt and simulate
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



     