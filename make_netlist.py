from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from engineering_notation import EngNumber

def make_netlist(components):
    '''
    given the array components it returns  a pyspice circuit object.

    each component should have the following necessary informations:
    comp class
    nodes connected to
    comp orientation

    the expected format of the components array:

    '''

    name_class_map = {'capacitor_unpolarized': 0, 'inductor': 1, 'resistor': 2, 'vdc': 3}
    name_class_map = list(name_class_map.keys())  

    circuit = Circuit('Captured Circuit from Image')

    r_count = 0 # will keep count of how many resistors have been added to the ckt
    c_up_count = 0
    l_count = 0
    vdc_count = 0


    for component in components:
        comp_class = component[0]
        comp_name = name_class_map[comp_class]
        
        nodes_connected = component[1]
        nodes_connected = [circuit.gnd if node == 0 else node for node in nodes_connected]
        
        comp_orient = component[2]
        # print(f'component class: {comp_class}, nodes connected to: {nodes_connected}, orientation: {comp_orient}')

        # TODO: using a default value for all components, gotta find a way to extract value
        if(comp_name == 'resistor'):
            # print("adding resistor to ckt")
            r_count = r_count + 1
            
            circuit.R( f"{r_count}", nodes_connected[0], nodes_connected[1], 1 @u_kΩ) 

        elif(comp_name == 'capacitor_unpolarized'):
            # print("adding capactor_unpolarized to ckt")
            c_up_count= c_up_count + 1

            circuit.C(f"{c_up_count}", nodes_connected[0], nodes_connected[1], 1@u_uF)


        elif(comp_name == 'inductor'):
            print("adding inductor to ckt")
            l_count = l_count + 1
            circuit.L(f"{l_count}", nodes_connected[0], nodes_connected[1], 1@u_mH)

        elif(comp_name == 'vdc'):
            print("adding vdc to ckt")
            vdc_count = vdc_count + 1
            
            # TODO: Add polarity handling here

            circuit.V(f"{vdc_count}",  nodes_connected[0], nodes_connected[1], 10@u_V)

        else:
            print(f"component '{comp_name}' not yet implemented")


    # print("------ netlist of the circuit: ")
    # print(circuit)
    return circuit
