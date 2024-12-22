from PySpice.Spice.Netlist import Circuit
import numpy
from PySpice.Unit import *

def get_comp_voltages(COMPONENTS, node_voltages):
    voltages = []
    for index, component in enumerate(COMPONENTS):     
        comp_class = component[0]
        nodes = component[1]

        t = []
        t.append(comp_class)
        
        volt = abs(node_voltages[nodes[1]] - node_voltages[nodes[0]]) #stores the absolute value of voltage difference across the component
        t.append(volt)

        voltages.append(t)
    
    return voltages


def analyse(circuit: Circuit, COMPONENTS:numpy.ndarray):
    '''
    given a pyspice circuit object, this function can run various types of simulation and analysis on it.
    
    args:
        circuit - 

    returns: 
        ??
    '''
    
    # create pyspsice simulator object
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    
    # perform a specific type of simulation 
    analysis = simulator.operating_point()

    # extract desired information from the analysis 
    for node in analysis.nodes.values():
        print('Node {}: {:4.1f} V'.format(str(node), float(node)))

    '''
    node_voltages[2] => voltage at node 2 of the circuit
    '''
    node_voltages = [0]


    for node in analysis.nodes.values():
        node_voltages.insert(int(str(node)), float(node))
    
    comp_voltages = get_comp_voltages(COMPONENTS, node_voltages)
    return comp_voltages

if __name__ == "__main__":

    # Create a new circuit
    circuit = Circuit('Captured Circuit from Image')

    # Add components to the circuit
    circuit.V(1, '1', '0', 10@u_V)
    circuit.C(1, '0', '2', 1@u_uF)
    circuit.R(1, '0', '3', 1@u_kΩ)
    circuit.V(2, '4', '5', 10@u_V)
    circuit.R(2, '0', '6', 1@u_kΩ)
    circuit.L(1, '8', '7', 1@u_mH)
    circuit.R(3, '2', '3', 1@u_kΩ)
    circuit.L(2, '0', '3', 1@u_mH)
    circuit.R(4, '4', '6', 1@u_kΩ)
    circuit.R(5, '8', '5', 1@u_kΩ)
    circuit.R(6, '1', '5', 1@u_kΩ)
    circuit.R(7, '3', '7', 1@u_kΩ)

    # Print the netlist
    print(str(circuit))