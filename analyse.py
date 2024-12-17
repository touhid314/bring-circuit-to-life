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


def analyse(circuit, COMPONENTS):
    '''
    given a pyspice circuit object, this function can run various types of simulation and analysis on it
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