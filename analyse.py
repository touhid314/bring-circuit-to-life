from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *


class Analyzer:
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        self.current_simulation_type = None
        self.analysis = None
        

    ############################### SIMULATION FUNCTIONS ########################################
    # operating_point
    # dc 
    # ac
    # transient
    # noise
    # dc_sweep
    # ac_sweep
    # temperature
    # monte_carlo
    # corner
    # parametric
    # sensitivity
    # transfer_function
    # poles_zeros

    def operating_point(self):
        """
        Performs operating point analysis on the circuit.
        """
        self.current_simulation_type = "operating_point"
        analysis = self.simulator.operating_point()

        self.analysis = analysis
    
    def dc_analysis(self):
        self.current_simulation_type = "dc_analysis"
        print("not implemented yet")
        return

    def transient_analysis(self, initial_conditions: list = None , start_time: float = 0, stop_time: float = 1e-3, step_time: float = 1e-6):
        """
        Performs transient analysis on the circuit.

        Parameters:
            start_time (float): Start time of the transient simulation (in seconds). Default is 0.
            stop_time (float): Stop time of the transient simulation (in seconds). Default is 1e-3.
            step_time (float): Time step for the simulation (in seconds). Default is 1e-6.
            
            initial_condition (list): 
                Initial conditions for the simulation. Default is None.
                [[element_name1, voltage_diff], [element_name2, voltage_diff], ....] where element_name is the name of the element and value is the initial condition value.
                NOTE: pyspice can only set initial condition for voltage nodes, not for current nodes. current node initial conditions are not supported as of now in pyspice

        """
        # TODO: transient analysis params will be remembered and used in re_simulate function, otherwise issues with get_power simulation

        self.current_simulation_type = "transient_analysis"


        if initial_conditions != None:
            use_initial_condition = True
            for condition in initial_conditions:
                element_name = condition[0]
                value = condition[1]
                nodes = self.circuit[element_name].node_names
                node1 = nodes[0]
                node2 = nodes[1]

                self.simulator.initial_condition(**{node1: value, node2: 0})
                # TODO: fixit: if the 2 elements are connected back to back, then voltage difference is not properly set. first element's set voltage difference is altered by the second element's set voltage difference

        else:
            use_initial_condition = False

        analysis = self.simulator.transient(step_time=step_time, end_time=stop_time, start_time=start_time, use_initial_condition=use_initial_condition)

        self.analysis = analysis
        

    def re_simulate(self):
        if self.current_simulation_type == "operating_point":
            self.operating_point()
        elif self.current_simulation_type == "dc_analysis":
            self.dc_analysis()
        elif self.current_simulation_type == "transient_analysis":
            self.transient_analysis()
        else:
            print("No simulation type found")

    ############################### ACTION FUNCTIONS ########################################

    def get_comp_voltages(self, COMPONENTS):
        self.operating_point()
        node_voltages = {str(node): float(node) for node in self.analysis.nodes.values()}

        voltages = []
        for index, component in enumerate(COMPONENTS):     
            comp_class = component[0]
            nodes = component[1]

            t = []
            t.append(comp_class)
            node_voltages['0'] = 0
            volt = abs(node_voltages[str(nodes[1])] - node_voltages[str(nodes[0])]) #stores the absolute value of voltage difference across the component
            t.append(volt)

            voltages.append(t)
        
        return voltages
    
    
    def change_value_of_element(self, element_name:str, value):
        '''
        changes the value of the element in the circuit.
        arguments:
            element_name: name of the element in the circuit. case sensitive
            value: new value of the element
        '''

        element = self.circuit[element_name]

        if element.ALIAS == "R":
            element.resistance = value
        elif element.ALIAS == "C":
            element.capacitance = value
        elif element.ALIAS == "L":
            element.inductance = value
        elif element.ALIAS == "V":
            element.dc_value = value
        else:
            print("Element type not found, element value not changed")

    
    def add_ammeter_with_element(self, circuit, element_name):
        '''Add a dummy voltage source in the circuit to measure current through the element.
        Name of dummy voltage source is V<element_name>_plus.
        so branch current through element of this name is the current through the element.
        '''
        element = self.circuit[element_name] # element name is case sensitive here. element name is as in the netlist
        a = element.plus
        try:
            a.add_current_probe(circuit)
        except NameError:
            print(f"Current probe already added to {element_name}")
    
    def get_voltage(self, nodes_name:list=None, show_plot=False):
        '''
        arguments:
        nodes_name - a list of nodes. if one node is passed, voltage at that node is returned.\\
              if 2 nodes are passed voltage difference between the first and 2nd node is returned.\\
              if more than 2 nodes are passed ValueError is raised.
        show_plot - if true, plots the voltage also. plotting will only work in transient analysis as of now.


        example use:
        get_voltage(nodes_name=['1'], show_plot=False)
        get_voltage(nodes_name=['5', '8'], show_plot=True)
        get_voltage(nodes_name=['5', '8'], show_plot=False)
        

        pyspice does not provide voltage for gnd node in the analysis.nodes dictionary. 
        so, for ground node, check voltage without this function.
        '''

        import matplotlib.pyplot as plt
        from PySpice.Probe.Plot import plot

        if(nodes_name):
            if str(self.circuit.gnd) in nodes_name:
                nodes_name.remove(str(self.circuit.gnd))

            if (len(nodes_name) == 1):
                node_name = nodes_name[0]
                voltage = self.analysis.nodes[node_name.lower()]

                if(voltage.shape[0] > 1 and show_plot):
                    figure = plt.figure(f"Bring Circuit to Life - time vs 'node {node_name}' voltage graph")
                    axe = plt.subplot(111)
                    plot(self.analysis.nodes[node_name], axis=axe)
                    plt.title(f"time vs 'node {node_name}' voltage graph")
                    plt.xlabel('Time [s]')
                    plt.ylabel(f"Voltage [V] at 'node {node_name}'")
                    
                    self.show_plot(plt, axe)

            elif(len(nodes_name) == 2):
                node1 = nodes_name[0]
                node2 = nodes_name[1]


                voltage = (self.analysis.nodes[node1.lower()]) - (self.analysis.nodes[node2.lower()])

                if(voltage.shape[0] > 1 and show_plot):
                    figure = plt.figure(f"Bring Circuit to Life - time vs  'node {node1} - node {node2}' voltage graph")
                    axe = plt.subplot(111)
                    plot(self.analysis.nodes[node1] - self.analysis.nodes[node2], axis=axe)
                    plt.title(f"time vs  'node {node1} - node {node2}' voltage graph")
                    plt.xlabel('Time [s]')
                    plt.ylabel(f"'node {node1} - node {node2}' Voltage [V]")
                    
                    self.show_plot(plt, axe)
            else:
                raise ValueError("Only 1 or 2 nodes can be provided")
            
            # if(voltage.shape[0] == 1):
            #     return float(voltage)
            # else:
            #     return voltage
            
            sim_descr = f"simulation type performed: {self.current_simulation_type}."
            if(voltage.shape[0] == 1):
                sim_descr = sim_descr + f"Voltage: {float(voltage)}"
            else:
                sim_descr = sim_descr + f"Voltage is a time series data."

            if(show_plot):
                return {"voltage": voltage, "plt":plt, "axe":axe, "sim_descr": sim_descr}
            else:
                return {"voltage": voltage, "sim_descr": sim_descr}
        
        else:
            raise ValueError("Nodes name not provided")

    def get_current(self, element_name:str, show_plot=False):

        if element_name.lower() not in self.analysis.branches:
            self.add_ammeter_with_element(self.circuit, element_name) # name of the ammeter will be V<element_name>_plus
            self.re_simulate()
            ammeter_name = f'V{element_name}_plus' # in the branch list, elemnt name is in lower case
            current = self.analysis.branches[ammeter_name.lower()]

            # now remove the ammeter
            # self.circuit._remove_element(self.circuit[ammeter_name])
            # no need to remove the element, removing the element does not remove the extra node added and causes issues later on
        else:
            current = self.analysis.branches[element_name.lower()]


        if(current.shape[0] > 1 and show_plot):
            import matplotlib.pyplot as plt
            from PySpice.Probe.Plot import plot
            import mplcursors

            figure = plt.figure(f'Bring Circuit to Life - time vs {element_name} current graph')
            axe = plt.subplot(111)
            plot(self.analysis.branches[element_name.lower()], axis=axe)
            
            plt.title(f"time vs 'current through {element_name}' graph")
            plt.xlabel('Time [s]')
            plt.ylabel(f"Current [A] through {element_name}")
            
            self.show_plot(plt, axe)

        # TODO: direction of current can be explicitly returned
        # TODO: return dict, add sim_descr

        return current
    
    def get_power(self, element_names: list, show_plot=False):
        import matplotlib.pyplot as plt
        from PySpice.Probe.Plot import plot

        powers = {}
        
        for element_name in element_names:
            current = self.get_current(element_name, show_plot=False)
            element = self.circuit[element_name]
            nodes = element.node_names
            voltage = self.get_voltage(nodes_name=nodes, show_plot=False)['voltage']
            power = current * voltage
            powers[element_name] = power

        if show_plot:
            figure = plt.figure(f'Bring Circuit to Life - time vs power graph')
            axe = plt.subplot(111)
            for element_name, power in powers.items():
                plot(power, axis=axe, label=f'{element_name} Power')
            
            plt.title(f"time vs power graph")
            plt.xlabel('Time [s]')
            plt.ylabel(f"Power [W]")
            plt.legend()
            self.show_plot(plt, axe)

        sim_descr = f"simulation type performed: {self.current_simulation_type}"

        if(show_plot):
            return {"powers": powers, "plt":plt, "axe":axe, "sim_descr": sim_descr}
        else:
            return {"powers": powers, "sim_descr": sim_descr}
        
    def change_ground(self, new_ground:str):
        # TODO: 
        print("not implemented yet")
        return
    
    def show_plot(self, plt, axe):
        import mplcursors

        plt.grid(True)
        axe.axhline(0, color='black', linewidth=1, linestyle='--')
        axe.axvline(0, color='black', linewidth=1, linestyle='--')
        
        # Dynamically set xlim and ylim
        x_min, x_max = axe.get_xlim()
        y_min, y_max = axe.get_ylim()
        
        axe.set_xlim(left=max(x_min, -15), right=min(x_max, 15))
        axe.set_ylim(bottom=max(y_min, -15), top=min(y_max, 15))
        
        mplcursors.cursor(axe, hover=True)
        plt.gcf().set_size_inches(10, 5)  # Change the figure size
        plt.show(block=False)  # Ensure the plot stays open until closed by the user

if __name__ == "__main__":

    # Analyse the circuit
    import warnings
    warnings.filterwarnings("ignore")

    ############### operaitng point analysis
    # # # Create a new circuit
    # circuit = Circuit('Captured Circuit from Image')

    # circuit.V(1, '1', '0', 10@u_V)
    # circuit.C(1, '0', '2', 1@u_uF)
    # circuit.R(1, '0', '3', 1@u_kΩ)
    # circuit.V(2, '4', '5', 10@u_V)
    # circuit.R(2, '0', '6', 1@u_kΩ)
    # circuit.L(1, '8', '7', 1@u_mH)
    # circuit.R(3, '2', '3', 1@u_kΩ)
    # circuit.L(2, '0', '3', 1@u_mH)
    # circuit.R(4, '4', '6', 1@u_kΩ)
    # circuit.R(5, '8', '5', 1@u_kΩ)
    # circuit.R(6, '1', '5', 1@u_kΩ)
    # circuit.R(7, '3', '7', 1@u_kΩ)

    # # print(circuit)


    # analyzer = Analyzer(circuit)
    # analyzer.operating_point()
    
    # v = analyzer.get_voltage(analysis=analysis, nodes_name=['5', '1'], show_plot=False)
    # if(v.shape[0] == 1):
    #     print(float(v))

    # current = analyzer.get_current(element_name='R4', show_plot=False)
    # if(current.shape[0] == 1):
    #     print(float(current))

    # power = analyzer.get_power(element_name='R2', show_plot=False)
    # if(power.shape[0] == 1):
    #     print(float(power))


    ####################### transient analysis
    # Create a new Circuit
    # circuit = Circuit('RC Circuit')

    # circuit.V(1, '1', '0', 10)  # DC Voltage Source: 5V between nodes 'in' and ground
    # circuit.R(1, '1', '2', 1e3)       # Resistor: 1 kOhm between 'in' and 'node1'
    # circuit.C(1, '2', '3', 1e-6) # Capacitor: 1uF between 'node1' and ground
    # circuit.C(2, '3', '4', 2e-6)
    # circuit.R(2, '4', '0', 2e3)

    # circuit.V(1, '1', '0', 10)  # DC Voltage Source: 5V between nodes 'in' and ground
    # circuit.R(1, '1', '2', 1e3)       # Resistor: 1 kOhm between 'in' and 'node1'
    # circuit.C(1, '2', '0', 1e-6) # Capacitor: 1uF between 'node1' and ground
    # print(circuit)

    # analyzer = Analyzer(circuit)
    # analyzer.transient_analysis(initial_conditions= [['C1', 3], ['C2', 8]] ,start_time=0, stop_time=1e-3, step_time=1e-6)
    # analyzer.transient_analysis(initial_conditions= [['C1', 3]] ,start_time=0, stop_time=20e-3, step_time=1e-6)

    # analyzer.get_voltage(nodes_name=['1','2'], show_plot=True)
    # analyzer.get_current(element_name='V1', show_plot=True)
    # analyzer.get_power(element_name='C1', show_plot=True)

    # keep these codes at the end of the simulation to keep the plot open after the simulation is done
   

    circuit= Circuit("new ckt") 
    circuit.V(1, '1','2', 10)
    circuit.R(1, '2','0', 1000)
    circuit.R(2, '0','1', 1000)
    circuit.C(1, '0','1', 1e-6)
    
    analyzer = Analyzer(circuit)
    analyzer.transient_analysis(initial_conditions= [['C1', 0]] , stop_time=10e-3, step_time=1e-6)
    analyzer.get_voltage(nodes_name=['0','1'], show_plot=True)
    output_dict = analyzer.get_power(element_names=['R1', 'R2', 'C1'], show_plot=True)
    print(output_dict["sim_descr"])

    import matplotlib.pyplot as plt
    plt.show()



