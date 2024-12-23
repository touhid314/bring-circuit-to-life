from PySpice.Spice.Netlist import Circuit
import numpy
from PySpice.Unit import *



from typing import Union, Tuple

class Analyzer:
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    def get_comp_voltages(self, COMPONENTS):
        analysis = self.simulator.operating_point()
        node_voltages = {str(node): float(node) for node in analysis.nodes.values()}

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


    def operating_point(self, output_type: str, nodes: Union[str, Tuple[str, str]] = None) -> Union[float, dict]:
        """
        Performs operating point analysis on the circuit with controllable output.

        Args:
            output_type (str): The type of output ("node_voltage", "branch_current", "equivalent_resistance").
            nodes (Union[str, Tuple[str, str]], optional): Node(s) for the analysis. Defaults to None.
                - For "node_voltage": A single node or two nodes for voltage difference.
                - For "branch_current": Two nodes specifying the branch.
                - For "equivalent_resistance": Two nodes between which resistance is calculated.

        Returns:
            Union[float, dict]: The computed result based on the output type.
        """
        analysis = self.simulator.operating_point()

        # Extract node voltages
        node_voltages = {str(node): float(node) for node in analysis.nodes.values()}
        print(node_voltages)

        # Extract branch currents
        branch_currents = {str(branch): float(branch) for branch in analysis.branches.values()}

        if output_type == "node_voltage":
            if isinstance(nodes, tuple) and len(nodes) == 2:
                # Voltage difference between two nodes
                v1 = node_voltages.get(nodes[0], 0.0)
                v2 = node_voltages.get(nodes[1], 0.0)
                return v1 - v2
            elif isinstance(nodes, str):
                # Voltage at a single node
                return node_voltages.get(nodes, 0.0)
            else:
                raise ValueError("Invalid nodes input for 'node_voltage'. Provide one or two node names.")
            # TODO: determine whether the node given is within the circuit, pyspice apparently doesn not throw an error if the node is not in the circuit



        elif output_type == "branch_current":
            print("not implemented yet")
    
    def dc_analysis(self):
        print("not implemented yet")

    def transient_analysis(self, start_time: float = 0, stop_time: float = 1e-3, step_time: float = 1e-6, plot: bool = True):
        """
        Performs transient analysis on the circuit and optionally plots the results.

        Parameters:
            start_time (float): Start time of the transient simulation (in seconds). Default is 0.
            stop_time (float): Stop time of the transient simulation (in seconds). Default is 1e-3.
            step_time (float): Time step for the simulation (in seconds). Default is 1e-6.
            plot (bool): Whether to plot the results. Default is True.

        Returns:
            dict: Dictionary containing time values, node voltages, and branch currents.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Run the transient analysis
        self.simulator.initial_condition(node1=0)
        analysis = self.simulator.transient(step_time=step_time, end_time=stop_time, start_time=start_time)


        # Extract time values
        time_values = np.array(analysis.time)

        # Extract node voltages
        node_voltages = {str(node): np.array(node) for node in analysis.nodes.values()}

        # Extract branch currents
        branch_currents = {str(branch): np.array(branch) for branch in analysis.branches.values()}

        # Plot results if requested
        if plot:
            plt.figure(figsize=(10, 6))
            # Plot node voltages
            for node, voltage in node_voltages.items():
                plt.plot(time_values, voltage, label=f"Node {node} Voltage")
            
            # Plot branch currents
            for branch, current in branch_currents.items():
                plt.plot(time_values, current, linestyle='--', label=f"Branch {branch} Current")
            
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title("Transient Analysis Results")
            plt.legend()
            plt.grid()
            plt.show()

        # Format the results
        results = {
            "time": time_values,
            "node_voltages": node_voltages,
            "branch_currents": branch_currents,
        }

        return results    

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


if __name__ == "__main__":

    # # Create a new circuit
    # circuit = Circuit('Captured Circuit from Image')

    # # Add components to the circuit
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

        # Create a new Circuit
    circuit = Circuit('RC Circuit')

    # Define components
    circuit.V(1, 'in', circuit.gnd, 5)  # DC Voltage Source: 5V between nodes 'in' and ground
    circuit.R(1, 'in', 'node1', 1e3)       # Resistor: 1 kOhm between 'in' and 'node1'
    circuit.C(1, 'node1', circuit.gnd, 1e-6) # Capacitor: 1uF between 'node1' and ground

    # Print the netlist
    # print(str(circuit))

    # Analyse the circuit
    import warnings
    warnings.filterwarnings("ignore")


    analyzer = Analyzer(circuit)
    # results = analyzer.operating_point("branch_current", nodes=("0", "3"))

    # print(f"Branch current between nodes 0 and 3: {results:.2f}A")
    results = analyzer.transient_analysis(start_time=0, stop_time=30e-3, step_time=1e-6)

