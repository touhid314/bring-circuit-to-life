'''
this module is to act as a language interface between the user and the program
'''

import time
from llm.llm_inference import get_llm_response

def process_prompt(prompt:str, llm_model, ckt_netlist, analyzer):
    '''
    arguments:

    given a prompt, this function, finds the response to the prompt by an LLM and perform 2 things:
    1.    execute code generated by the prompt 
    2.    return non-code things
    '''
    import re
    gen_plt = None
    gen_axe = None

    response = get_llm_response(prompt, llm_model, ckt_netlist, show_execution_time=True)

    exec_pattern = re.compile(r'<exec>(.*?)</exec>')
    exec_commands = exec_pattern.findall(response)
    non_exec_response = exec_pattern.sub('', response).strip()

    if len(exec_commands) != 0:
        # execute analysis as per code_str
        import traceback

        for exec_cmd in exec_commands:
            try:
                print(f"executing...: {exec_cmd}")

                # Define a context with the analyzer
                context = {'analyzer': analyzer}
                exec(f"def dynamic_func():\n    return {exec_cmd}", context)
                result = context['dynamic_func']()

                print(f"return of execution: {result}, type of return value: {type(result)}")

                # plotting, if the output has any plotting info
                if isinstance(result, dict) and 'plt' in result and 'axe' in result and result['plt'] is not None and result['axe'] is not None:
                    show_plot(result['plt'], result['axe'])
                    gen_plt = result['plt']
                    gen_axe = result['axe']
            except Exception as e:
                print(f"llm generated code execution failed. execution code: {exec_cmd}")
                print(f"> Error message: {e}")
                traceback.print_exc()
                print(f"end of error message <")
    
    return {"non_exec_response":non_exec_response, "plt":gen_plt, "axe":gen_axe}

def show_plot(plt, axe):
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
    prompt = "plot the power consumed by the elements R1, R2 and the capacitor"
    model_path = "llm\models\llama-2-13b-chat.ggmlv3.q5_1.bin"

    # circuit
    from PySpice.Spice.Netlist import Circuit
    # circuit = Circuit('RC Circuit')
    # circuit.V(1, '1', '0', 10)  # DC Voltage Source: 5V between nodes 'in' and ground
    # circuit.R(1, '1', '2', 1e3)       # Resistor: 1 kOhm between 'in' and 'node1'
    # circuit.C(1, '2', '0', 1e-6) # Capacitor: 1uF between 'node1' and ground
    circuit= Circuit("new ckt") 
    circuit.V(1, '1','2', 10)
    circuit.R(1, '2','0', 1000)
    circuit.R(2, '0','1', 1000)
    circuit.C(1, '0','1', 1e-6)

    # create an analyzer object for the circuit and the llm will perform operations by using this analyzer object
    # analyzer object is like the connecting wire between the LLM and the simulator to send instructions
    from analyse import Analyzer
    analyzer = Analyzer(circuit)

    response_text = process_prompt(prompt, model_path, circuit, analyzer)

    print(f"CIRCUIT SAYS: {response_text}")

    import matplotlib.pyplot as plt
    plt.show()
