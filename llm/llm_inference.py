import time
from llm.llm_model import get_llm_model
from llm.system_instruction import get_system_instruction

def get_llm_response(prompt:str, model_path:str, circuit, show_execution_time:bool = True):

    # lcpp_llm = get_llm_model(model_path=model_path, show_execution_time=show_execution_time)
    
    if(show_execution_time): inference_start_time = time.time()

    ckt_netlist = str(circuit)
    system_instruction = get_system_instruction(context=None)
    prompt = "Write a linear regression in python"

    prompt_template=f''' {system_instruction}

    USER: {prompt}

    ASSISTANT:
    '''

    # response = lcpp_llm(
    #     prompt=prompt_template,
    #     max_tokens=256,
    #     temperature=0.5,
    #     top_p=0.95,
    #     repeat_penalty=1.2,
    #     top_k=50,
    #     stop = ['USER:'], # Dynamic stopping when such token is detected.
    #     echo=True # return the prompt
    # )
    # response = response["choices"][0]["text"]
    # # print(response)

    response = """this is a simulated response generated by the llm.<exec>analyzer.operating_point()</exec> <exec>analyzer.change_element_value('R1', 200)</exec> <exec>analyzer.get_voltage(['1','2'], show_plot=True)</exec>"""

    if(show_execution_time):
        inference_end_time = time.time()
        print(f"Inference completed in {inference_end_time - inference_start_time:.2f} seconds.")

    return response