import time
from llm.llm_model import get_llm_model
from llm.system_instruction import get_system_instruction

def get_llm_response(prompt:str, lcpp_llm, context=None, ckt_netlist=None, show_execution_time:bool = True):
    """
    lcpp_llm - a llama cpp compatible LLM model
    """

    if(show_execution_time): inference_start_time = time.time()
 
    # prompt generation
    system_instruction = get_system_instruction(ckt_netlist, context)
    
    prompt_template=f"""SYSTEM:{system_instruction}

    USER: {prompt}

    RESPONSE:

    """

    # inference on prompt
    if(lcpp_llm != None):        
        response = lcpp_llm(
        prompt=prompt_template,
        max_tokens=2000,
        temperature=0.5,
        top_p=0.95,
        repeat_penalty=1, #1.2
        top_k=25, #50
        stop = ['USER:'], # Dynamic stopping when such token is detected.
        echo=False # don't return the prompt
    )
        response = response["choices"][0]["text"]
        # print(response)
    else:
        from llm.llm_prompts import get_response
        response = get_response(prompt)

    if(show_execution_time):
        inference_end_time = time.time()
        print(f"Inference completed in {inference_end_time - inference_start_time:.2f} seconds.")

    return response