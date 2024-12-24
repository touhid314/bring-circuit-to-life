'''
this module is to act as a language interface between the user and the program
'''

'''
setting up llama-cpp-python for CPU:

requirements:
pip install llama-cpp-python==0.1.78
pip install huggingface_hub

'''

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format

from llama_cpp import Llama
import torch

########################## SETTING UP MODEL ###########################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model_path = "./llm/models/llama-2-13b-chat.ggmlv3.q5_1.bin"
import multiprocessing
cpu_threads = multiprocessing.cpu_count()

if(device == "cpu"):
    print("Using CPU, inference may take longer time to complete.")
    
    lcpp_llm = Llama(model_path=model_path,
                n_threads=cpu_threads, # CPU cores
                )
    
else:
    print("Using GPU, inference will be faster.")
    
    lcpp_llm = Llama(
    model_path=model_path,
    n_threads=cpu_threads, # CPU cores
    n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=43, # Change this value based on your model and your GPU VRAM pool.
    n_ctx=4096, # Context window
    )


########################## INFERNCE WITH MODEL ###########################

system_instruction = "SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully."
prompt = "Write a linear regression in python"

prompt_template=f''' {system_instruction}

USER: {prompt}

ASSISTANT:
'''

response = lcpp_llm(
    prompt=prompt_template,
    max_tokens=256,
    temperature=0.5,
    top_p=0.95,
    repeat_penalty=1.2,
    top_k=50,
    stop = ['USER:'], # Dynamic stopping when such token is detected.
    echo=True # return the prompt
)

print(response["choices"][0]["text"])



