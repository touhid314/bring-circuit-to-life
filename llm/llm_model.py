'''
setting up llama-cpp-python for CPU:

requirements:
pip install llama-cpp-python==0.1.78
pip install huggingface_hub

'''

# for download from huggingface
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format

from llama_cpp import Llama
import torch
import time

def get_llm_model(model_path = "llm\models\llama-2-13b-chat.ggmlv3.q5_1.bin", show_execution_time:bool = True):

    # model_path = "./llm/models/llama-2-13b-chat.ggmlv3.q5_1.bin"
    # model_path = "llm\models\llama-7b.ggmlv3.q5_0.bin"
    
    if(show_execution_time): start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")


    import multiprocessing

    cpu_threads = multiprocessing.cpu_count()
    print(f"cpu threads using: {cpu_threads}")

    if(device == "cpu"):
        print("Using CPU, inference may take longer time to complete.")
        
        lcpp_llm = Llama(model_path=model_path,
                    n_threads=cpu_threads, # CPU cores
                    )

        # TODO: perform quantization
        
    else:
        print("Using GPU, inference will be faster.")
        
        lcpp_llm = Llama(
        model_path=model_path,
        n_threads=cpu_threads, # CPU cores
        n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_gpu_layers=43, # Change this value based on your model and your GPU VRAM pool.
        n_ctx=4096, # Context window
        )

    if(show_execution_time):
        end_time = time.time()
        print(f"Model setup completed in {end_time - start_time:.2f} seconds.")

    return lcpp_llm