'''
for finding bounding box for component and text
'''

# from git import Repo  # pip install gitpython
# Repo.clone_from("https://github.com/ultralytics/yolov5", "./yolo_model")

import os, subprocess, shutil
from pathlib import Path
import time 

os.environ['WANDB_DISABLED'] = 'true'

def get_comp_bbox(img_path):
    '''
    given an image path, returns an array of format:
    [
    [comp_class, x, y, w, h],
    [comp_class, x, y, w, h],
    ...
    ]

    comp_class = 0,1,2
    uses YOLO v5 model for segmentation
    '''

    img_path = img_path.replace("\\", "/") # converting windows style image path to linux  style

    model_path = "./yolo_model/runs/train/exp/weights/best.pt"
    parent_dir = Path(r".\yolo_model\runs\detect").resolve() # all folders within it are deleted in the beginning


    # had to change the pathlib.py file in lib at dir "C:\Users\TestUser\AppData\Local\Programs\Python\Python39\Lib" 
    # otherwise, posixPath not implemented error was coming out.

    # def __new__(cls, *args, **kwargs):
    #         if cls is Path:
    #             cls = WindowsPath if os.name == 'nt' else PosixPath
    #         self = cls._from_parts(args, init=False)

    #         ## COMMENTED OUT THE FOLLOWING LINE (LINE NUMBER 959-961 in python 3.10.0)
    #         # if not self._flavour.is_supported:
    #         #     raise NotImplementedError("cannot instantiate %r on your system"
    #         #                               % (cls.__name__,))
    #         self._init()
    #         return self 
            

    # predict
    remove_after_display = True
    # remove detection result after detection is complete

    if(remove_after_display):
        # Loop through each item in the directory
        for item in parent_dir.iterdir():
            # Check if the item is a directory (folder)
            if item.is_dir():
                shutil.rmtree(item)  # Remove the folder and all its contents
                # print(f"Removed folder: {item}")
            # else:
                # print(f"Skipped file: {item}")


    # run detection
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    print(f"Using device: {device}")    

    from yolo_model import detect
    
    detect.run(weights=model_path,
               source=img_path,
               save_txt=True,
               name="my_detection",
               line_thickness=1,
               device=device)
    
    # fetch output
    img_name = img_path.split('/')[-1]
    img_name = '.'.join(img_name.split('.')[:-1]) #stripping off the extention in the end

    # fetch label txt file into an array
    # det_result_arr - an array containing the output txt file holding the object detection result into an array of float format
    with open(f"./yolo_model/runs/detect/my_detection/labels/{img_name}.txt", "r") as file:
        det_result = file.read()

    det_result_arr = det_result.splitlines()
    det_result_arr = [list(map(float, item.split())) for item in det_result_arr]
  
    return det_result_arr


if __name__ == "__main__":
    img_path = "../dataset/_BC2L_SEGMENTATION_DATASET/test/images/autockt_-3_png_jpg.rf.fe802879b294b548a8ed4854227978cf.jpg"
    # img_path = r"C:\Users\TestUser\Desktop\bring_ckt_to_life_project\dataset\_BC2L_SEGMENTATION_DATASET\test\images\20220904_104814_jpg.rf.a9de0b3dc08c6aaa8e6191c5e18cecc0.jpg"

    a = get_comp_bbox(img_path)

    print(a)

