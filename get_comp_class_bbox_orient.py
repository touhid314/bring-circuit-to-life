'''
finds the class and orientation of a component image
'''

from get_comp_bbox import get_comp_bbox

import torch
from PIL import Image
import matplotlib.pyplot as plt

def crop_bounding_boxes(img_path, bbox_arr, should_plot=False):
    '''
    returns a list of PIL images and their class
    '''

    # Load the image as a tensor
    img = Image.open(img_path).convert("RGB")
    img_tensor = torch.tensor([[[pixel for pixel in img.getdata(band=i)] for i in range(3)]], dtype=torch.float32)
    img_tensor = img_tensor.view(3, img.height, img.width)  # Format as (C, H, W)

    # List to store cropped images
    cropped_images = []

    for bbox in bbox_arr:
        cls, x, y, w, h = bbox
        
        x_center, y_center = int(x * img.width), int(y * img.height)
        width, height = int(w * img.width), int(h * img.height)

        # Calculate the top-left and bottom-right coordinates
        x1, y1 = x_center - width // 2, y_center - height // 2
        x2, y2 = x_center + width // 2, y_center + height // 2

        # Crop the bounding box area
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_images.append((cls, cropped_img))

        # Plotting cropped images if should_plot is True
        if should_plot:
            plt.imshow(cropped_img)
            plt.title(f"Class: {cls}")
            plt.axis("off")
            plt.show()
        

        

    return cropped_images


# model
from classification_model.model import predict


def get_comp_bbox_class_orient(img_path):
    '''
    given an image path, returns a python array of format:
    [
    [comp_class, x, y, w, h, comp_orientation],
    [comp_class, x, y, w, h, comp_orientation],
    ...
    ]
    '''
    import time

    start_time = time.time()  # Record the start time
    bbox_arr = get_comp_bbox(img_path)

    end_time = time.time()  # Record the end time
    print(f">>>>>>>>>> Execution time for yolo: {end_time - start_time:.4f} seconds")

    # print(bbox_arr)
    # # only keeping the bounding boxes of COMPONENTS, ignoring text altogether now
    # for i,row in enumerate(bbox_arr):
    #     if int(row[0]) == 0: pass
    #     elif int(row[0]) == 1:      
    #         bbox_arr[i][0] = -1 # -1 class means it's text
    #     elif int(row[0]) == 2:
    #         bbox_arr[i][0] = -2 # -2 class means it's wire overlap
    
    # print(bbox_arr)

    cropped_images = crop_bounding_boxes(img_path, bbox_arr, should_plot=False)

    start_time = time.time()
    for index, row in enumerate(cropped_images):
        img = row[1]
        if bbox_arr[index][0] == 0:
            # this cropped image is for a electrical component        
            comp_class, orient = predict(img)
            bbox_arr[index][0] = comp_class
            bbox_arr[index].append(orient)
        elif bbox_arr[index][0] == 1:
            # this cropped image is for text
            bbox_arr[index][0] = -1 # -1 class means it's text
        elif bbox_arr[index][0] == 2:
            # this cropped image is for wire overlap
            bbox_arr[index][0] = -2 # -2 class means it's wire overlap
        else:
            raise Exception("invalid class number found in bbox_arr")

    end_time = time.time()  # Record the end time
    print(f">>>>>>>>>> Execution time for classfier model: {end_time - start_time:.4f} seconds")

    return bbox_arr


if __name__ == "__main__":
    import pprint
    img_path = r"C:\Users\TestUser\Desktop\bring_ckt_to_life_project\example ckts which the final product should work properly on\signal-2024-11-11-034730_002.jpeg"

    comp_bbox = get_comp_bbox_class_orient(img_path)
    pprint.pprint(comp_bbox)