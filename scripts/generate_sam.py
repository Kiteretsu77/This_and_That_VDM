import os, sys, shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


def show_anns(anns):
    if len(anns) == 0:
        return
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    # img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3)])
        img[m] = color_mask
    
    return img*255




if __name__ == "__main__":
    input_parent_folder = "../Bridge_filter_flow"


    # Init SAM for segmentation task
    model_type = "vit_h"
    weight_path = "pretrained/sam_vit_h_4b8939.pth"



    sam = sam_model_registry[model_type](checkpoint=weight_path).to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)     # There is a lot of setting here


    for sub_dir_name in sorted(os.listdir(input_parent_folder)):
        print("We are processing ", sub_dir_name)
        ref_img_path = os.path.join(input_parent_folder, sub_dir_name, 'im_0.jpg')
        store_path = os.path.join(input_parent_folder, sub_dir_name, 'sam.png')
        
        image = cv2.imread(ref_img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = mask_generator.generate(image)
        mask_img = show_anns(mask)

        cv2.imwrite(store_path, mask_img)

        

