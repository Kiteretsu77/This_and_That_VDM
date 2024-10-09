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


def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    return mask_image * 255


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1)


if __name__ == "__main__":
    input_parent_folder = "validation_tmp"


    # Init SAM for segmentation task
    model_type = "vit_h"
    weight_path = "pretrained/sam_vit_h_4b8939.pth"



    sam = sam_model_registry[model_type](checkpoint=weight_path).to(device="cuda")
    sam_predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    
    # Iterate the folder    
    for sub_dir_name in sorted(os.listdir(input_parent_folder)):
        print("We are processing ", sub_dir_name)
        ref_img_path = os.path.join(input_parent_folder, sub_dir_name, 'im_0.jpg')
        data_txt_path = os.path.join(input_parent_folder, sub_dir_name, 'data.txt')


        # Read the image and process
        image = cv2.imread(ref_img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # Read the positive point
        data_file = open(data_txt_path, 'r')
        lines = data_file.readlines()
        for idx in range(len(lines)):
            frame_idx, horizontal, vertical = lines[idx].split(' ')
            vertical, horizontal = int(float(vertical)), int(float(horizontal))
            positive_point_cords = [[horizontal, vertical]]
            
            positive_point_cords = np.array(positive_point_cords)
            positive_point_labels = np.ones(len(positive_point_cords))
            print(positive_point_cords)
            


            # Set the SAM predictor
            sam_predictor.set_image(np.uint8(image))
            masks, scores, logits = sam_predictor.predict(
                                                point_coords = positive_point_cords,  # Only positive points here
                                                point_labels = positive_point_labels, 
                                                multimask_output = False,
                                                )
            # print("Detected mask length is ", len(masks))
            
            # Visualize
            mask_img = show_mask(masks[0])
            cv2.imwrite(os.path.join(input_parent_folder, sub_dir_name, "first_contact0.png"), mask_img)

            break


        # SAM all
        sam_all = mask_generator.generate(image)
        all_sam_imgs = show_anns(sam_all)
        cv2.imwrite("sam_all.png", all_sam_imgs)


        

