import os, shutil, sys
import cv2
import imageio
import numpy as np


def compress_gif(sub_folder_path):

    # Check valid length
    all_files = os.listdir(sub_folder_path)
    num_frames_input = 0
    valid = True
    for file_name in os.listdir(sub_folder_path):
        if file_name.startswith("im_"):
            num_frames_input += 1
    for idx in range(num_frames_input):
        img_path = 'im_' + str(idx) + '.jpg'
        if img_path not in all_files:            # Should be sequential existing
            valid = False
            break
    if not valid:
        print("We cannot generate a video because the video is not sequential")
        return False
    

    if num_frames_input == 0:
        print("We cannot generate a video because the input length is 0")
        return False

    img_lists = []
    for idx in range(num_frames_input):
        img_path = os.path.join(sub_folder_path, "im_" + str(idx) + ".jpg")
        img_lists.append(cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (384, 256)))
    
    imageio.mimsave(os.path.join(sub_folder_path, 'combined.gif'), np.array(img_lists), duration=0.05, quality=100)

    return True


if __name__ == "__main__":
    dataset_path = "../datasets_rob/Bridge_human_evaluation"  # ../datasets_rob/Bridge_v1_raw

    for sub_folder_name in sorted(os.listdir(dataset_path)):
        print("We are processing ", sub_folder_name)
        sub_folder_path = os.path.join(dataset_path, sub_folder_name)

        status = compress_gif(sub_folder_path)

       


        