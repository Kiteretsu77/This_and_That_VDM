'''
    This repo is to combine multiple generated images with same index together 
'''

import os, shutil, sys
import imageio
import math
import cv2
from PIL import Image
import collections
import numpy as np


if __name__ == "__main__":

    # Basic setting
    data_paths = [
                    "human_evaluation_v3_V_raw_prompt",
                    "human_evaluation_v3_VG_raw_prompt_no_sam",
                    "human_evaluation_v3_VL_ambiguous_prompt",

                    "../datasets_rob/Bridge_human_evaluation",

                    "human_evaluation_v3_VL_raw_prompt",
                    "human_evaluation_v3_VGL_raw_prompt_no_sam",
                    "human_evaluation_v3_VGL_ambiguous_prompt_no_sam",
                ]
    store_path = "combined_results_human_evaluation"
    sample_data_path = data_paths[0]
    gif_per_row = 4     # Number of GIF files per row


    # Create folder
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    os.makedirs(store_path)


    # Iterate the sample
    for instance_idx, sub_folder_name in enumerate(sorted(os.listdir(sample_data_path))):
        print("we are processing ", sub_folder_name)
        
        collected_gif_paths = []
        for data_path in data_paths:
            collected_gif_paths.append(os.path.join(data_path, sub_folder_name, 'combined.gif'))

        # Merge frames together
        rows = math.ceil(len(collected_gif_paths) / gif_per_row)
        cols = gif_per_row

        # Read all input GIFs and find maximum dimensions
        gifs = []
        max_width, max_height = 0, 0
        for path in collected_gif_paths:
            gif = imageio.mimread(path)
            max_width = max(max_width, gif[0].shape[1])
            max_height = max(max_height, gif[0].shape[0])
            gifs.append(gif)

            # Create blank canvas for concatenated GIF
            frames_length = len(gifs[0])
            canvas_width = max_width * cols
            canvas_height = max_height * rows
            canvas = np.zeros((frames_length, canvas_height, canvas_width, 3), dtype=np.uint8)


        # push each frame into the canvas placeholder   
        gif_index = 0
        for row in range(rows):
            for col in range(cols):
                gif = gifs[gif_index]
                gif_height, gif_width, _ = gif[0].shape
                start_y = row * max_height
                start_x = col * max_width
                for i in range(frames_length):
                    canvas[i, start_y:start_y+gif_height, start_x:start_x+gif_width, :] = gif[i]

                # Update index
                gif_index += 1
                if gif_index == len(collected_gif_paths):
                    break


        # Write the concatenated GIF
        imageio.mimsave(os.path.join(store_path, sub_folder_name + ".gif"), canvas, duration=0.05, quality=100)