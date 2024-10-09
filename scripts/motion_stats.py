import os, sys, shutil
import numpy as np
import math
from statistics import mean
import matplotlib.pyplot as plt


if __name__ == "__main__":
    input_folder_paths = ["../datasets_rob/Bridge_v1_raw", "../datasets_rob/Bridge_v2_raw"]     # "../datasets_rob/Bridge_v1_raw", "../datasets_rob/Bridge_v2_raw"
    num_frames = 14
    store_name = "movement.png"
    

    average_movement_list = []
    not_valid_num = 0
    not_exists_num = 0
    # Iterate each file
    for input_folder_path in input_folder_paths:
        for sub_folder_name in sorted(os.listdir(input_folder_path)):
            sub_folder_path = os.path.join(input_folder_path, sub_folder_name)
            flow_path = os.path.join(sub_folder_path, 'flow.txt')

            if not os.path.exists(flow_path):
                not_exists_num += 1
                continue


            # Read the movement
            file = open(flow_path, 'r')
            info = file.readlines()
            print(info)
            if len(info) == 0:
                not_valid_num += 1
                continue
            info = info[0][:-2]
            per_video_movement = float(info)


            # Calculate the number of frames in this video
            num_frames_input = 0
            valid = True
            for file_name in os.listdir(sub_folder_path):   # num_frames_input is the total number of files with name begin with im_
                if file_name.startswith("im_"):
                    num_frames_input += 1
            for idx in range(num_frames_input):     # Ensure that this number is concurrent
                img_path = os.path.join(sub_folder_path, 'im_' + str(idx) + '.jpg')
                if not os.path.exists(img_path):            # Should be sequential existing
                    valid = False
                    break
            if num_frames_input < 2:
                valid = False
            if not valid:
                not_valid_num += 1
                print("This is not valid path")
                continue
            
            average_movement_list.append(per_video_movement * (num_frames_input/num_frames))       # Have more than one than expected, but we keep this 
            print("average movement of {} is {}".format(sub_folder_name, average_movement_list[-1]))

    print("not_exists_num is ", not_exists_num)
    print("not_valid_num is ", not_valid_num)
    print("average_movement_list length is ", len(average_movement_list))

    # Get mean and variance data
    mean_value = mean(average_movement_list)
    std_value = math.sqrt(np.var(average_movement_list))
    print("Mean is ", mean_value)
    print("std_value is ", std_value)

    # Plot the figure
    n, bins, patches = plt.hist(average_movement_list, bins=100)
    plt.title("Mean" + str(mean_value) + "_STD"+str(std_value))
    plt.savefig(store_name)


