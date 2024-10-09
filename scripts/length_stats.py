import os, sys, shutil
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    input_folder_path = "../Bridge_v2"
    
    average_length = []

    # Iterate each file
    for sub_folder_name in sorted(os.listdir(input_folder_path)):
        sub_folder_path = os.path.join(input_folder_path, sub_folder_name)
        
        average_length.append(len(os.listdir(sub_folder_path)))       # Have more than one than expected, but we keep this 
        print("average length of {} is {}".format(sub_folder_name, average_length[-1]))

    print("average_movement_list is ", average_length)
    n, bins, patches = plt.hist(average_length, bins=100)
    plt.savefig("dataset_length2.png")

