'''
    This file is to split the video sources in a folder to folder with images, for the mass evaluation
'''
import os, shutil, sys
import cv2


if __name__ == "__main__":
    input_folder = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/StreamingT2V_results"
    needed_frame_length = 14

    idx = 0
    for file_name in sorted(os.listdir(input_folder)):
        print("We are processing ", file_name)
        sub_folder_path = os.path.join(input_folder, file_name)

        for idx in range(len(os.listdir(sub_folder_path))):
            if idx >= needed_frame_length:
                target_path = os.path.join(sub_folder_path, str(idx)+".png")
                os.remove(target_path)


