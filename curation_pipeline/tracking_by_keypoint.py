import os, shutil, sys
import argparse
import gdown
import cv2
import numpy as np
import os
import sys
import requests
import json
import torchvision
import torch 
import psutil
import time
try: 
    from mmcv.cnn import ConvModule
except:
    os.system("mim install mmcv")


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from track_anything_code.model import TrackingAnything
from track_anything_code.track_anything_module import get_frames_from_video, download_checkpoint, parse_augment, sam_refine, vos_tracking_video
from scripts.compress_videos import compress_video




if __name__ == "__main__":
    dataset_path = "Bridge_v1_TT14"
    video_name = "combined.mp4"
    verbose = True      # If this is verbose, you will continue to write the code


    ################################################## Model setup ####################################################
    # check and download checkpoints if needed
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    xmem_checkpoint = "XMem-s012.pth"
    xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"


    folder ="./pretrained"
    SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
    xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)

    # argument
    args = parse_augment()
    args.device = "cuda"      # Any GPU is ok

    # Initialize the Track model
    track_model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, args)
    ###################################################################################################################


    # Iterate all files under the folder
    for sub_folder_name in sorted(os.listdir(dataset_path)):

        ################################################## Setting ####################################################
        sub_folder_path = os.path.join(dataset_path, sub_folder_name)

        click_state = [[],[]]
        interactive_state = {
                                "inference_times": 0,
                                "negative_click_times" : 0,
                                "positive_click_times": 0,
                                "mask_save": args.mask_save,
                                "multi_mask": {
                                    "mask_names": [],
                                    "masks": []
                                },
                                "track_end_number": None,
                                "resize_ratio": 1
                            }
        ###################################################################################################################
        

        video_path = os.path.join(sub_folder_path, video_name)
        if not os.path.exists(video_path):
            print("We cannot find the path of the ", video_path, " and we will compress one")
            status = compress_video(sub_folder_path, video_name)
            if not status:
                print("We still cannot generate a video")
                continue

        # Read video state
        video_state = { 
                        "user_name": "",
                        "video_name": "",
                        "origin_images": None,
                        "painted_images": None,
                        "masks": None,
                        "inpaint_masks": None,
                        "logits": None,
                        "select_frame_number": 0,
                        "fps": 30
                    }
        video_state, template_frame = get_frames_from_video(video_path, video_state, track_model)
        


        ########################################################## Get the sam point based on the data.txt ###########################################################
        data_txt_path = os.path.join(sub_folder_path, "data.txt")
        if not os.path.exists(data_txt_path):
            print("We cannot find data.txt in this folder")
            continue

        data_file = open(data_txt_path, 'r')
        lines = data_file.readlines()
        frame_idx, horizontal, vertical = lines[0][:-2].split(' ')   # Only read the first point
        point_cord = [int(float(horizontal)), int(float(vertical))]

        # Process by SAM
        track_model.samcontroler.sam_controler.reset_image() # Reset the image to clean history
        painted_image, video_state, interactive_state, operation_log = sam_refine(track_model, video_state, "Positive", click_state, interactive_state, point_cord)
        ################################################################################################################################################################



        ######################################################### Get the tracking output ########################################################################
        
        # Track the video for processing
        segment_output_path = os.path.join(sub_folder_path, "segment_output.gif")
        video_state = vos_tracking_video(track_model, segment_output_path, video_state, interactive_state, mask_dropdown=[])[0]   # mask_dropdown is empty now
        
        # Extract the mask needed by us for further point calculating
        masks = video_state["masks"]        # In the range [0, 1]
        
        if verbose:
            for idx, mask in enumerate(masks):
                cv2.imwrite(os.path.join(sub_folder_path, "mask"+str(idx)+".png"), mask*255)

        ##############################################################################################################################################################


