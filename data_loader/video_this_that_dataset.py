import os, sys
import json
import cv2
import math
import shutil
import numpy as np
import random
from PIL import Image
import torch.nn.functional as F
import torch
import os.path as osp
import time
from moviepy.editor import VideoFileClip
from torch.utils.data import Dataset    

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from utils.img_utils import resize_with_antialiasing, numpy_to_pt
from utils.optical_flow_utils import flow_to_image, filter_uv, bivariate_Gaussian
from data_loader.video_dataset import tokenize_captions


# For the 2D dilation
blur_kernel = bivariate_Gaussian(99, 10, 10, 0, grid = None, isotropic = True)


def get_thisthat_sam(config, intput_dir, store_dir = None, flip = False, verbose=False):
    '''
    Args:
        idx (int): The index to the folder we need to process
    '''

    # Read file
    file_path = os.path.join(intput_dir, "data.txt")
    file1 = open(file_path, 'r')
    Lines = file1.readlines()


    # Initial the optical flow format we want
    thisthat_condition = np.zeros((config["video_seq_length"], config["conditioning_channels"], config["height"], config["width"]), dtype=np.float32)  # The last image should be empty


    # Init the image
    sample_img = cv2.imread(os.path.join(intput_dir, "im_0.jpg"))
    org_height, org_width, _ = sample_img.shape

    # Prepare masking
    controlnet_image_index = []
    coordinate_values = []

    # Iterate all points in the txt file
    for idx in range(len(Lines)):

        # Read points
        frame_idx, horizontal, vertical = Lines[idx].split(' ')
        frame_idx, vertical, horizontal = int(frame_idx), int(float(vertical)), int(float(horizontal))

        # Read the mask frame idx
        controlnet_image_index.append(frame_idx)
        coordinate_values.append((vertical, horizontal))


        # Init the base image
        base_img = np.zeros((org_height, org_width, 3)).astype(np.float32)      # Use the original image size
        base_img.fill(255)

        # Draw square around the target position
        dot_range = 10       # Diameter
        for i in range(-1*dot_range, dot_range+1):
            for j in range(-1*dot_range, dot_range+1):
                dil_vertical, dil_horizontal = vertical + i, horizontal + j
                if (0 <= dil_vertical and dil_vertical < base_img.shape[0]) and (0 <= dil_horizontal and dil_horizontal < base_img.shape[1]):
                    if idx == 0:
                        base_img[dil_vertical][dil_horizontal] = [0, 0, 255]        # The first point should be red
                    else:
                        base_img[dil_vertical][dil_horizontal] = [0, 255, 0]        # The second point should be green to distinguish the first point
        
        # Dilate
        if config["dilate"]:
            base_img = cv2.filter2D(base_img, -1, blur_kernel)


        ##############################################################################################################################
        ### The core pipeline of processing is: Dilate -> Resize -> Range Shift -> Transpose Shape -> Store

        # Resize frames  Don't use negative and don't resize in [0,1]
        base_img = cv2.resize(base_img, (config["width"], config["height"]), interpolation = cv2.INTER_CUBIC)


        # Flip the image for aug if needed
        if flip:
            base_img = np.fliplr(base_img)


        # Channel Transform and Range Shift
        if config["conditioning_channels"] == 3:
            # Map to [0, 1] range 
            if store_dir is not None and verbose:    # For the first frame condition visualization
                cv2.imwrite(os.path.join(store_dir, "condition_TT"+str(idx)+".png"), base_img)
            base_img = base_img / 255.0         

        else:
            raise NotImplementedError()

        
        # ReOrganize shape
        base_img = base_img.transpose(2, 0, 1)  # hwc -> chw


        # Check the min max value range
        # if verbose:
        #     print("{} min, max range value is {} - {}".format(intput_dir, np.min(base_img), np.max(base_img)))


        # Write base img based on frame_idx
        thisthat_condition[frame_idx] = base_img        # Only the first frame, the rest is 0 initialized

    ##############################################################################################################################


    if config["motion_bucket_id"] is None:
        # take the motion to stats collected before
        reflected_motion_bucket_id = 200
    else:
        reflected_motion_bucket_id = config["motion_bucket_id"]
    

    # print("Motion Bucket ID is ", reflected_motion_bucket_id)
    return (thisthat_condition, reflected_motion_bucket_id, controlnet_image_index, coordinate_values)    



class Video_ThisThat_Dataset(Dataset):
    '''
        Video Dataset to load sequential frames for training with needed pre-processing and process with optical flow
    '''
    
    def __init__(self, config, device, normalize=True, tokenizer=None):
        # Attribute variables
        self.config = config
        self.device = device
        self.normalize = normalize
        self.tokenizer = tokenizer

        # Obtain values
        self.video_seq_length = config["video_seq_length"]
        self.height = config["height"]
        self.width = config["width"]

        # Process data
        self.video_lists = []
        for dataset_path in config["dataset_path"]:
            for video_name in sorted(os.listdir(dataset_path)):
                if not os.path.exists(os.path.join(dataset_path, video_name, "data.txt")):
                    continue

                self.video_lists.append(os.path.join(dataset_path, video_name))
        print("length of the dataset is ", len(self.video_lists))



        
    def __len__(self):
        return len(self.video_lists)
    

    def _extract_frame_bridge(self, idx, flip=False):
        ''' Extract the frame in video based on the needed fps from already extracted frame
        Args:
            idx (int):                  The index to the file in the directory
            flip (bool):                Bool for whether we will flip
        Returns:
            video_frames (numpy):       Extracted video frames in numpy format
        '''

        # Init the the Video Reader
        # The naming of the Bridge dataset follow a pattern: im_x.jpg, so we need to 
        video_frame_path = self.video_lists[idx]


        # Find needed file
        needed_img_path = []
        for idx in range(self.video_seq_length):
            img_path = os.path.join(video_frame_path, "im_" + str(idx) + ".jpg")
            needed_img_path.append(img_path)



        # Read all img_path based on the order 
        video_frames = []
        for img_path in needed_img_path:
            if not os.path.exists(img_path):
                print("We don't have ", img_path)
            frame = cv2.imread(img_path)

            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception:
                print("The exception place is ", img_path)
            # Resize frames
            frame = cv2.resize(frame, (self.width, self.height), interpolation = cv2.INTER_CUBIC)

            # Flip aug
            if flip:
                frame = np.fliplr(frame)

            # Collect frames
            video_frames.append(np.expand_dims(frame, axis=0))       # The frame is already RGB, there is no need to convert here.

        
        # Concatenate
        video_frames = np.concatenate(video_frames, axis=0)
        assert(len(video_frames) == self.video_seq_length)

        # Returns
        return video_frames




    def __getitem__(self, idx):
        ''' Get item by idx and pre-process by Resize and Normalize to [0, 1]
        Args:
            idx (int):                  The index to the file in the directory
        Returns:
            return_dict (dict):         video_frames (torch.float32) [-1, 1] and controlnet_condition (torch.float32) [0, 1]
        '''

        # Prepare the text if needed:
        if self.config["use_text"]:
            # Read the file
            file_path = os.path.join(self.video_lists[idx], "lang.txt")
            file = open(file_path, 'r')
            prompt = file.readlines()[0]  # Only read the first line

            if self.config["mix_ambiguous"] and os.path.exists(os.path.join(self.video_lists[idx], "processed_text.txt")):
                # If we don't have this txt file, we skip

                ######################################################## Mix up prompt ########################################################
            
                # Read the file
                file_path = os.path.join(self.video_lists[idx], "processed_text.txt")
                file = open(file_path, 'r')
                prompts = [line for line in file.readlines()]  # Only read the first line

                # Get the componenet
                action = prompts[0][:-1]    
                this = prompts[1][:-1]
                there = prompts[2][:-1]


                random_value = random.random()
                # If less than 0.4, we don't care, just use the most concrete one
                if random_value >= 0.4 and random_value < 0.6:
                    # Mask pick object to "This"
                    prompt = action + " this to " + there
                elif random_value >= 0.6 and random_value < 0.8:
                    # Mask place position to "There"
                    prompt = action + " " + this + " to there"
                elif random_value >= 0.8 and random_value < 1.0:
                    # Just be like "this to there"
                    prompt = action + " this to there"
                
                # print("New prompt is ", prompt)
                ###################################################################################################################################################
            
            # else:
            #     print("We don't have llama processed prompt at ", self.video_lists[idx])
                
        else:
            prompt = ""

        # Tokenize text prompt
        tokenized_prompt = tokenize_captions(prompt, self.tokenizer, self.config)



        # Dataset aug by chance (it is needed to check whether there is any object position words [left|right] in the prompt text)
        flip = False
        if random.random() < self.config["flip_aug_prob"]:
            if self.config["use_text"]:
                if prompt.find("left") == -1 and prompt.find("right") == -1:    # Cannot have position word, like left and right (up and down is ok)
                    flip = True
            else:
                flip = True


        
        # Read frames for different dataset; Currently, we have WebVid / Bridge
        if self.config["dataset_name"] == "Bridge":
            video_frames_raw = self._extract_frame_bridge(idx, flip=flip)
        else:
            raise NotImplementedError("We don't support this dataset loader")


        # Scale [0, 255] -> [-1, 1] if needed
        if self.normalize:
            video_frames = video_frames_raw.astype(np.float32) / 127.5 - 1      # Be careful to cast to float32

        # Transform to Pytorch Tensor in the range [-1, 1]
        video_frames = numpy_to_pt(video_frames)


        # Generate the pairs we need
        intput_dir = self.video_lists[idx]

        # Get the This That point information
        controlnet_condition, reflected_motion_bucket_id, controlnet_image_index, coordinate_values = get_thisthat_sam(self.config, intput_dir, flip=flip)
        controlnet_condition = torch.from_numpy(controlnet_condition)

        # Cast other value to tensor
        reflected_motion_bucket_id = torch.tensor(reflected_motion_bucket_id, dtype=torch.float32) 
        controlnet_image_index = torch.tensor(controlnet_image_index, dtype=torch.int32) 
        coordinate_values = torch.tensor(coordinate_values, dtype=torch.int32)
        

        # The tensor we returned is torch float32. We won't cast here for mixed precision training!
        return {"video_frames" : video_frames, 
                "controlnet_condition" : controlnet_condition, 
                "reflected_motion_bucket_id" : reflected_motion_bucket_id,
                "controlnet_image_index": controlnet_image_index,
                "prompt": tokenized_prompt,
                "coordinate_values": coordinate_values,     # Useless now, but I still passed back
                }

