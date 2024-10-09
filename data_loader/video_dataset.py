import os, sys
import json
import cv2
import math
import shutil
import numpy as np
import random
import collections
from PIL import Image
import torch
from torch.utils.data import Dataset    

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from utils.img_utils import resize_with_antialiasing, numpy_to_pt



def get_video_frames(config, video_frame_path, flip = False):

    video_seq_length = config["video_seq_length"]

    # Calculate needed parameters
    num_frames_input = 0
    for file_name in os.listdir(video_frame_path):
        if file_name.startswith("im_"):
            num_frames_input += 1
    total_frames_needed = video_seq_length
    division_factor = num_frames_input // total_frames_needed
    remain_frames = (num_frames_input % total_frames_needed) - 1    # -1 for adaptation


    # Define the gap
    gaps = [division_factor for _ in range(total_frames_needed-1)]
    for idx in range(remain_frames):
        if idx % 2 == 0:
            gaps[idx//2] += 1      # Start to end order
        else:
            gaps[-1*(1+(idx//2))] += 1   # End to start order


    # Find needed file
    needed_img_path = []
    cur_idx = 0    
    for gap in gaps:
        img_path = os.path.join(video_frame_path, "im_" + str(cur_idx) + ".jpg")
        needed_img_path.append(img_path)

        # Update the idx
        cur_idx += gap
    # Append the last one
    img_path = os.path.join(video_frame_path, "im_" + str(cur_idx) + ".jpg")
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
            print("The exception places is ", img_path)

        # Resize frames
        frame = cv2.resize(frame, (config["width"], config["height"]), interpolation = cv2.INTER_CUBIC)

        # Flip aug
        if flip:
            frame = np.fliplr(frame)

        # Collect frames
        video_frames.append(np.expand_dims(frame, axis=0))       # The frame is already RGB, there is no need to convert here.

    
    # Concatenate
    video_frames = np.concatenate(video_frames, axis=0)
    assert(len(video_frames) == video_seq_length)

    return video_frames



def tokenize_captions(prompt, tokenizer, config, is_train=True):
    '''
        Tokenize text prompt be prepared tokenizer from SD2.1
    '''

    captions = []
    if random.random() < config["empty_prompts_proportion"]:
        captions.append("")
    elif isinstance(prompt, str):
        captions.append(prompt)
    elif isinstance(prompt, (list, np.ndarray)):
        # take a random caption if there are multiple       
        captions.append(random.choice(prompt) if is_train else prompt[0])
    else:
        raise ValueError(
            f"Caption column should contain either strings or lists of strings."
        )
    
    inputs = tokenizer(
        captions, max_length = tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids[0]



class Video_Dataset(Dataset):
    '''
        Video Dataset to load sequential frames for training with needed pre-processing
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
        stats_analysis = collections.defaultdict(int)
        print("Process all files to check valid datasets....")
        for dataset_path in config["dataset_path"]:
            for video_name in sorted(os.listdir(dataset_path)):
                video_path = os.path.join(dataset_path, video_name)
                all_files = os.listdir(video_path)

                
                valid = True
                # Valid check 1: the number of files should be in sequential order
                num_frames_input = 0
                for file_name in os.listdir(video_path):
                    if file_name.startswith("im_"):
                        num_frames_input += 1
                for idx in range(num_frames_input):
                    img_path = 'im_' + str(idx) + '.jpg'
                    if img_path not in all_files:            # Should be sequential existing
                        valid = False
                        stats_analysis["incomplete_img"] += 1
                        break


                # Valid check 1.5: the number of files must be longer than video_seq_length and less than self.config["acceleration_tolerance"]*self.config["video_seq_length"]
                if num_frames_input < self.config["video_seq_length"]:
                    stats_analysis["too_little_frames"] += 1
                    valid = False
                if num_frames_input > self.config["acceleration_tolerance"] * self.config["video_seq_length"]:
                    stats_analysis["too_many_frames"] += 1
                    valid = False

                if not valid:   # SpeedUp so set in the middle here
                    continue


                # Valid check 2: language if needed
                if config["use_text"] and not os.path.exists(os.path.join(dataset_path, video_name, "lang.txt")):
                    stats_analysis["no_lang_txt"] += 1
                    valid = False


                # Valid check 3: motion if needed
                if config["motion_bucket_id"] is None:
                    flow_path = os.path.join(dataset_path, video_name, "flow.txt")
                    if "flow.txt" not in all_files:
                        stats_analysis["no_flow_txt"] += 1
                        valid = False
                    else:
                        file = open(flow_path, 'r')
                        info = file.readlines()
                        if len(info) == 0:
                            stats_analysis["no_flow_txt"] += 1
                            valid = False


                if valid:
                    self.video_lists.append(video_path)
        print("stats_analysis is ", stats_analysis)
        print("Valid dataset length is ", len(self.video_lists))

        
    def __len__(self):
        return len(self.video_lists)
    


    def _get_motion_value(self, sub_folder_path):
        ''' Read the motion value from the flow.txt file prepared; preprocess the flow to accelerate
        '''

        # Read the flow.txt
        flow_path = os.path.join(sub_folder_path, 'flow.txt')       
        file = open(flow_path, 'r')
        info = file.readlines()
        per_video_movement = float(info[0][:-2])

        # Map the raw reflected_motion_bucket_id to target range based on the number of images have
        num_frames_input = 0
        for file_name in os.listdir(sub_folder_path):   # num_frames_input is the total number of files with name begin with im_
            if file_name.startswith("im_"):
                num_frames_input += 1

        # Correct the value based on the number of frames relative to video_seq_length
        per_video_movement_correct = per_video_movement * (num_frames_input/self.config["video_seq_length"])  

        # Map from one Normal Distribution to another Normal Distribution
        z = (per_video_movement_correct - self.config["dataset_motion_mean"]) / (self.config["dataset_motion_std"] + 0.001)
        reflected_motion_bucket_id = int((z * self.config["svd_motion_std"]) + self.config["svd_motion_mean"])
        

        print("We map " + str(per_video_movement) + " to " + str(per_video_movement_correct) + " by length " + str(num_frames_input) + " to bucket_id of " + str(reflected_motion_bucket_id))
        return reflected_motion_bucket_id   
    


    def __getitem__(self, idx):
        ''' Get item by idx and pre-process by Resize and Normalize to [0, 1]
        Args:
            idx (int):                  The index to the file in the directory
        Returns:
            video_frames (torch.float32):           The Pytorch tensor format of obtained frames (max: 1.0; min: 0.0)
            reflected_motion_bucket_id (tensor):    Motion value is there is optical flow provided, else they are fixed value from config
            prompt (tensor):                        Tokenized text
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


        # Read frames for different datasets; Currently, we have WebVid / Bridge
        if self.config["dataset_name"] == "Bridge":
            video_frames = get_video_frames(self.config, self.video_lists[idx], flip=flip)
        else:
            raise NotImplementedError("We don't support this dataset loader")


        # Scale [0, 255] -> [-1, 1]
        if self.normalize:
            video_frames = video_frames.astype(np.float32) / 127.5 - 1      # Be careful to cast to float32

        # Transform to Pytorch Tensor in the range [-1, 1]
        video_frames = numpy_to_pt(video_frames)
        # print("length of input frames has ", len(video_frames))


        # Get the motion value based on the optical flow
        if self.config["motion_bucket_id"] is None:
            reflected_motion_bucket_id = self._get_motion_value(self.video_lists[idx])   
        else:
            reflected_motion_bucket_id = self.config["motion_bucket_id"]

            
        # The tensor we returned is torch float32. We won't cast here for mixed precision training!
        return {
                "video_frames" : video_frames, 
                "reflected_motion_bucket_id" : reflected_motion_bucket_id,
                "prompt": tokenized_prompt,
                }