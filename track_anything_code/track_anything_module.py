import gradio as gr
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
import imageio
try: 
    from mmcv.cnn import ConvModule
except:
    os.system("mim install mmcv")


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from track_anything_code.model import TrackingAnything




def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--mask_save', default=False)
    args = parser.parse_args()

    if args.debug:
        print(args)
    return args 


# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath

def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"False",
    }
    return prompt



# extract frames from upload video
def get_frames_from_video(video_path, video_state, model):
    """ Extract video information based on the input
    Args:
        video_path: str
        timestamp: float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """ 
    frames = []
    user_name = time.time()
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    image_size = (frames[0].shape[0],frames[0].shape[1]) 

    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": fps
        }
    model.samcontroler.sam_controler.reset_image() 
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    return video_state, video_state["origin_images"][0]



def run_example(example):
    return video_input
# get the select frame from gradio slider
def select_template(image_selection_slider, video_state, interactive_state, mask_dropdown):

    # images = video_state[1]
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    if mask_dropdown:
        print("ok")
    operation_log = [("",""), ("Select frame {}. Try click image and add mask for tracking.".format(image_selection_slider),"Normal")]


    return video_state["painted_images"][image_selection_slider], video_state, interactive_state, operation_log

# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider
    operation_log = [("",""),("Set the tracking finish at frame {}".format(track_pause_number_slider),"Normal")]

    return video_state["painted_images"][track_pause_number_slider],interactive_state, operation_log

def get_resize_ratio(resize_ratio_slider, interactive_state):
    interactive_state["resize_ratio"] = resize_ratio_slider

    return interactive_state

# use sam to get the mask
def sam_refine(model, video_state, point_prompt, click_state, interactive_state, point_cord):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(point_cord[0], point_cord[1]) # Height and Width
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(point_cord[0], point_cord[1])
        interactive_state["negative_click_times"] += 1
    
    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click( 
                                                      image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=False,  # False by default
                                                      )
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    operation_log = [("",""), ("Use SAM for segment. You can try add positive and negative points by clicking. Or press Clear clicks button to refresh the image. Press Add mask button when you are satisfied with the segment","Normal")]
    return painted_image, video_state, interactive_state, operation_log


def clear_click(video_state, click_state):
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [("",""), ("Clear points history and refresh the image.","Normal")]
    return template_frame, click_state, operation_log

def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("",""), ("Remove all mask, please add new masks","Normal")]
    return interactive_state, gr.update(choices=[],value=[]), operation_log



# tracking vos
def vos_tracking_video(model, output_path, video_state, interactive_state, mask_dropdown):
    operation_log = [("",""), ("Track the selected masks, and then you can select the masks for inpainting.","Normal")]
    model.xmem.clear_memory()
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]

    # operation error
    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1
        operation_log = [("Error! Please add at least one mask to track by clicking the left image.","Error"), ("","")]
        # return video_output, video_state, interactive_state, operation_error
    masks, logits, painted_images = model.generator(images=following_frames, template_mask=template_mask)
    # clear GPU memory
    model.xmem.clear_memory()

    if interactive_state["track_end_number"]: 
        video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = masks
        video_state["logits"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = logits
        video_state["painted_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = painted_images
    else:
        video_state["masks"][video_state["select_frame_number"]:] = masks
        video_state["logits"][video_state["select_frame_number"]:] = logits
        video_state["painted_images"][video_state["select_frame_number"]:] = painted_images

    generate_video_from_frames(video_state["painted_images"], output_path=output_path, fps=fps) # import video_input to name the output video
    interactive_state["inference_times"] += 1
    
    print("For generating this tracking result, inference times: {}, click times: {}, positive: {}, negative: {}".format(interactive_state["inference_times"], 
                                                                                                                                           interactive_state["positive_click_times"]+interactive_state["negative_click_times"],
                                                                                                                                           interactive_state["positive_click_times"],
                                                                                                                                        interactive_state["negative_click_times"]))

    #### shanggao code for mask save
    if interactive_state["mask_save"]:      # May not need to use this branch
        if not os.path.exists('./result/mask/{}'.format(video_state["video_name"].split('.')[0])):
            os.makedirs('./result/mask/{}'.format(video_state["video_name"].split('.')[0]))
        i = 0
        print("save mask")
        for mask in video_state["masks"]:
            np.save(os.path.join('./result/mask/{}'.format(video_state["video_name"].split('.')[0]), '{:05d}.npy'.format(i)), mask)
            i+=1
        # save_mask(video_state["masks"], video_state["video_name"])
    #### shanggao code for mask save
    return video_state, video_state, interactive_state, operation_log

# extracting masks from mask_dropdown
# def extract_sole_mask(video_state, mask_dropdown):
#     combined_masks = 
#     unique_masks = np.unique(combined_masks)
#     return 0 

# inpaint 
def inpaint_video(video_state, interactive_state, mask_dropdown):
    operation_log = [("",""), ("Removed the selected masks.","Normal")]

    frames = np.asarray(video_state["origin_images"])
    fps = video_state["fps"]
    inpaint_masks = np.asarray(video_state["masks"])
    if len(mask_dropdown) == 0:
        mask_dropdown = ["mask_001"]
    mask_dropdown.sort()
    # convert mask_dropdown to mask numbers
    inpaint_mask_numbers = [int(mask_dropdown[i].split("_")[1]) for i in range(len(mask_dropdown))]
    # interate through all masks and remove the masks that are not in mask_dropdown
    unique_masks = np.unique(inpaint_masks)
    num_masks = len(unique_masks) - 1
    for i in range(1, num_masks + 1):
        if i in inpaint_mask_numbers:
            continue
        inpaint_masks[inpaint_masks==i] = 0
    # inpaint for videos

    try:
        inpainted_frames = model.baseinpainter.inpaint(frames, inpaint_masks, ratio=interactive_state["resize_ratio"])   # numpy array, T, H, W, 3
    except:
        operation_log = [("Error! You are trying to inpaint without masks input. Please track the selected mask first, and then press inpaint. If VRAM exceeded, please use the resize ratio to scaling down the image size.","Error"), ("","")]
        inpainted_frames = video_state["origin_images"]
    video_output = generate_video_from_frames(inpainted_frames, output_path="./result/inpaint/{}".format(video_state["video_name"]), fps=fps) # import video_input to name the output video

    return video_output, operation_log


# generate video after vos inference
def generate_video_from_frames(frames, output_path=None, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): If provided, it is the path to save the generated video. Else, we won't store it
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    
    # frames = torch.from_numpy(np.asarray(frames))
    imageio.mimsave(output_path, frames)
    # return output_path




if __name__ == "__main__":
    # args, defined in track_anything.py
    args = parse_augment()

    # check and download checkpoints if needed
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    xmem_checkpoint = "XMem-s012.pth"
    xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"


    folder ="./pretrained"
    SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
    xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
    args.device = "cuda"      # Any GPU is ok

    # initialize sam, xmem, e2fgvi models
    model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, args)
