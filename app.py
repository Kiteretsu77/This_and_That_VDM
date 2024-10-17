# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

import os, shutil, sys
import urllib.request
import argparse
import imageio
import math
import cv2
import collections
import numpy as np
import gradio as gr
from PIL import Image

import torch
from pathlib import Path
from omegaconf import OmegaConf
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import (
    AutoencoderKLTemporalDecoder,
    DDPMScheduler,
)
from diffusers.utils import check_min_version, is_wandb_available, load_image, export_to_video
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PretrainedConfig    


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from train_code.train_svd import import_pretrained_text_encoder
from data_loader.video_dataset import tokenize_captions
from data_loader.video_this_that_dataset import get_thisthat_sam
from svd.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from svd.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from svd.temporal_controlnet import ControlNetModel
from svd.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionControlNetPipeline
from utils.optical_flow_utils import bivariate_Gaussian


# For the 2D dilation
blur_kernel = bivariate_Gaussian(99, 10, 10, 0, grid = None, isotropic = True)


# Import 
# LENGTH=480 # length of the square area displaying/editing images
HEIGHT = 256 
WIDTH = 384  


MARKDOWN = \
    """
    ## <p style='text-align: center'> This&That: Language-Gesture Controlled Video Generation for Robot Planning </p>
    
    [GitHub](https://github.com/Kiteretsu77/This_and_That_VDM) | [Paper](http://arxiv.org/abs/2407.05530) | [Webpage](https://cfeng16.github.io/this-and-that/)

    This&That is a Robotics scenario (Bridge-dataset-based for this repo) Language-Gesture-Image-conditioned Video Generation Model for Robot Planning.

    This Demo is on the Video Diffusion Model part.
    Only GestureNet is provided in this Gradio Demo, you can check the full test code for all pretrained weight available.

    ### Note: The index we put the gesture point by default here is [4, 10] (5th and 11th) for two gesture points or [4] (5th) for one gesture point.
    ### Note: The resolution now only support is 256x384.
    ### Note: Click "Clear All" to restart everything; Click "Undo Point" to cancel the point you put
    ### Note: The first run may be long. Click "Clear All" for each run is the safest choice.
    
    If **This&That** is helpful, please help star the [GitHub Repo](https://github.com/Kiteretsu77/This_and_That_VDM). Thanks! 
    
    """


def store_img(img):

    # when new image is uploaded, `selected_points` should be empty
    return img, []



def clear_all():
    return None, \
        gr.Image(value=None, height=HEIGHT, width=WIDTH, interactive=False), \
        None, []    # selected points


def undo_points(original_image):
    img = original_image.copy()
    return img, []


# User click the image to get points, and show the points on the image [From https://github.com/Yujun-Shi/DragDiffusion]
def get_points(img, original_image, sel_pix, evt: gr.SelectData):

    # collect the selected point
    sel_pix.append(evt.index)

    if len(sel_pix) > 2:
        raise gr.Error("We only at most support two points")

    if original_image is None:
        original_image = img.copy()

    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 10, (0, 255, 0), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        # if len(points) == 2:
        #     cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
        #     points = []

    return [img if isinstance(img, np.ndarray) else np.array(img), original_image]


def gesturenet_inference(ref_image, prompt, selected_points):

    # Check some paramter, must have prompt and selected points
    if prompt == "" or prompt is None:
        raise gr.Error("Please input text prompt")
    if selected_points == []:
        raise gr.Error("Please click one/two points in the Image")

    # Prepare the setting
    frame_idxs = [4, 10]
    use_ambiguous_prompt = False
    model_type = "GestureNet"
    huggingface_pretrained_path = "HikariDawn/This-and-That-1.1"

    print("Text prompt is ", prompt)

    # Prepare tmp folder
    store_folder_name = "tmp"
    if os.path.exists(store_folder_name):
        shutil.rmtree(store_folder_name)
    os.makedirs(store_folder_name)


    # Read the yaml setting files (Very important for loading hyperparamters needed)
    if not os.path.exists(huggingface_pretrained_path):
        yaml_download_path = hf_hub_download(repo_id=huggingface_pretrained_path, subfolder="unet", filename="train_image2video.yaml")
        if model_type == "GestureNet":
            yaml_download_path = hf_hub_download(repo_id=huggingface_pretrained_path, subfolder="gesturenet", filename="train_image2video_gesturenet.yaml")
    else:   # If the path is a local path we can concatenate it here
        yaml_download_path = os.path.join(huggingface_pretrained_path, "unet", "train_image2video.yaml")
        if model_type == "GestureNet":
            yaml_download_path = os.path.join(huggingface_pretrained_path, "gesturenet", "train_image2video_gesturenet.yaml")

    # Load the config
    assert(os.path.exists(yaml_download_path))
    config = OmegaConf.load(yaml_download_path)


    ################################################ Prepare vae, unet, image_encoder Same as before #################################################################
    print("Prepare the pretrained model")
    accelerator = Accelerator(
        gradient_accumulation_steps = config["gradient_accumulation_steps"],
        mixed_precision = config["mixed_precision"],
        log_with = config["report_to"],
        project_config = ProjectConfiguration(project_dir=config["output_dir"], logging_dir=Path(config["output_dir"], config["logging_name"])),
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        config["pretrained_model_name_or_path"], subfolder="feature_extractor", revision=None
    )   # This instance has now weight, they are just seeting file
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        config["pretrained_model_name_or_path"], subfolder="image_encoder", revision=None, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        config["pretrained_model_name_or_path"], subfolder="vae", revision=None, variant="fp16"
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        huggingface_pretrained_path, 
        subfolder = "unet", 
        low_cpu_mem_usage = True,
        # variant = "fp16",
    )
    

    # For text ..............................................
    tokenizer = AutoTokenizer.from_pretrained(
        config["pretrained_tokenizer_name_or_path"],
        subfolder = "tokenizer",
        revision = None,
        use_fast = False,
    )
    # Clip Text Encoder
    text_encoder_cls = import_pretrained_text_encoder(config["pretrained_tokenizer_name_or_path"], revision=None)
    text_encoder = text_encoder_cls.from_pretrained(config["pretrained_tokenizer_name_or_path"], subfolder = "text_encoder", revision = None, variant = None)


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae + image_encoder to gpu and cast to weight_dtype
    vae.requires_grad_(False)       
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)  # Will switch back at the end
    text_encoder.requires_grad_(False)

    # Move to accelerator
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # For GestureNet
    if model_type == "GestureNet":
        unet.to(accelerator.device, dtype=weight_dtype)     # There is no need to cast unet in unet training, only needed in controlnet one 

        # Handle the Controlnet first from UNet
        gesturenet = ControlNetModel.from_pretrained(
                                                        huggingface_pretrained_path, 
                                                        subfolder = "gesturenet", 
                                                        low_cpu_mem_usage = True,
                                                        variant = None,
                                                    )

        gesturenet.requires_grad_(False)
        gesturenet.to(accelerator.device)
    ##############################################################################################################################################################




    # Init the pipeline
    pipeline = StableVideoDiffusionControlNetPipeline.from_pretrained(
        config["pretrained_model_name_or_path"],        # Still based on regular SVD config
        vae = vae,
        image_encoder = image_encoder,
        unet = unet,
        revision = None,    # Set None directly now
        torch_dtype = weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)



    ############################## Prepare and Process the condition here ##############################
    org_height, org_width, _ = ref_image.shape
    ref_image_pil = Image.fromarray(ref_image)
    ref_image_pil = ref_image_pil.resize((config["width"], config["height"]))


    # Initial the optical flow format we want
    gesture_condition_img = np.zeros((config["video_seq_length"], config["conditioning_channels"], config["height"], config["width"]), dtype=np.float32)  # The last image should be empty

    # Handle the selected points to the condition we want
    for point_idx, point in enumerate(selected_points):

        frame_idx = frame_idxs[point_idx]
        horizontal, vertical = point

        # Init the base image
        base_img = np.zeros((org_height, org_width, 3)).astype(np.float32)      # Use the original image size
        base_img.fill(255)

        # Draw square around the target position
        dot_range = 10       # Diameter
        for i in range(-1*dot_range, dot_range+1):
            for j in range(-1*dot_range, dot_range+1):
                dil_vertical, dil_horizontal = vertical + i, horizontal + j
                if (0 <= dil_vertical and dil_vertical < base_img.shape[0]) and (0 <= dil_horizontal and dil_horizontal < base_img.shape[1]):
                    if point_idx == 0:
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

        # Channel Transform and Range Shift
        if config["conditioning_channels"] == 3:
            # Map to [0, 1] range 
            base_img = base_img / 255.0         

        else:
            raise NotImplementedError()

        # ReOrganize shape
        base_img = base_img.transpose(2, 0, 1)  # hwc -> chw

        # Write base img based on frame_idx
        gesture_condition_img[frame_idx] = base_img        # Only the first frame, the rest is 0 initialized


    ####################################################################################################

    # Use the same tokenize process as the dataset preparation stage
    tokenized_prompt = tokenize_captions(prompt, tokenizer, config, is_train=False).unsqueeze(0).to(accelerator.device)    # Use unsqueeze to expand dim
    


    # Call the pipeline
    with torch.autocast("cuda"):
        frames = pipeline(
                            image = ref_image_pil, 
                            condition_img = gesture_condition_img,       # numpy [0,1] range
                            controlnet = accelerator.unwrap_model(gesturenet),
                            prompt = tokenized_prompt,
                            use_text = config["use_text"],
                            text_encoder = text_encoder,
                            height = config["height"],
                            width = config["width"],
                            num_frames = config["video_seq_length"], 
                            decode_chunk_size = 8, 
                            motion_bucket_id = 200,
                            # controlnet_image_index = controlnet_image_index,
                            # coordinate_values = coordinate_values,
                            num_inference_steps = config["num_inference_steps"],
                            max_guidance_scale = config["inference_max_guidance_scale"],
                            fps = 7,
                            use_instructpix2pix = config["use_instructpix2pix"],
                            noise_aug_strength = config["inference_noise_aug_strength"],
                            controlnet_conditioning_scale = config["outer_conditioning_scale"],
                            inner_conditioning_scale = config["inner_conditioning_scale"],
                            guess_mode = config["inference_guess_mode"],        # False in inference
                            image_guidance_scale = config["image_guidance_scale"],
                        ).frames[0]    

    # Save frames
    video_file_path = os.path.join(store_folder_name, "tmp.mp4")
    writer = imageio.get_writer(video_file_path, fps=4)
    for idx, frame in enumerate(frames):
        frame.save(os.path.join(store_folder_name, str(idx)+".png"))
        writer.append_data(cv2.cvtColor(cv2.imread(os.path.join(store_folder_name, str(idx)+".png")), cv2.COLOR_BGR2RGB))
    writer.close()



    # Cleaning process
    del pipeline
    torch.cuda.empty_cache()

    return gr.update(value=video_file_path, width=config["width"], height=config["height"])   # Return resuly based on the need



if __name__ == '__main__':


    # Gradio demo part
    with gr.Blocks() as demo:
        # layout definition
        with gr.Row():
            gr.Markdown(MARKDOWN)

        # UI components for editing real images
        with gr.Row(elem_classes=["container"]):
            selected_points = gr.State([]) # store points
            original_image = gr.State(value=None) # store original input image
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""<p style="text-align: center; font-size: 30px">Click two Points</p>""")
                    input_image = gr.Image(label="Input Image", height=HEIGHT, width=WIDTH, interactive=False, elem_id="input_img")
                    # gr.Image(type="numpy", label="Click Points", height=HEIGHT, width=WIDTH, interactive=False) # for points clicking
                    undo_button = gr.Button("Undo point")

                    # Text prompt
                    with gr.Row():
                        prompt = gr.Textbox(label="Text Prompt")


                with gr.Column():
                    gr.Markdown("""<p style="text-align: center; font-size: 30px">Results</p>""")
                    frames = gr.Video(value=None, label="Generate Video", show_label=True, height=HEIGHT, width=WIDTH)
                    with gr.Row():
                        run_button = gr.Button("Run")
                        clear_all_button = gr.Button("Clear All")

            


            # with gr.Tab("Base Model Config"):
            #     with gr.Row():
            #         local_models_dir = 'local_pretrained_models'
            #         local_models_choice = \
            #             [os.path.join(local_models_dir,d) for d in os.listdir(local_models_dir) if os.path.isdir(os.path.join(local_models_dir,d))]
            #         model_path = gr.Dropdown(value="runwayml/stable-diffusion-v1-5",
            #             label="Diffusion Model Path",
            #             choices=[
            #                 "runwayml/stable-diffusion-v1-5",
            #                 "gsdf/Counterfeit-V2.5",
            #                 "stablediffusionapi/anything-v5",
            #                 "SG161222/Realistic_Vision_V2.0",
            #             ] + local_models_choice
            #         )
            #         vae_path = gr.Dropdown(value="default",
            #             label="VAE choice",
            #             choices=["default",
            #             "stabilityai/sd-vae-ft-mse"] + local_models_choice
            #         )

        # Examples
        with gr.Row(elem_classes=["container"]):
            gr.Examples(
                [
                    ["__assets__/Bridge_example/Task1_v1_511/im_0.jpg", "take this to there"],
                    ["__assets__/Bridge_example/Task2_v2_164/im_0.jpg", "put this to there"],
                    ["__assets__/Bridge_example/Task3_v2_490/im_0.jpg", "fold this"],
                    ["__assets__/Bridge_example/Task4_v2_119/im_0.jpg", "open this"],

                    # ["__assets__/0.jpg", "take this to there"],
                    ["__assets__/91.jpg", "take this to there"],
                    ["__assets__/156.jpg", "take this to there"],
                    # ["__assets__/274.jpg", "take this to there"],
                    ["__assets__/375.jpg", "take this to there"],
                    # ["__assets__/551.jpg", "take this to there"],
                ],
                [input_image, prompt, selected_points],
            )




        ####################################### Event Definition #######################################

        # Draw the points
        input_image.select(
            get_points,
            [input_image, original_image, selected_points],
            [input_image, original_image],
        )

        # Clean the points
        undo_button.click(
            undo_points,
            [original_image],
            [input_image, selected_points],
        )

        run_button.click(
            gesturenet_inference,
            inputs = [
                # vae, unet, gesturenet, image_encoder, text_encoder, tokenizer,
                original_image, prompt, selected_points, 
                # frame_idxs,
                # config, accelerator, weight_dtype
             ],
            outputs = [frames]
        )

        clear_all_button.click(
            clear_all,
            [],
            outputs = [original_image, input_image, prompt, selected_points],
        )


    demo.queue().launch(share=True, debug=True)
