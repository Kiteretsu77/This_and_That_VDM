'''
    This file is to test UNet and GestureNet.
'''

import os, shutil, sys
import urllib.request
import argparse
import imageio
import math
import cv2
from PIL import Image
import collections
import numpy as np

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



# Seed
# torch.manual_seed(42)
# np.random.seed(42)


def unet_inference(vae, unet, image_encoder, text_encoder, tokenizer, config, accelerator, weight_dtype, step, 
                        parent_store_folder = None, force_close_flip = False, use_ambiguous_prompt=False):

    # Init
    validation_source_folder = config["validation_img_folder"] 
    

    # Init the pipeline
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        config["pretrained_model_name_or_path"],
        vae = accelerator.unwrap_model(vae),
        image_encoder = accelerator.unwrap_model(image_encoder),
        unet = accelerator.unwrap_model(unet),
        revision = None,    # Set None directly now
        torch_dtype = weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)


    # Process all image in the folder
    frames_collection = []
    for image_name in sorted(os.listdir(validation_source_folder)):
        if accelerator.is_main_process:
            if parent_store_folder is None:
                validation_store_folder = os.path.join(config["validation_store_folder"] + "_" + config["scheduler"], "step_" + str(step), image_name)
            else:
                validation_store_folder = os.path.join(parent_store_folder, image_name)
                
            if os.path.exists(validation_store_folder):
                shutil.rmtree(validation_store_folder)
            os.makedirs(validation_store_folder)

        image_path = os.path.join(validation_source_folder, image_name, 'im_0.jpg')
        ref_image = load_image(image_path)
        ref_image = ref_image.resize((config["width"], config["height"]))
        
        
        # Decide the motion score in SVD (mostly what we use is fix value now)
        if config["motion_bucket_id"] is None:      
            raise NotImplementedError("We need a fixed motion_bucket_id in the config")
        else:
            reflected_motion_bucket_id = config["motion_bucket_id"]
        # print("Inference Motion Bucket ID is ", reflected_motion_bucket_id)


        # Prepare text prompt
        if config["use_text"]:
            # Read the file
            file_path = os.path.join(validation_source_folder, image_name, "lang.txt")
            file = open(file_path, 'r')
            prompt = file.readlines()[0]  # Only read the first line
            if use_ambiguous_prompt:
                prompt = prompt.split(" ")[0] + " this to there"
                print("We are creating ambiguous prompt, which is: ", prompt)
        else:
            prompt = ""
        # Use the same tokenize process as the dataset preparation stage
        tokenized_prompt = tokenize_captions(prompt, tokenizer, config, is_train=False).unsqueeze(0).to(accelerator.device)    # Use unsqueeze to expand dim

        # Store the prompt for the sanity check
        f = open(os.path.join(validation_store_folder, "lang_cond.txt"), "a")
        f.write(prompt)
        f.close()


        # Flip the image by chance (it is needed to check whether there is any object position words [left|right] in the prompt text)
        flip = False
        if not force_close_flip:        # force_close_flip is True in testing time; else, we cannot match in the same standard
            if random.random() < config["flip_aug_prob"]:
                if config["use_text"]:
                    if prompt.find("left") == -1 and prompt.find("right") == -1:    # Cannot have position word, like left and right (up and down is ok)
                        flip = True
                else:
                    flip = True
            if flip:
                print("Use flip in validation!")
                ref_image = ref_image.transpose(Image.FLIP_LEFT_RIGHT)


        # Call the model for inference
        with torch.autocast("cuda"):
            frames = pipeline(
                                ref_image, 
                                tokenized_prompt,
                                config["use_text"],
                                text_encoder,
                                height = config["height"],
                                width = config["width"],
                                num_frames = config["video_seq_length"], 
                                num_inference_steps = config["num_inference_steps"],
                                decode_chunk_size = 8, 
                                motion_bucket_id = reflected_motion_bucket_id,
                                fps = 7,
                                noise_aug_strength = config["inference_noise_aug_strength"],
                              ).frames[0]     

        # Store the frames
        # breakpoint()
        for idx, frame in enumerate(frames):
            frame.save(os.path.join(validation_store_folder, str(idx)+".png"))
        imageio.mimsave(os.path.join(validation_store_folder, 'combined.gif'), frames)      # gif storage quality is not high, recommend to check png images

        frames_collection.append(frames)


    # Cleaning process
    del pipeline
    torch.cuda.empty_cache()

    return frames_collection   # Return resuly based on the need


    
def gesturenet_inference(vae, unet, controlnet, image_encoder, text_encoder, tokenizer, config, accelerator, weight_dtype, step, 
                        parent_store_folder=None, force_close_flip=False, use_ambiguous_prompt=False):


    # Init
    validation_source_folder = config["validation_img_folder"] 
    

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


    # Process all image in the folder
    frames_collection = []
    for image_name in sorted(os.listdir(validation_source_folder)):
        if accelerator.is_main_process:
            if parent_store_folder is None:
                validation_store_folder = os.path.join(config["validation_store_folder"] + "_" + config["scheduler"], "step_" + str(step), image_name)
            else:
                validation_store_folder = os.path.join(parent_store_folder, image_name)
                
            if os.path.exists(validation_store_folder):
                shutil.rmtree(validation_store_folder)
            os.makedirs(validation_store_folder)

        image_path = os.path.join(validation_source_folder, image_name, 'im_0.jpg')
        ref_image = load_image(image_path)      # [0, 255] Range
        ref_image = ref_image.resize((config["width"], config["height"]))


        # Prepare text prompt
        if config["use_text"]:
            # Read the file
            file_path = os.path.join(validation_source_folder, image_name, "lang.txt")
            file = open(file_path, 'r')
            prompt = file.readlines()[0]  # Only read the first line
            if use_ambiguous_prompt:
                prompt = prompt.split(" ")[0] + " this to there"
                print("We are creating ambiguous prompt, which is: ", prompt)
        else:
            prompt = ""
        # Use the same tokenize process as the dataset preparation stage
        tokenized_prompt = tokenize_captions(prompt, tokenizer, config, is_train=False).unsqueeze(0).to(accelerator.device)    # Use unsqueeze to expand dim

        # Store the prompt for the sanity check
        f = open(os.path.join(validation_store_folder, "lang_cond.txt"), "a")
        f.write(prompt)
        f.close()

        # Flip the image by chance (it is needed to check whether there is any object position words [left|right] in the prompt text)
        flip = False
        if not force_close_flip:    # force_close_flip is True in testing time; else, we cannot match in the same standard
            if random.random() < config["flip_aug_prob"]:
                if config["use_text"]:
                    if prompt.find("left") == -1 and prompt.find("right") == -1:    # Cannot have position word, like left and right (up and down is ok)
                        flip = True
                else:
                    flip = True
            if flip:
                print("Use flip in validation!")
                ref_image = ref_image.transpose(Image.FLIP_LEFT_RIGHT)


        if config["data_loader_type"] == "thisthat":
            condition_img, reflected_motion_bucket_id, controlnet_image_index, coordinate_values = get_thisthat_sam(config, 
                                                                                                                    os.path.join(validation_source_folder, image_name),
                                                                                                                    flip = flip, 
                                                                                                                    store_dir = validation_store_folder,
                                                                                                                    verbose = True)
        else:
            raise NotImplementedError("We don't support such data loader type")



        # Call the pipeline
        with torch.autocast("cuda"):
            frames = pipeline(
                                image = ref_image, 
                                condition_img = condition_img,       # numpy [0,1] range
                                controlnet = accelerator.unwrap_model(controlnet),
                                prompt = tokenized_prompt,
                                use_text = config["use_text"],
                                text_encoder = text_encoder,
                                height = config["height"],
                                width = config["width"],
                                num_frames = config["video_seq_length"], 
                                decode_chunk_size = 8, 
                                motion_bucket_id = reflected_motion_bucket_id,
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

        for idx, frame in enumerate(frames):
            frame.save(os.path.join(validation_store_folder, str(idx)+".png"))
        imageio.mimsave(os.path.join(validation_store_folder, 'combined.gif'), frames, duration=0.05)

        frames_collection.append(frames)


    # Cleaning process
    del pipeline
    torch.cuda.empty_cache()

    return frames_collection   # Return resuly based on the need



def execute_inference(huggingface_pretrained_path, model_type, validation_path, parent_store_folder, use_ambiguous_prompt):

    # Check path
    if os.path.exists(parent_store_folder):
        shutil.rmtree(parent_store_folder)
    os.makedirs(parent_store_folder)


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
    base_config = OmegaConf.load(yaml_download_path)


    # Other Settings
    base_config["validation_img_folder"] = validation_path      



    ################################################ Prepare vae, unet, image_encoder Same as before #################################################################
    accelerator = Accelerator(
        gradient_accumulation_steps = base_config["gradient_accumulation_steps"],
        mixed_precision = base_config["mixed_precision"],
        log_with = base_config["report_to"],
        project_config = ProjectConfiguration(project_dir=base_config["output_dir"], logging_dir=Path(base_config["output_dir"], base_config["logging_name"])),
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        base_config["pretrained_model_name_or_path"], subfolder="feature_extractor", revision=None
    )   # This instance has now weight, they are just seeting file
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_config["pretrained_model_name_or_path"], subfolder="image_encoder", revision=None, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        base_config["pretrained_model_name_or_path"], subfolder="vae", revision=None, variant="fp16"
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        huggingface_pretrained_path, 
        subfolder = "unet", 
        low_cpu_mem_usage = True,
        # variant = "fp16",
    )
    

    # For text ..............................................
    tokenizer = AutoTokenizer.from_pretrained(
        base_config["pretrained_tokenizer_name_or_path"],
        subfolder = "tokenizer",
        revision = None,
        use_fast = False,
    )
    # Clip Text Encoder
    text_encoder_cls = import_pretrained_text_encoder(base_config["pretrained_tokenizer_name_or_path"], revision=None)
    text_encoder = text_encoder_cls.from_pretrained(base_config["pretrained_tokenizer_name_or_path"], subfolder = "text_encoder", revision = None, variant = None)


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



    ############################################################### Execution #####################################################################################

    # Prepare the iterative calling
    if model_type == "UNet":
        generated_frames = unet_inference(
                                            vae, unet, image_encoder, text_encoder, tokenizer, 
                                            base_config, accelerator, weight_dtype, step="", 
                                            parent_store_folder=parent_store_folder, force_close_flip = True,
                                            use_ambiguous_prompt = use_ambiguous_prompt,
                                        )
    
    elif model_type == "GestureNet":
        generated_frames = gesturenet_inference(
                                                vae, unet, gesturenet, image_encoder, text_encoder, tokenizer, 
                                                base_config, accelerator, weight_dtype, step="",
                                                parent_store_folder=parent_store_folder, force_close_flip = True,
                                                use_ambiguous_prompt = use_ambiguous_prompt,
                                            )

    else:
        raise NotImplementedError("model_type is no the predefined choices we provide!")

    ################################################################################################################################################################


    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="GestureNet",
        help="\"UNet\" for VL (vision language) / \"GestureNet\" for VGL (vision gesture language)",
    )
    parser.add_argument(
        "--huggingface_pretrained_path",
        type=str,
        default="HikariDawn/This-and-That-1.1",
        help="Path to the unet folder path.",
    )
    parser.add_argument(
        "--validation_path",
        type=str,
        default="__assets__/Bridge_example/",
        help="Sample dataset path, default to the Bridge example.",
    )
    parser.add_argument(
        "--parent_store_folder",
        type=str,
        default="generated_results/",
        help="Path to the store result folder.",
    )
    parser.add_argument(
        "--use_ambiguous_prompt",
        type=str,
        default=False,
        help="Whether we will use action verb + \"this to there\" ambgiguous prompt combo.",
    )
    args = parser.parse_args()


    # File Setting
    model_type = args.model_type
    huggingface_pretrained_path = args.huggingface_pretrained_path
    # validation_path Needs to have subforder for each instance.
    # Each instance requries "im_0.jpg" for the first image; data.txt for the gesture position; lang.txt for the language
    validation_path = args.validation_path      
    parent_store_folder = args.parent_store_folder
    use_ambiguous_prompt = args.use_ambiguous_prompt      


    # Execution
    execute_inference(huggingface_pretrained_path, model_type, validation_path, parent_store_folder, use_ambiguous_prompt)

    
    print("All finished!!!")


