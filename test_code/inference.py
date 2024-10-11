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
from transformers import AutoTokenizer, PretrainedConfig    

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from train_code.train_svd import log_validation as unet_log_validation
from train_code.train_csvd import log_validation as controlnet_log_validation
from train_code.train_svd import import_pretrained_text_encoder
from svd.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from svd.temporal_controlnet import ControlNetModel

# Seed
# torch.manual_seed(42)
# np.random.seed(42)

def execute_inference(huggingface_pretrained_path, model_type, validation_path, parent_store_folder, use_ambiguous_prompt):

    # Check path
    if os.path.exists(parent_store_folder):
        shutil.rmtree(parent_store_folder)
    os.makedirs(parent_store_folder)


    # Hard-Code the remote url
    yaml_url = "https://huggingface.co/HikariDawn/This-and-That-1.0/resolve/main/unet/train_image2video.yaml"
    if model_type == "GestureNet":    # If it is GestureNet, this UNet is based on the file path recorded inside
        yaml_url = "https://huggingface.co/HikariDawn/This-and-That-1.0/resolve/main/gesturenet/train_image2video_gesturenet.yaml"
        

    # Prepare the temporary Store Path
    yaml_download_path = "pretrained/current_yaml.yaml"
    if os.path.exists(yaml_download_path):
        os.remove(yaml_download_path)
    
    # Download the yaml path
    urllib.request.urlretrieve(yaml_url, yaml_download_path)
    assert(os.path.exists(yaml_download_path))

    # Load the config
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
        generated_frames = unet_log_validation(
                                                vae, unet, image_encoder, text_encoder, tokenizer, 
                                                base_config, accelerator, weight_dtype, step="", 
                                                parent_store_folder=parent_store_folder, force_close_flip = True,
                                                use_ambiguous_prompt = use_ambiguous_prompt,
                                            )
    
    elif model_type == "GestureNet":
        generated_frames = controlnet_log_validation(
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
        default="HikariDawn/This-and-That-1.0",
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


