#!/usr/bin/env python
'''
    This file is to train Stable Video Diffusion with Conditioning design by my peronal implementation which is based on diffusers' training example code.
'''

import argparse
import logging
import math
import os, sys
import time
import random
import shutil
import warnings
from PIL import Image 
from einops import rearrange, repeat
from pathlib import Path
from omegaconf import OmegaConf
import imageio
import cv2


import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig


import diffusers
from diffusers import (
    AutoencoderKLTemporalDecoder,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, load_image, export_to_video
from diffusers.utils.import_utils import is_xformers_available
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
if is_wandb_available():
    import wandb


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from svd.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionControlNetPipeline
from svd.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from svd.temporal_controlnet import ControlNetModel
from utils.img_utils import resize_with_antialiasing
from utils.optical_flow_utils import flow_to_image, filter_uv, bivariate_Gaussian
from data_loader.video_dataset import tokenize_captions
from data_loader.video_this_that_dataset import Video_ThisThat_Dataset, get_thisthat_sam
from train_code.train_svd import import_pretrained_text_encoder

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.25.0.dev0")

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


###################################################################################################################################################
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/train_image2video_gesturenet.yaml",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    args = parser.parse_args()
    return args


    
def log_validation(vae, unet, controlnet, image_encoder, text_encoder, tokenizer, config, accelerator, weight_dtype, step, 
                        parent_store_folder=None, force_close_flip=False, use_ambiguous_prompt=False):
    # This function will also be used in other files
    print("Running validation... ")


    # Init
    validation_source_folder = config["validation_img_folder"] 
    if not os.path.exists(validation_source_folder):    
        # If you don't have the validation dataset, we skip.
        return
    

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
                                controlnet_image_index = controlnet_image_index,
                                coordinate_values = coordinate_values,
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


def tensor_to_vae_latent(inputs, vae):
    video_length = inputs.shape[1]

    inputs = rearrange(inputs, "b f c h w -> (b f) c h w")
    latents = vae.encode(inputs).latent_dist.mode()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)    # Use f or b to rearrage should have the same effect
    latents = latents * vae.config.scaling_factor

    return latents



def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7 # In the range [0, 1]
    # TODO: "* (1 - 2e-7) + 1e-7" is not included in previous code, I add it back, don't why whether there is any influence now
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def get_add_time_ids(
        unet_config,
        expected_add_embed_dim,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance = False,
    ):

    # Construct Basic add_time_ids items
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]


    # Sanity Check
    passed_add_embed_dim = unet_config.addition_time_embed_dim * len(add_time_ids)
    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

    return add_time_ids
####################################################################################################################################################################



def main(config):
    # Read Config Setting
    resume_from_checkpoint = config["resume_from_checkpoint"]
    output_dir = config["output_dir"]
    logging_name = config["logging_name"]
    mixed_precision = config["mixed_precision"]
    report_to = config["report_to"]
    pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
    pretrained_tokenizer_name_or_path = config["pretrained_tokenizer_name_or_path"]
    gradient_checkpointing = config["gradient_checkpointing"]
    learning_rate = config["learning_rate"]
    adam_beta1 = config["adam_beta1"]
    adam_beta2 = config["adam_beta2"]
    adam_weight_decay = config["adam_weight_decay"]
    adam_epsilon = config["adam_epsilon"]
    train_batch_size = config["train_batch_size"]
    dataloader_num_workers = config["dataloader_num_workers"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    num_train_iters = config["num_train_iters"]
    lr_warmup_steps = config["lr_warmup_steps"]
    checkpointing_steps = config["checkpointing_steps"]
    process_fps = config["process_fps"]
    train_noise_aug_strength = config["train_noise_aug_strength"]
    use_8bit_adam = config["use_8bit_adam"]
    scale_lr = config["scale_lr"]
    conditioning_dropout_prob = config["conditioning_dropout_prob"]
    checkpoints_total_limit = config["checkpoints_total_limit"]
    validation_step = config["validation_step"]
    partial_finetune = config['partial_finetune']
    load_unet_path = config['load_unet_path']

    if mixed_precision == 'None':   # For mixed precision use
        mixed_precision = 'no'


    # Default Setting
    revision = None
    variant = "fp16"        # TODO: 这里进行了调整，不知道会有多少区别，现在跟unet training保持一致
    lr_scheduler = "constant"
    max_grad_norm = 1.0
    tracker_project_name = "img2video"
    num_videos_per_prompt = 1
    seed = 42
    # No CFG in training now



    # Define the accelerator
    logging_dir = Path(output_dir, logging_name)
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps = gradient_accumulation_steps,
        mixed_precision = mixed_precision,
        log_with = report_to,
        project_config = accelerator_project_config,
    )
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    # Handle the repository creation
    if accelerator.is_main_process and resume_from_checkpoint != "latest":      # For the latest checkpoint version, we don't need to delete our folders
        # Validation file
        validation_store_folder = config["validation_store_folder"] + "_" + config["scheduler"]
        print("We will remove ", validation_store_folder)
        if os.path.exists(validation_store_folder):
            archive_name = validation_store_folder + "_archive"
            if os.path.exists(archive_name):
                shutil.rmtree(archive_name)
            print("We will move to archive ", archive_name)
            os.rename(validation_store_folder, archive_name)
        os.makedirs(validation_store_folder)

        # Output Dir
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            # os.makedirs(output_dir, exist_ok=True)

        # Log
        if os.path.exists("runs"):
            shutil.rmtree("runs")
        
        # Copy the config to here
        os.system(" cp config/train_image2video_gesturenet.yaml " + validation_store_folder + "/")


    # Load All Module Needed
    feature_extractor = CLIPImageProcessor.from_pretrained(
        pretrained_model_name_or_path, subfolder="feature_extractor", revision=revision
    )   # This instance has now weight, they are just seeting file
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder="image_encoder", revision=revision, variant=variant
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant
    )
    if load_unet_path != None:
        print("We will use pretrained UNet path by our, at ", load_unet_path)
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            load_unet_path, 
            subfolder = "unet", 
            low_cpu_mem_usage = True,       
        )   # For the variant, we don't have fp16 version, so we will read from fp32
    else:
        print("We will still use provided UNet path")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder = "unet", 
            low_cpu_mem_usage = True,
            variant = variant,
        )

    # Prepare for the tokenizer if use text
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_tokenizer_name_or_path,
        subfolder = "tokenizer",
        revision = revision,
        use_fast = False,
    )

    if config["use_text"]:
        # Clip Text Encoder
        text_encoder_cls = import_pretrained_text_encoder(pretrained_tokenizer_name_or_path, revision)
        text_encoder = text_encoder_cls.from_pretrained(
            pretrained_tokenizer_name_or_path, subfolder = "text_encoder", revision = revision, variant = variant
        )
    else:
        text_encoder = None

    # Init for the Controlnet (check if has pretrained path to load)
    if config["load_controlnet_path"] != None:
        print("We will load pre-trained controlnet from ", config["load_controlnet_path"])
        controlnet = ControlNetModel.from_pretrained(config["load_controlnet_path"], subfolder="controlnet")
    else:
        controlnet = ControlNetModel.from_unet(unet, load_weights_from_unet=True, conditioning_channels=config["conditioning_channels"])


    # Store the config due to the disappearance after accelerator prepare
    unet_config = unet.config
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features


    # Freeze vae + feature_extractor + image_encoder, but set unet to trainable
    vae.requires_grad_(False)       
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)          # UNet won't be trained in conditioning branch
    controlnet.requires_grad_(False)    # Will turn back to requires grad later on
    if config["use_text"]:
        text_encoder.requires_grad_(False)


    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # Move vae + unet + image_encoder to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)     # we don't train UNet anymore, so we cast it here
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    if config["use_text"]:
        text_encoder.to(accelerator.device, dtype=weight_dtype)



    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    if gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()


    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )
    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )
    
    


    ################################ Make Training dataset  ######################################
    if config["data_loader_type"] == "thisthat":    # Only keep thisthat mode now
        train_dataset = Video_ThisThat_Dataset(config, accelerator.device, tokenizer=tokenizer)
    else:
        raise NotImplementedError("We don't support such data loader type")

    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler = sampler,
        batch_size = train_batch_size,
        num_workers = dataloader_num_workers * accelerator.num_processes,
    )       
    ##############################################################################################
    


    ####################################### Optimizer Setting ##############################################################
    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # 8bit adam to save more memory
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    
    # Make ControlNet require Grad
    controlnet.requires_grad_(True)


    ###################### For partial fine-tune setting #######################
    parameters_list = []
    for name, para in controlnet.named_parameters():
        if partial_finetune:    # The partial finetune we use is to only train attn layers, which will be ~190M params (TODO:needs to check later for exact value)
            if not name.find("attn") != -1:     # Only block the spatial Transformer
                para.requires_grad = False
            else:
                parameters_list.append(para)
                para.requires_grad = True  
        else:
            parameters_list.append(para)
            para.requires_grad = True

    # Double check the weight that will be trained
    total_params_for_training = 0
    for name, param in controlnet.named_parameters():
        if param.requires_grad:
            total_params_for_training += param.numel()
            print(name + " requires grad update")    
    print("Total parameter that will be trained in controlnet has ", total_params_for_training)
    #############################################################################

    # Optimizer creation
    optimizer = optimizer_cls(
        parameters_list,
        lr = learning_rate,
        betas = (adam_beta1, adam_beta2),
        weight_decay = adam_weight_decay,
        eps = adam_epsilon,
    )


    # Scheduler and Training steps
    dataset_length = len(train_dataset)
    print("Dataset length read from the train side is ", dataset_length)
    num_update_steps_per_epoch = math.ceil(dataset_length / gradient_accumulation_steps)
    max_train_steps = num_train_iters * train_batch_size

    # Learning Rate Scheduler   (we all use constant)
    lr_scheduler = get_scheduler(
        "constant",
        optimizer = optimizer,
        num_warmup_steps = lr_warmup_steps * accelerator.num_processes,
        num_training_steps = max_train_steps *  accelerator.num_processes,
        num_cycles = 1,
        power = 1.0,
    )
    #######################################################################################################################
    


    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )   



    # We need to RECALCULATE our total training steps as the size of the training dataloader may have changed.
    print("accelerator.num_processes is ", accelerator.num_processes)
    print("num_train_iters is ", num_train_iters)
    num_train_epochs = math.ceil(num_train_iters * accelerator.num_processes * gradient_accumulation_steps / dataset_length) 
    print("num_train_epochs is ", num_train_epochs)

    # We need to initialize the trackers we use, and also store our configuration.
    if accelerator.is_main_process: # Only on the main process!
        tracker_config = dict(vars(args))
        accelerator.init_trackers(tracker_project_name, tracker_config)



    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Dataset Length = {dataset_length}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")



    # Load the Closest / Best weight       TODO: need to check how to use checkpoints from pre-trained weights!!!
    global_step = 0     # Catch the current iteration
    first_epoch = 0
    if resume_from_checkpoint:          # Resume Checkpoints!!!!!
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            print("We will resume the latest weight ", path)

        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    if accelerator.is_main_process:
        print("Initial Learning rate is ", optimizer.param_groups[0]['lr'])
        print("global_step will start from ", global_step)

    progress_bar = tqdm(
                            range(initial_global_step, max_train_steps),
                            initial=initial_global_step,
                            desc="Steps",
                            # Only show the progress bar once on each machine.
                            disable=not accelerator.is_local_main_process,
                        )

    

    # Prepare tensorboard log
    writer = SummaryWriter() 


    ################################### Auxiliary Function ################################################################################################
    def encode_clip(pixel_values, prompt):
        ''' Encoder hidden states input source
            pixel_values:   first frame pixel information
            prompt:         language prompt with takenized
        '''

        ########################################## Prepare the Text Embedding #####################################################
        # pixel_values is in the range [-1, 1]
        pixel_values = resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0   # [-1, 1] -> [0, 1]

        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        # The following is the same as _encode_image in SVD pipeline
        pixel_values = pixel_values.to(device=accelerator.device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        encoder_hidden_states = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        #############################################################################################################################



        ########################################## Prepare the Text embedding if needed #############################################
        if config["use_text"]:
            text_embeddings = text_encoder(prompt)[0]
            
            # Concat two embeddings together on dim 1
            encoder_hidden_states = torch.cat((text_embeddings, encoder_hidden_states), dim=1)      # 目前先用text_embeddings 再用encoder_hidden_states感觉好一点

            # Layer norm on the last dim
            layer_norm = nn.LayerNorm((78, 1024)).to(device=accelerator.device, dtype=weight_dtype)
            encoder_hidden_states_norm = layer_norm(encoder_hidden_states)

            # Return
            return encoder_hidden_states_norm

        else:   # Just return back default on
            return encoder_hidden_states
        #############################################################################################################################

    #########################################################################################################################################################


    ############################################################################################################################
    # For the training, we mimic the code from test2image in diffusers  TODO: check the data loader conflict
    for epoch in range(first_epoch, num_train_epochs):
        controlnet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # batch is a dictionary with video_frames and controlnet_condition
                video_frames = batch["video_frames"].to(weight_dtype).to(accelerator.device, non_blocking=True)        # [-1, 1] range
                condition_img = batch["controlnet_condition"].to(dtype=weight_dtype)        # [0, 1] range
                reflected_motion_bucket_id = batch["reflected_motion_bucket_id"]
                controlnet_image_index = batch["controlnet_image_index"]
                prompt = batch["prompt"]


                # Images to VAE latent space
                latents = tensor_to_vae_latent(video_frames, vae)       # For all frames
                

                ##################################### Add Noise ########################################
                bsz, num_frames = latents.shape[:2]
                

                # Encode the first frame
                conditional_pixel_values = video_frames[:, 0, :, :, :]      # First frame
                # Following AnimateSomething, we use constant to repace cond_sigmas
                conditional_pixel_values = conditional_pixel_values + torch.randn_like(conditional_pixel_values) * train_noise_aug_strength # cond_sigmas
                conditional_latents = vae.encode(conditional_pixel_values).latent_dist.mode()   
                conditional_latents = repeat(conditional_latents, 'b c h w->b f c h w', f=num_frames)       # conditional_latents没有noise的成分的


                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = rand_log_normal(shape=[bsz,], loc=config["noise_mean"], scale=config["noise_std"]).to(weight_dtype).to(latents.device)      # TODO: 我觉得noise这块，sigma算法是最不确定是否正确的地方
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = latents + torch.randn_like(latents) * sigmas
                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)        # multiplied by c_in in paper


                # For the encoder hidden states based on the first frame and prompt
                encoder_hidden_states = encode_clip(video_frames[:, 0, :, :, :].float(), prompt)     # First Frame + Text Prompt


                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if conditioning_dropout_prob != 0:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)

                    # Sample masks for the encoder_hidden_states (to replace prompts in InstructPix2Pix). 
                    prompt_mask = random_p < 2 * conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final encoder_hidden_states conditioning.
                    null_conditioning = torch.zeros_like(encoder_hidden_states) # encoder_hidden_states has already been used with .unsqueeze(1)
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states) 

                    # Sample masks for the original image latents.
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - ((random_p >= conditioning_dropout_prob).to(image_mask_dtype) * (random_p < 3 * conditioning_dropout_prob).to(image_mask_dtype))
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)

                    # Final image conditioning.
                    conditional_latents = image_mask * conditional_latents   

                # The Concatenation is move downward with the masking feature


                # GT noise
                target = latents
                ##########################################################################################


                ################################ Other Embedding and Conditioning ###################################
                reflected_motion_bucket_id = torch.sum(reflected_motion_bucket_id)/len(reflected_motion_bucket_id)
                reflected_motion_bucket_id = int(reflected_motion_bucket_id.cpu().detach().numpy())
                # print("Training reflected_motion_bucket_id is ", reflected_motion_bucket_id)

                added_time_ids = get_add_time_ids(
                                                    unet_config,
                                                    expected_add_embed_dim,
                                                    process_fps,
                                                    reflected_motion_bucket_id,
                                                    train_noise_aug_strength,       # Note: noise strength
                                                    weight_dtype,
                                                    train_batch_size,
                                                    num_videos_per_prompt,
                                                )       # The same as SVD pipeline's _get_add_time_ids
                added_time_ids = added_time_ids.to(accelerator.device)

                timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)
                ##########################################################################################



                ################################### Get ControlNet Output ###################################

                # Transform controlnet_image_index to the data format we want
                controlnet_image_index = list(controlnet_image_index.cpu().detach().numpy()[0])
                assert condition_img.shape[1] >= len(controlnet_image_index)  

                # Designing the 0/1 mask for Sparse Conditioning
                controlnet_conditioning_mask_shape = list(condition_img.shape)
                controlnet_conditioning_mask_shape[2] = 1       # frame dim
                controlnet_conditioning_mask = torch.zeros(controlnet_conditioning_mask_shape).to(dtype=weight_dtype).to(accelerator.device)
                controlnet_conditioning_mask[:, controlnet_image_index] = 1


                # Add vae latent mask to controlnet noise
                if config["mask_controlnet_vae"]:
                    b, f, c, h, w = conditional_latents.shape

                    # Create a mask: Value less than the threshold is set to be True
                    mask = torch.rand((b, f, 1, h, w), device=accelerator.device) < (1-config["mask_proportion"])      # channel sync
                    # mask[:,0,:,:,:] = 1     # For the first frame, we still keep it

                    # Multiply to the conditional latents, we will just make the mean and variance zero to present those with zero masking
                    masked_conditional_latents = conditional_latents * mask
                    controlnet_inp_noisy_latents = torch.cat([inp_noisy_latents, masked_conditional_latents], dim=2)
                else:
                    controlnet_inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)


                # VAE encode
                controlnet_cond = condition_img.flatten(0, 1)
                controlnet_cond = vae.encode(controlnet_cond).latent_dist.mode()


                down_block_res_samples, mid_block_res_sample = controlnet(
                    sample = controlnet_inp_noisy_latents,          
                    timestep = timesteps,
                    encoder_hidden_states = encoder_hidden_states,     
                    added_time_ids = added_time_ids,
                    controlnet_cond = controlnet_cond,
                    return_dict = False,
                    conditioning_scale = config["outer_conditioning_scale"],
                    inner_conditioning_scale = config["inner_conditioning_scale"],
                    guess_mode = False,         # No Guess Mode
                )   

                #############################################################################################



                ###################################### Predict Noise ########################################
                # Add vae latent mask to controlnet noise
                if config["mask_unet_vae"]:
                    b, f, c, h, w = conditional_latents.shape

                    # Create a mask
                    mask = torch.rand((b, f, 1, h, w), device=accelerator.device) < (1-config["mask_proportion"])      # channel sync
                    # mask[:,0,:,:,:] = 1     # For the first frame, we still keep it

                    # Multiply to the conditional latents, we will just make the mean and variance zero to present those with zero masking
                    if not config["mask_controlnet_vae"]:  
                        masked_conditional_latents = conditional_latents * mask
                    unet_inp_noisy_latents = torch.cat([inp_noisy_latents, masked_conditional_latents], dim=2)
                else:
                    unet_inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)

                # Add with controlnet middle output layers
                model_pred = unet(
                                    unet_inp_noisy_latents,
                                    timesteps, 
                                    encoder_hidden_states, 
                                    added_time_ids = added_time_ids,
                                    down_block_additional_residuals = [
                                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                                    ],
                                    mid_block_additional_residual = mid_block_res_sample.to(dtype=weight_dtype),
                                ).sample    


                # Denoise the latents
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents  # What our loss will optimize with
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)
                ##########################################################################################


                ############################### Calculate Loss and Update Optimizer #######################
                # MSE loss
                loss = torch.mean(
                    (  weighing.float() * (denoised_latents.float() - target.float())**2 ).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Update Tensorboard
                writer.add_scalar('Loss/train-Loss-Step', avg_loss.item()/ gradient_accumulation_steps, global_step)        # 我觉得loss的report就用这个avg_loss就行了
                

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:  # For ControlNet
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()     # I think constant will take no influence here
                optimizer.zero_grad(set_to_none=True)
                ##########################################################################################


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                ########################################## Checkpoints #########################################
                if global_step != 0 and global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        start = time.time()
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if checkpoints_total_limit is not None:
                            checkpoints = os.listdir(output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        print("Save time use " + str(time.time() - start) + " s")
                ########################################################################################################


            # Update Log
            logs = {"step_loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)


           ##################################### Validation per XXX iterations #######################################
            if accelerator.is_main_process:
                if global_step > -1 and global_step % validation_step == 0:         # Fixed 100 steps to validate
                    
                    log_validation(
                                    vae,
                                    unet,
                                    controlnet,
                                    image_encoder,
                                    text_encoder,
                                    tokenizer,
                                    config,
                                    accelerator,
                                    weight_dtype,
                                    global_step,
                                    use_ambiguous_prompt = config["mix_ambiguous"],
                                )
                                
            ###############################################################################################################

            # Update Steps and Break if needed      global step should be updated together
            global_step += 1

            if global_step >= max_train_steps:
                break
    
    ############################################################################################################################


if __name__ == "__main__":
    args = parse_args()

    config = OmegaConf.load(args.config_path)
    main(config)
