#!/usr/bin/env python
'''
    This file is to train stable video diffusion by my personal implementation which is based on diffusers' training example code.
'''

import argparse
import logging
import math
import os, sys
import time
import random
import shutil
import warnings
import cv2
from PIL import Image 
from einops import rearrange, repeat
from pathlib import Path
from omegaconf import OmegaConf
import imageio

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
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKLTemporalDecoder,
    DDPMScheduler,
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
from svd.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline 
from svd.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from data_loader.video_dataset import Video_Dataset, get_video_frames, tokenize_captions
from utils.img_utils import resize_with_antialiasing


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
        default="config/train_image2video.yaml",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    args = parser.parse_args()
    return args


    
def log_validation(vae, unet, image_encoder, text_encoder, tokenizer, config, accelerator, weight_dtype, step, 
                        parent_store_folder = None, force_close_flip = False, use_ambiguous_prompt=False):
    # This function will also be used in other files
    print("Running validation... ")


    # Init
    validation_source_folder = config["validation_img_folder"] 
    if not os.path.exists(validation_source_folder):    
        # If you don't have the validation dataset, we skip.
        return
    

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
        print("Inference Motion Bucket ID is ", reflected_motion_bucket_id)


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


def tensor_to_vae_latent(inputs, vae):
    video_length = inputs.shape[1]
    inputs = rearrange(inputs, "b f c h w -> (b f) c h w")
    latents = vae.encode(inputs).latent_dist.mode()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)    # Use f or b to rearrage should have the same effect
    latents = latents * vae.config.scaling_factor

    return latents


def import_pretrained_text_encoder(pretrained_model_name_or_path: str, revision: str):
    ''' Import Text encoder information
    
    '''
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    
    else:   # No other cases will be considerred
        raise ValueError(f"{model_class} is not supported.")
    


def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
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
    ):

    # Construct Basic add_time_ids items
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

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


    # Default Setting
    revision = None
    variant = "fp16"
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
        os.system(" cp config/train_image2video.yaml " + validation_store_folder + "/")
            

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
    if config["load_unet_path"] != None:
        print("We will load UNet from ", config["load_unet_path"])
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            config["load_unet_path"], 
            subfolder = "unet", 
            low_cpu_mem_usage = True,       
        )   # For the variant, we don't have fp16 version, so we will read from fp32
    else:
        print("We will only use SVD pretrained UNet")
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


    # Store the config due to the disappearance after accelerator prepare (This is written to handle some unknown phenomenon)
    unet_config = unet.config
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features
    

    # Freeze vae + feature_extractor + image_encoder, but set unet to trainable
    vae.requires_grad_(False)       
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)  # Will switch back to train mode later on
    if config["use_text"]:
        text_encoder.requires_grad_(False)      # All set with no grad needed (like VAE) follow other T2I papers



    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # Move vae + image_encoder to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    # unet.to(accelerator.device, dtype=weight_dtype)
    if config["use_text"]:
        text_encoder.to(accelerator.device, dtype=weight_dtype)



    # Acceleration: `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetSpatioTemporalConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()



    ################################ Make Training dataset  ###############################
    train_dataset = Video_Dataset(config, device = accelerator.device, tokenizer=tokenizer)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler = sampler,
        batch_size = train_batch_size,
        num_workers = dataloader_num_workers * accelerator.num_processes,
    )       
    #######################################################################################


    ####################################### Optimizer  Setting #####################################################################
    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # 8bit adam to save more memory (Usally we need this to save the memory)
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


    # Switch back to unet in training mode
    unet.requires_grad_(True) 


    ############################## For partial fine-tune setting ##############################
    parameters_list = []
    for name, param in unet.named_parameters():
        if partial_finetune:    # The partial finetune we use is to only train attn layers, which will be ~190M params (TODO:needs to check later for exact value)
            # Full Spatial: .transformer_blocks. && spatial_
            # Attn + All emb: attn && emb
            if name.find("attn") != -1 or name.find("emb") != -1:     # Only block the spatial Transformer
                parameters_list.append(param)
                param.requires_grad = True  
            else:
                param.requires_grad = False 
        else:
            parameters_list.append(param)
            param.requires_grad = True

    # Double check what will be trained
    total_params_for_training = 0
    # if os.path.exists("param_lists.txt"):
    #     os.remove("param_lists.txt")
    # file1 = open("param_lists.txt","a")
    for name, param in unet.named_parameters():
        # file1.write(name + "\n")
        if param.requires_grad:
            total_params_for_training += param.numel()
            print(name + " requires grad update")     
    print("Total parameter that will be trained has ", total_params_for_training)
    ##########################################################################################

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

    # Learning Rate Scheduler  (we all use constant)
    lr_scheduler = get_scheduler(
        "constant",
        optimizer = optimizer,
        num_warmup_steps = lr_warmup_steps * accelerator.num_processes,
        num_training_steps = max_train_steps *  accelerator.num_processes,
        num_cycles = 1,
        power = 1.0,
    )
    #####################################################################################################################################
    


    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
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



    # Load the Closest / Best weight   
    global_step = 0     # Catch the current iteration
    first_epoch = 0
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            print("We will resume the latest weight ", path)

        if path is None:     # Don't resume
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
            initial_global_step = 0
        else:       # Resume from the closest checkpoint
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


    ######################################################### Auxiliary Function #################################################################
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
            encoder_hidden_states = torch.cat((text_embeddings, encoder_hidden_states), dim=1)
        
            # Layer norm on the last dim
            layer_norm = nn.LayerNorm((78, 1024)).to(device=accelerator.device, dtype=weight_dtype)
            encoder_hidden_states_norm = layer_norm(encoder_hidden_states)

            # Return
            return encoder_hidden_states_norm

        else:   # Just return back default on
            return encoder_hidden_states
        #############################################################################################################################

    ####################################################################################################################################################


    ############################################################################################################################
    # For the training, we mimic the code from T2I in diffusers
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # batch is a torch tensor with range of [-1, 1] but no other pre-porcessing
                video_frames = batch["video_frames"].to(weight_dtype).to(accelerator.device, non_blocking=True)
                reflected_motion_bucket_id = batch["reflected_motion_bucket_id"]
                prompt = batch["prompt"]
                

                # Images to VAE latent space
                latents = tensor_to_vae_latent(video_frames, vae)


                ##################################### Add Noise ########################################
                bsz, num_frames = latents.shape[:2]
            
                # Encode the first frame
                conditional_pixel_values = video_frames[:, 0, :, :, :]      # First frame
                # Following AnimateSomething, we use constant to repace cond_sigmas
                conditional_pixel_values = conditional_pixel_values + torch.randn_like(conditional_pixel_values) * train_noise_aug_strength        
                conditional_latents = vae.encode(conditional_pixel_values).latent_dist.mode()           # mode() returns mean value no std influence
                conditional_latents = repeat(conditional_latents, 'b c h w->b f c h w', f=num_frames)   # copied across the frame axis to be the same shape as noise


                # Add noise to the latents according to the noise magnitude at each timestep
                # This is the forward diffusion process
                sigmas = rand_log_normal(shape=[bsz,], loc=config["noise_mean"], scale=config["noise_std"]).to(latents.device)
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = latents + torch.randn_like(latents) * sigmas
                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)


                # For the encoder hidden states based on the first frame and prompt
                encoder_hidden_states = encode_clip(video_frames[:, 0, :, :, :].float(), prompt)     # First Frame + Text Prompt


                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800 (InstructPix2Pix).
                if conditioning_dropout_prob != 0:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    # Sample masks for the original images.
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - ((random_p >= conditioning_dropout_prob).to(image_mask_dtype) * (random_p < 3 * conditioning_dropout_prob).to(image_mask_dtype))
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)

                    # Final image conditioning.
                    conditional_latents = image_mask * conditional_latents


                # Concatenate the `conditional_latents` with the `noisy_latents`.
                inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)


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
                                    train_noise_aug_strength,       
                                    weight_dtype,
                                    train_batch_size,
                                    num_videos_per_prompt,
                                )       # The same as SVD pipeline's _get_add_time_ids
                added_time_ids = added_time_ids.to(accelerator.device)

                timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)
                #####################################################################################################



                ###################################### Predict Noise ######################################
                model_pred = unet(
                                    inp_noisy_latents,
                                    timesteps, 
                                    encoder_hidden_states, 
                                    added_time_ids = added_time_ids
                                  ).sample      

                # Denoise the latents
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
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
                writer.add_scalar('Loss/train-Loss-Step', avg_loss, global_step)
                

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()   
                optimizer.zero_grad()
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
            logs = {"step_loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr']}
            progress_bar.set_postfix(**logs)


           ##################################### Validation per XXX iterations #######################################
            if accelerator.is_main_process:
                if global_step % validation_step == 0:         # Fixed 100 steps to validate
                    
                    if config["validation_img_folder"] is not None:
                        log_validation(
                                        vae,
                                        unet,
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

            # Update Steps and Break if needed
            global_step += 1

            if global_step >= max_train_steps:
                break
    
    ############################################################################################################################


if __name__ == "__main__":
    args = parse_args()

    config = OmegaConf.load(args.config_path)
    main(config)
