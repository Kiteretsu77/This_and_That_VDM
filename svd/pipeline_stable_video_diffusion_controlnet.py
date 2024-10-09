# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
import cv2, os, sys
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from utils.img_utils import tensor2np
from svd.temporal_controlnet import ControlNetModel
from svd.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for zero-shot text-to-video pipeline.

    Args:
        frames (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    frames: Union[List[PIL.Image.Image], np.ndarray]


class StableVideoDiffusionControlNetPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()

        # TODO: multi-controlnet consideration
        self.register_modules(
            vae = vae,
            image_encoder = image_encoder,
            unet = unet,
            scheduler = scheduler,
            feature_extractor = feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)       # The vae_scale_factor is for image dimension, not for image size
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )


    def encode_clip(self, image, prompt, use_text, text_encoder, device, num_videos_per_prompt, do_classifier_free_guidance, use_instructpix2pix):
        
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)        # Map [0, 255] to [0, 1] range
            image = self.image_processor.numpy_to_pt(image)     

            # We normalize the image before resizing to match with the original implementation.
            # Then, we unnormalize it after resizing.
            image = image * 2.0 - 1.0       # [-1, 1] range
            image = _resize_with_antialiasing(image, (224, 224))    # Resize to square image
            image = (image + 1.0) / 2.0     # [0, 1] range

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values      # The value range is a little deviated now, and I got [-1.76, 2.15] for one sample

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        encoder_hidden_states = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)


        # Prepare for the text embeddings if needed
        if use_text:
            text_embeddings = text_encoder(prompt)[0]
            
            # Concat two embeddings together on dim 1
            encoder_hidden_states = torch.cat((text_embeddings, encoder_hidden_states), dim=1) 

            # Layer norm on the last dim  TODO: 这里order小改了一下顺序，变成先encoder hidden states了
            layer_norm = nn.LayerNorm((78, 1024)).to(device=device, dtype=dtype)
            encoder_hidden_states = layer_norm(encoder_hidden_states) 


        if do_classifier_free_guidance:
            negative_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if use_instructpix2pix:
                encoder_hidden_states = torch.cat([encoder_hidden_states, negative_encoder_hidden_states, negative_encoder_hidden_states])
            else:
                encoder_hidden_states = torch.cat([negative_encoder_hidden_states, encoder_hidden_states])


        return encoder_hidden_states


    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        use_instructpix2pix,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if use_instructpix2pix:
                image_latents = torch.cat([image_latents, image_latents, negative_image_latents])      
            else:
                image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents


    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        coordinate_values,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        guess_mode,
        use_instructpix2pix,
    ):
        # Define the default values from SVD
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]


        # Sanity Check
        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            if use_instructpix2pix:
                add_time_ids = torch.cat([add_time_ids, add_time_ids, add_time_ids]) 
            else:
                add_time_ids = torch.cat([add_time_ids, add_time_ids])


        # Return the info
        return add_time_ids


    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        accepts_num_frames = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, 
                     image, 
                     height, 
                     width,
                     ):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # TODO: I didn't test input for controlnet_conditioning_scale, control_guidance_start, and control_guidance_end

    def prepare_latents(
        self,
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # We don't directly have do_classifier_free_guidance function, we judge simply by max_guidance

    @property
    def num_timesteps(self):
        return self._num_timesteps


    def prepare_condition_image(
        self,
        condition_img,
        width,
        height,
        batch_size,
        num_videos_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        # The input of condition_img is already in the range [0, 1]
        condition_img = torch.from_numpy(condition_img)  # hwc -> chw
        condition_img = condition_img.to(torch.float16).to(self._execution_device)        # Set this in default

        # CFG will be done in main function, not here now
        return condition_img    # [0, 1] range && Torch data type



    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        condition_img: np, 
        controlnet: ControlNetModel, 
        prompt = None, 
        use_text: bool = False,
        text_encoder = None,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        controlnet_image_index: Optional[int] = [0],
        coordinate_values = None,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        use_instructpix2pix: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        inner_conditioning_scale: float = 1.0,
        guess_mode: bool = True,
        image_guidance_scale: float = 7.5,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """

        # align format for control guidance
        mult = 1
        control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )
        

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor       #  It seems that self.unet.config.sample_size * self.vae_scale_factor  is a default image size input setting

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = max_guidance_scale > 1.0
        if do_classifier_free_guidance:
            print("We will use CFG!!!")


        # 3. Encode input image
        encoder_hidden_states = self.encode_clip(image, prompt, use_text, text_encoder, device, num_videos_per_prompt, do_classifier_free_guidance, use_instructpix2pix)



        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width)      # [0, 255] to [-1, 1]
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        image = image + noise_aug_strength * noise


        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, do_classifier_free_guidance, use_instructpix2pix)
        image_latents = image_latents.to(encoder_hidden_states.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)



        # 4.5 Prepare control image (Will need to consider multiControlNet)
        condition_img = self.prepare_condition_image(
                            condition_img = condition_img,
                            width = width,
                            height = height,
                            batch_size = batch_size * num_videos_per_prompt,
                            num_videos_per_prompt = num_videos_per_prompt,
                            device = device,
                            dtype = controlnet.dtype,
                            do_classifier_free_guidance = do_classifier_free_guidance,
                            guess_mode = guess_mode,
                        )   # [0, 255] to [0, 1] range
        

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
                                                fps,
                                                motion_bucket_id,
                                                noise_aug_strength,
                                                coordinate_values,
                                                encoder_hidden_states.dtype,
                                                batch_size,
                                                num_videos_per_prompt,
                                                do_classifier_free_guidance,
                                                guess_mode = guess_mode,
                                                use_instructpix2pix = use_instructpix2pix,
                                            )
        added_time_ids = added_time_ids.to(device)


        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps


        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            encoder_hidden_states.dtype,
            device,
            generator,
            latents,
        )   # Nosiy latents across all frames needed


        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale


        # 7.5 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)


        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                if use_instructpix2pix:
                    latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents  
                else:
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents  

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)        # I think that this is where sequential generation takes influence

                # Concatenate image_latents over channels dimension for video diffusion purposes
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)      # image_latents is fixed and latent_model_input will be based on latents which is updated frequently


                # ControlNet Scale
                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]


                assert condition_img.shape[1] >= len(controlnet_image_index)   


                # VAE encode
                controlnet_cond = self.vae.encode(condition_img).latent_dist.mode()
                

                if do_classifier_free_guidance:
                    if use_instructpix2pix:
                        controlnet_cond = torch.cat([controlnet_cond, controlnet_cond, controlnet_cond])
                        # controlnet_conditioning_mask = torch.cat([controlnet_conditioning_mask, controlnet_conditioning_mask, controlnet_conditioning_mask])
                    else:
                        controlnet_cond = torch.cat([controlnet_cond, controlnet_cond])
                        # controlnet_conditioning_mask = torch.cat([controlnet_conditioning_mask, controlnet_conditioning_mask])

                
                down_block_res_samples, mid_block_res_sample = controlnet(
                    sample = latent_model_input,          
                    timestep = t,
                    encoder_hidden_states = encoder_hidden_states,   
                    added_time_ids = added_time_ids,
                    controlnet_cond = controlnet_cond,
                    return_dict = False,
                    inner_conditioning_scale = inner_conditioning_scale,        # Inner conditioning scale
                    conditioning_scale = cond_scale,                            # Outer conditioning scale
                    guess_mode = guess_mode,
                )

                if guess_mode and do_classifier_free_guidance:  # Won't consider this one, since we don't use guess mode
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])


                # predict the noise residual
                noise_pred = self.unet(
                    sample = latent_model_input,     # [batch, frames, 4*2, height, width]
                    timestep = t,
                    encoder_hidden_states = encoder_hidden_states,     
                    added_time_ids = added_time_ids,
                    down_block_additional_residuals = down_block_res_samples,
                    mid_block_additional_residual = mid_block_res_sample,
                    return_dict = False,
                )[0]        # image_embeddings is used for cross attention metioned in the paper


                # perform guidance
                if do_classifier_free_guidance:
                    if use_instructpix2pix:
                        noise_pred_1st_frame, noise_pred_cond, noise_pred_uncond = noise_pred.chunk(3)        # There are two noises here: one is unconditional and one is conditional
                        noise_pred = noise_pred_uncond + \
                                        self.guidance_scale * (noise_pred_cond - noise_pred_uncond) + \
                                        image_guidance_scale * (noise_pred_cond - noise_pred_1st_frame)     # InstructPix2Pix is (noise_pred_1st_frame - noise_pred_cond)
                    else:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)        # There are two noises here: one is unconditional and one is conditional
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)


                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents
        
        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)


# resizing utils
# TODO: clean up later  (put to shared utils file)
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def is_compiled_module(module) -> bool:
    """Check whether the module was compiled with torch.compile()"""
    if is_torch_version("<", "2.0.0") or not hasattr(torch, "_dynamo"):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)