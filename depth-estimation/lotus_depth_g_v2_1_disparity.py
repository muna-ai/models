#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# Compilation builds only meta-device component topology and the CLIP tokenizer.
# Set LOTUS_LOAD_WEIGHTS=1 for ordinary Python inference with real weights.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "muna", "accelerate==0.29.3", "diffusers==0.28.0", "huggingface_hub==0.22.2",
#     "numpy==1.26.4", "peft==0.7.0", "pillow", "safetensors==0.8.0",
#     "torch==2.10.0", "torchvision==0.25.0", "transformers==4.40.1"
# ]
# ///

from accelerate import init_empty_weights
from diffusers import AutoencoderKL, DiffusionPipeline, UNet2DConditionModel
from muna import compile, Parameter, Sandbox
from muna.beta import Annotations, DiffusersToSGLangInferenceMetadata
from numpy import ndarray
from os import environ
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from typing import Annotated
import numpy as np
import torch

CHECKPOINT = "jingheya/lotus-depth-g-v2-1-disparity"
CHECKPOINT_REVISION = "5a6f234319ae90f4bdf1b2d326a575e2715eb40d"
SOURCE_REVISION = "b737e12238d2a41fc582ee8c6c6b1104d318d799"

class LotusDepthGPipeline(DiffusionPipeline):
    """
    Empty-prompt Lotus G disparity, sampled VAE posterior and one x0 step at 999.

    The native call accepts the original RGB image and returns original-size
    H×W float32 relative disparity. Python execution below exposes the same
    boundary without relying on the checkpoint's obsolete pipeline index.
    """
    model_cpu_offload_seq = "text_encoder->vae->unet"

    def __init__(
        self,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        checkpoint: str=CHECKPOINT,
        checkpoint_revision: str=CHECKPOINT_REVISION,
        source_revision: str=SOURCE_REVISION
    ):
        super().__init__()
        self.register_modules(text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, vae=vae)
        self.register_to_config(
            checkpoint=checkpoint,
            checkpoint_revision=checkpoint_revision,
            source_revision=source_revision
        )

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image,
        *,
        processing_resolution: int=768,
        seed: int=42
    ) -> ndarray:
        if self.unet.device.type == "meta":
            raise RuntimeError("Python inference requires LOTUS_LOAD_WEIGHTS=1; compilation uses meta topology.")
        width, height = image.size
        if width < 1 or height < 1 or width > 16384 or height > 16384:
            raise ValueError("Lotus input axes must be between 1 and 16384")
        if processing_resolution < 8 or processing_resolution > 1024:
            raise ValueError("Lotus processing_resolution must be in [8,1024]")
        if seed < 0:
            raise ValueError("Lotus seed must be nonnegative")
        ratio = processing_resolution / max(width, height)
        output_height, output_width = int(height * ratio), int(width * ratio)
        if output_height < 8 or output_width < 8:
            raise ValueError("Both processing axes must be at least 8")
        device = self.unet.device
        dtype = self.unet.dtype
        # Lotus image_utils.py:78-98: CPU FP32 division precedes tensor resize.
        pixels = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)[None]
        normalized = pixels.float() / 127.5 - 1.0
        normalized = resize(
            normalized,
            [output_height, output_width],
            InterpolationMode.BILINEAR,
            antialias=True
        ).to(device=device, dtype=dtype)
        ids = self.tokenizer(
            "",
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(device)
        prompt = self.text_encoder(ids)[0]
        generator = torch.Generator(device=device).manual_seed(seed)
        shape = (1, 4, output_height // 8, output_width // 8)
        target = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        posterior_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        moments = self.vae.quant_conv(self.vae.encoder(normalized))
        mean, logvar = moments.chunk(2, dim=1)
        logvar = logvar.clamp(-30.0, 20.0)
        rgb_latent = (mean + torch.exp(0.5 * logvar) * posterior_noise) * self.vae.config.scaling_factor
        task = torch.tensor([[np.sin(1.0), 0.0, np.cos(1.0), 1.0]], device=device, dtype=dtype)
        prediction = self.unet(
            torch.cat([rgb_latent, target], dim=1),
            torch.tensor([999], device=device, dtype=torch.int64),
            encoder_hidden_states=prompt,
            class_labels=task,
            return_dict=False
        )[0]
        # The one-step LotusGPipeline path aliases x0; it never calls a scheduler.
        latent = (prediction.float() * (1.0 / self.vae.config.scaling_factor)).to(dtype)
        decoded = self.vae.decode(latent, return_dict=False)[0]
        denormalized = (decoded / 2.0 + 0.5).clamp(0.0, 1.0)
        restored = resize(denormalized, [height, width], InterpolationMode.NEAREST_EXACT)
        return restored.float().mean(dim=1)[0].cpu().numpy()

def _create_pipeline() -> LotusDepthGPipeline:
    """
    Instantiate the exact component configs on meta, without downloading weights.
    """
    clip_config = CLIPTextConfig.from_pretrained(
        CHECKPOINT,
        subfolder="text_encoder",
        revision=CHECKPOINT_REVISION
    )
    unet_config = UNet2DConditionModel.load_config(CHECKPOINT, subfolder="unet", revision=CHECKPOINT_REVISION)
    vae_config = AutoencoderKL.load_config(CHECKPOINT, subfolder="vae", revision=CHECKPOINT_REVISION)
    with init_empty_weights():
        text_encoder = CLIPTextModel(clip_config).to(dtype=torch.float16)
        unet = UNet2DConditionModel.from_config(unet_config).to(dtype=torch.float16)
        vae = AutoencoderKL.from_config(vae_config).to(dtype=torch.float16)
    tokenizer = CLIPTokenizer.from_pretrained(CHECKPOINT, subfolder="tokenizer", revision=CHECKPOINT_REVISION)
    return LotusDepthGPipeline(text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, vae=vae)

def _load_pipeline() -> LotusDepthGPipeline:
    """
    Load the three components explicitly; the checkpoint's root index is stale.
    """
    options = dict(revision=CHECKPOINT_REVISION, torch_dtype=torch.float16, use_safetensors=True)
    text_encoder = CLIPTextModel.from_pretrained(CHECKPOINT, subfolder="text_encoder", **options).eval()
    unet = UNet2DConditionModel.from_pretrained(CHECKPOINT, subfolder="unet", **options).eval()
    vae = AutoencoderKL.from_pretrained(CHECKPOINT, subfolder="vae", **options).eval()
    tokenizer = CLIPTokenizer.from_pretrained(CHECKPOINT, subfolder="tokenizer", revision=CHECKPOINT_REVISION)
    pipeline = LotusDepthGPipeline(text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, vae=vae)
    return pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

pipe = (
    _load_pipeline()
    if environ.get("LOTUS_LOAD_WEIGHTS") == "1"
    else _create_pipeline()
)

@compile(
    targets=["x86_64-unknown-linux-gnu"],
    sandbox=Sandbox()
        .pip_install("torch==2.10.0", "torchvision==0.25.0", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install(
            "accelerate==0.29.3", "diffusers==0.28.0", "huggingface_hub==0.22.2",
            "transformers==4.40.1", "peft==0.7.0", "numpy==1.26.4", "safetensors==0.8.0"
        ),
    metadata=[
        DiffusersToSGLangInferenceMetadata(
            pipeline=pipe,
            compute_architecture="sm_100"
        )
    ]
)
def lotus_depth_g_v2_1_disparity(
    images: Annotated[
        list[Image.Image],
        Annotations.ChatImages(description="Input images. Lotus requires exactly one image.")
    ],
    *,
    prompt: Annotated[
        str,
        Annotations.ImagePrompt(description="Accepted for OpenAI Images compatibility (ignored).")
    ]="",
    processing_resolution: Annotated[int, Parameter.Numeric(
        description="Maximum processing edge. Both resized axes must be at least 8 pixels.",
        min=8,
        max=1024
    )]=768,
    seed: Annotated[int, Parameter.Numeric(
        description="Seed for target noise and the VAE posterior sample.",
        min=0,
        max=2_147_483_647
    )]=42
) -> Annotated[
    list[ndarray],
    Parameter.DepthMap(description="Relative disparity maps. Larger values are nearer.")
]:
    """
    Estimate relative disparity from one RGB image with Lotus G v2.1.
    """
    if len(images) != 1:
        raise ValueError("Lotus requires exactly one input image")
    depth = pipe(
        image=images[0].convert("RGB"),
        processing_resolution=processing_resolution,
        seed=seed
    )
    return [depth]
