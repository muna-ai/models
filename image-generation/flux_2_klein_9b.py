#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = ["diffusers[torch]>=0.37.0", "muna", "sentencepiece", "transformers>=5.7"]
# ///

from accelerate import init_empty_weights
from contextlib import contextmanager
from diffusers import (
    AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline, Flux2Transformer2DModel
)
from muna import compile, BatchConfig, Parameter, Sandbox
from muna.beta import Annotations, DiffusersToSGLangInferenceMetadata
from os import environ
from PIL import Image
from torch import Generator
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from typing import Annotated

# Skip weight initialization when instantiating the transformers text encoder
# from its config (mirrors the LLM predictors). diffusers `from_config` already
# avoids loading weights, so this only guards the transformers sub-model.
@contextmanager
def suppress_init_weights():
    saved = PreTrainedModel.init_weights
    PreTrainedModel.init_weights = lambda self, *a, **kw: None
    try:
        yield
    finally:
        PreTrainedModel.init_weights = saved

# Load FLUX.2 [klein] 9B
# Suppress weight download for speed
CHECKPOINT = "black-forest-labs/FLUX.2-klein-9B"
transformer_config = Flux2Transformer2DModel.load_config(CHECKPOINT, subfolder="transformer")
vae_config = AutoencoderKLFlux2.load_config(CHECKPOINT, subfolder="vae")
text_encoder_config = AutoConfig.from_pretrained(CHECKPOINT, subfolder="text_encoder")
with suppress_init_weights(), init_empty_weights():
    transformer = Flux2Transformer2DModel.from_config(transformer_config)
    vae = AutoencoderKLFlux2.from_config(vae_config)
    text_encoder = AutoModel.from_config(text_encoder_config)

# Load tokenizer and scheduler
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, subfolder="tokenizer")
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(CHECKPOINT, subfolder="scheduler")

# Create the diffusion pipeline. This is the object muna inspects (via the
# metadata below) to recover the full topology of all three sub-models.
pipe = Flux2KleinPipeline(
    transformer=transformer,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler
)
MAX_BATCH_SIZE = 4
MAX_NUM_IMAGES = 4

@compile(
    targets=["x86_64-unknown-linux-gnu"], # Linux x64 + CUDA only
    sandbox=Sandbox()
        .pip_install("diffusers[torch]>=0.37.0", "sentencepiece", "transformers>=5.7")
        .env({ "HF_TOKEN": environ.get("HF_TOKEN") }), # FLUX.2 [klein] is a gated repo
    metadata=[
        DiffusersToSGLangInferenceMetadata(
            pipeline=pipe,
            compute_architecture="sm_100", # Blackwell
            max_batch_size=MAX_BATCH_SIZE * MAX_NUM_IMAGES
        )
    ]
)
def flux_2_klein_9b(
    prompt: Annotated[list[str], Parameter.Generic(
        description="Text descriptions of the desired images.",
        batch=BatchConfig(mode="dynamic", capacity=MAX_BATCH_SIZE)
    )],
    *,
    width: Annotated[int, Annotations.ImageWidth(
        description="Generated image width in pixels.",
        min=256,
        max=2048
    )]=1024,
    height: Annotated[int, Annotations.ImageHeight(
        description="Generated image height in pixels.",
        min=256,
        max=2048
    )]=1024,
    num_images: Annotated[int, Annotations.ImageCount(
        description="Number of images to generate per prompt.",
        min=1,
        max=MAX_NUM_IMAGES
    )]=1,
    num_inference_steps: Annotated[int, Parameter.Numeric(
        description="Number of denoising steps. Klein is step-distilled to 4.",
        min=1,
        max=8
    )]=4,
    guidance_scale: Annotated[float, Parameter.Numeric(
        description="Embedded (guidance-distilled) scale. Klein takes a baked-in guidance value, not true CFG, so a single forward runs per step.",
        min=1.0,
        max=10.0
    )]=4.0,
    seed: Annotated[int, Parameter.Numeric(
        description="Random seed for the initial latent noise.",
        min=0,
        max=2_147_483_647
    )]=0,
) -> Annotated[
    list[Image.Image],
    Parameter.Generic(description="Generated images.")
]:
    """
    Generate images from text prompts with FLUX.2 [klein] 9B.
    """
    generator = Generator(device="cuda").manual_seed(seed)
    output = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    return output.images