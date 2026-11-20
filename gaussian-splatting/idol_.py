#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "muna",
#     "idol @ git+https://github.com/speridlabs/idol-sdk.git@4702a7f00850e34b6e83a2170cc123c3e9570744",
#     "diffusers==0.38.0",
#     "torch==2.11.0",
#     "torchvision==0.26.0",
# ]
# ///

from idol import IdolPipeline
from io import BytesIO
from muna import compile, Parameter, Sandbox
from muna.beta import Annotations
from os import environ
from PIL import Image
from typing import Annotated

pipe = IdolPipeline.from_pretrained("speridlabs/idol", device="cpu")

@compile(
    targets=["x86_64-unknown-linux-gnu"],
    sandbox=Sandbox()
        .pip_install("torch==2.11.0", "torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install(
            "diffusers==0.38.0",
            "idol @ git+https://github.com/speridlabs/idol-sdk.git@4702a7f00850e34b6e83a2170cc123c3e9570744"
        )
        .env({ "HF_TOKEN": environ.get("HF_TOKEN") })
)
def idol_real(
    images: Annotated[
        list[Image.Image],
        Annotations.ChatImages(description="Reference images. Idol requires exactly one image; alpha is preserved.")
    ],
    *,
    prompt: Annotated[
        str,
        Annotations.ImagePrompt(description="Ignored. Idol does not use text conditioning.")
    ] = "",
    seed: Annotated[int, Parameter.Numeric(
        description="Random seed for posterior sampling, flow noise and octree sampling.",
        min=0,
        max=2_147_483_647
    )] = 42,
    steps: Annotated[int, Parameter.Numeric(
        description="Number of Euler sampling steps.",
        min=1,
        max=1000
    )] = 50,
    cfg_strength: Annotated[float, Parameter.Numeric(
        description="Classifier-free guidance strength; zero disables guidance.",
        min=0.0,
        max=10.0
    )] = 3.0,
    rescale_t: Annotated[float, Parameter.Numeric(
        description="Positive timestep schedule rescaling factor.",
        min=1.0,
        max=6.0
    )] = 3.0,
    num_gaussians: Annotated[int, Parameter.Numeric(
        description="Requested Gaussian count, rounded down to a multiple of 32.",
        min=32768,
        max=1048576
    )] = 262144,
    crop_margin: Annotated[float, Parameter.Numeric(
        description="Foreground crop margin; 1.7 for photographs, 1.2 for rendered objects.",
        min=1.0,
        max=4.0
    )] = 1.7
) -> Annotated[
    bytes,
    Parameter.Model3D(description="Gaussian splat PLY.")
]:
    """
    Generate one 3D Gaussian asset from an image with Idol on Blackwell.
    """
    if len(images) != 1:
        raise ValueError("Idol requires exactly one input image")
    image = images[0].convert("RGBA")
    output = pipe(
        image=image,
        seed=seed,
        steps=steps,
        cfg_strength=cfg_strength,
        rescale_t=rescale_t,
        num_gaussians=num_gaussians,
        crop_margin=crop_margin
    )
    buffer = BytesIO()
    output.gaussian.save_ply(buffer)
    return buffer.getvalue()
