#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "muna", "accelerate", "addict==2.4.0", "depth-anything-3",
#   "einops", "evo", "huggingface_hub", "numpy<2", "omegaconf",
#   "pillow", "safetensors", "torch", "torchvision"
# ]
# ///

from accelerate import init_empty_weights
from depth_anything_3.cfg import create_object
from huggingface_hub import hf_hub_download
from json import loads
from muna import compile, Parameter, Sandbox
from muna.beta import Annotations, TorchToSGLangInferenceMetadata
from numpy import ndarray
from os import environ
from pathlib import Path
from PIL import Image
from safetensors.torch import load_file
from torch import inference_mode, Tensor
from torch.nn import Module
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from typing import Annotated
import torch

CHECKPOINT = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
CHECKPOINT_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
INPUT_HEIGHT = 504
INPUT_WIDTH = 504

class DA3NestedMetricDepth(Module):
    """
    FP32 normalized [1,1,3,H,W] images to aligned metric depth [1,1,H,W].

    Call the nested network directly: the upstream high-level API enables
    autocast, whereas this native engine preserves checkpoint-native FP32.
    Camera conditioning, Gaussian export and auxiliary taps are omitted here.
    """

    def __init__(self):
        super().__init__()
        self.checkpoint = CHECKPOINT
        self.checkpoint_revision = CHECKPOINT_REVISION
        self.source_revision = SOURCE_REVISION
        self.height = INPUT_HEIGHT
        self.width = INPUT_WIDTH
        config_path = hf_hub_download(CHECKPOINT, "config.json", revision=CHECKPOINT_REVISION)
        constructor = loads(Path(config_path).read_text())["config"]
        self.constructor_config = {key: value for key, value in constructor.items() if key != "__object__"}
        # CPU buffers keep vision_transformer.py's linspace(...).item()
        # construction valid while parameters allocate no checkpoint payload.
        with init_empty_weights(include_buffers=False):
            self.network = create_object(constructor).float()
        if environ.get("DA3_LOAD_WEIGHTS") == "1":
            weights_path = hf_hub_download(CHECKPOINT, "model.safetensors", revision=CHECKPOINT_REVISION)
            state = load_file(weights_path, device="cpu")
            self.network.load_state_dict(
                {key.removeprefix("model."): value for key, value in state.items()},
                strict=True,
                assign=True
            )
            self.network.to("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()

    @inference_mode()
    def forward(self, images: Tensor) -> Tensor:
        parameter = next(self.network.parameters())
        if parameter.device.type == "meta":
            raise RuntimeError("Python inference requires DA3_LOAD_WEIGHTS=1; compilation uses meta topology.")
        if images.dtype != torch.float32 or tuple(images.shape) != (1, 1, 3, self.height, self.width):
            raise ValueError("DA3 expects normalized FP32 [1,1,3,H,W] at the configured resolution")
        images = images.to(parameter.device)
        with torch.autocast(device_type=parameter.device.type, enabled=False):
            return self.network(images)["depth"].float().cpu()

model = DA3NestedMetricDepth()

@compile(
    targets=["x86_64-unknown-linux-gnu"],
    sandbox=Sandbox()
        .pip_install("torch==2.11.0", "torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install(
            "accelerate", "addict==2.4.0", "einops", "evo", "huggingface_hub",
            "numpy<2", "omegaconf", "pillow", "safetensors"
        )
        .pip_install("depth-anything-3"),
    metadata=[
        TorchToSGLangInferenceMetadata(
            model=model,
            compute_architecture="sm_100"
        )
    ]
)
@inference_mode()
def da3nested_giant_large_1_1(
    images: Annotated[
        list[Image.Image],
        Annotations.ChatImages(description="Input images. DA3 requires exactly one image.")
    ],
    *,
    prompt: Annotated[
        str,
        Annotations.ImagePrompt(description="Accepted for OpenAI Images compatibility (ignored).")
    ]="",
) -> Annotated[
    list[ndarray],
    Parameter.DepthMap(description="FP32 metric depth maps in meters, each shaped (H,W) at input resolution.")
]:
    """
    Estimate monocular metric depth with DA3 nested Giant/Large 1.1.
    """
    if len(images) != 1:
        raise ValueError("DA3 requires exactly one input image")
    image = images[0]
    width, height = image.size
    resized = F.resize(image.convert("RGB"), [INPUT_HEIGHT, INPUT_WIDTH])
    pixels = F.normalize(
        F.to_tensor(resized),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    depth = model(pixels[None, None])
    restored = interpolate(
        depth,
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )
    return [restored[0, 0].cpu().numpy()]
