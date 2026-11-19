#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# NOTE: In order to run and/or compile this model, you must clone Depth Anything V3 into the current working directory.
# Run `git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git` then run/compile this script.

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "addict",
#     "einops",
#     "huggingface_hub",
#     "muna",
#     "omegaconf",
#     "opencv-python-headless",
#     "safetensors",
#     "torchvision"
# ]
# ///

from contextlib import contextmanager
from huggingface_hub import hf_hub_download
from muna import compile, Parameter, Sandbox
from muna.beta import TorchToOnnxRuntimeInferenceMetadata
from numpy import ndarray, uint8
from pathlib import Path
from PIL import Image
from safetensors.torch import load_file
from torch import inference_mode, randn
from torch.nn import Module
from torch.nn.functional import upsample
from torchvision.transforms import functional as F
from typing import Annotated

# Depth Anything uses bicubic interpolation to compute positional encodings.
# This interpolation mode is not supported by several inference backends (ORT, CoreML, etc).
# This context manager overrides the interpolation mode to use bilinear instead.
@contextmanager
def force_bilinear_interpolate():
    from torch.nn import functional as F
    original_interpolate = F.interpolate  # keep reference to the real function
    def wrapped_interpolate(*args, **kwargs):
        if kwargs.get("mode") == "bicubic":
            kwargs["mode"] = "bilinear"
        return original_interpolate(*args, **kwargs)
    F.interpolate = wrapped_interpolate
    try:
        yield
    finally:
        F.interpolate = original_interpolate

# Wrapper to get only the depth output from the model
class _GetDepthWrapper(Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)["depth"]

# Import Depth Anything V3 Components
import sys
sys.path.insert(0, str(Path.cwd() / "Depth-Anything-3" / "src"))
from depth_anything_3.model.da3 import DepthAnything3Net
from depth_anything_3.model.dinov2.dinov2 import DinoV2
from depth_anything_3.model.dpt import DPT

# Build the DA3METRIC-LARGE model directly and load the model
backbone = DinoV2(
    name="vitl",
    out_layers=[4, 11, 17, 23],
    alt_start=-1,
    qknorm_start=-1,
    rope_start=-1,
    cat_token=False
)
head = DPT(
    dim_in=1024,
    output_dim=1,
    features=256,
    out_channels=[256, 512, 1024, 1024]
)
model = DepthAnything3Net(net=backbone, head=head)
INPUT_SIZE = 504

# Load the model weights from HuggingFace
weights_path = hf_hub_download(
    repo_id="depth-anything/DA3METRIC-LARGE",
    filename="model.safetensors"
)
model.load_state_dict({ k.replace("model.", ""): v for k, v in load_file(weights_path).items() })
model.eval()

# Wrap model and force bilinear interpolation
model = _GetDepthWrapper(model)
bilinear_ctx = force_bilinear_interpolate()
bilinear_ctx.__enter__()
example_input = randn(1, 1, 3, INPUT_SIZE, INPUT_SIZE)

@compile(
    sandbox=Sandbox()
        .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("addict", "einops", "huggingface_hub", "omegaconf", "opencv-python-headless",  "safetensors")
        .run_commands("git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git"),
    metadata=[
        TorchToOnnxRuntimeInferenceMetadata(
            model=model,
            model_args=[example_input],
            exporter="none",
            optimization="basic"
        )
    ]
)
@inference_mode()
def depth_anything_v3_metric_large(
    image: Annotated[
        Image.Image,
        Parameter.Generic(description="Input image.")
    ]
) -> Annotated[
    ndarray,
    Parameter.DepthMap(description="Metric depth tensor with shape (H,W).")
]:
    """
    Estimate metric depth from a monocular image with Depth Anything V3 (metric large).
    """    
    # Compute scaled size and padding
    width, height = image.size
    ratio = min(INPUT_SIZE / width, INPUT_SIZE / height)
    scaled_width = int(width * ratio)
    scaled_height = int(height * ratio)
    # Downscale and pad image
    image = F.resize(image, [scaled_height, scaled_width])
    image = image.convert("RGB")
    padding = (0, 0, INPUT_SIZE - scaled_width, INPUT_SIZE - scaled_height)
    image = F.pad(image, padding, fill=0)
    # Convert to tensor and normalize with ImageNet stats
    img_tensor = F.to_tensor(image)
    img_tensor = F.normalize(
        img_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # Run model forward pass and extract depth tensor
    depth_output = model(img_tensor[None, None]) # model takes (1, 1, 3, H, W)
    # Upsample depth map
    depth_resized = upsample(
        depth_output[:,:,:scaled_height, :scaled_width], # Remove padding
        size=(height, width),
        mode="bilinear"
    )
    depth = depth_resized[0,0].cpu().numpy()
    # Return
    return depth

def _visualize_depth(depth: ndarray) -> Image.Image:
    """
    Colorize a depth array using OpenCV's COLORMAP_INFERNO heatmap.
    """
    depth_range = depth.max() - depth.min()
    depth_normalized = (depth - depth.min()) / depth_range
    depth_uint8 = (depth_normalized * 255).astype(uint8)
    depth_colored = applyColorMap(depth_uint8, COLORMAP_INFERNO)
    depth_colored = cvtColor(depth_colored, COLOR_BGR2RGB)
    return Image.fromarray(depth_colored)

if __name__ == "__main__":
    from cv2 import applyColorMap, cvtColor, COLOR_BGR2RGB, COLORMAP_INFERNO
    # Predict
    image_path = Path(__file__).parent / "demo" / "room.jpg"
    image = Image.open(image_path)
    depth = depth_anything_v3_metric_large(image)
    # Visualize result
    depth_image = _visualize_depth(depth)
    depth_image.show()