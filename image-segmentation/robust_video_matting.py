#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceSessionMetadata
from numpy import concatenate, cumsum, ndarray, prod
from onnxruntime import InferenceSession
from pathlib import Path
from PIL import Image
from requests import get
from torch import tensor, zeros
from torchvision.transforms.functional import to_pil_image, to_tensor
from typing import Annotated

# Download model
model_url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.onnx"
model_path = Path(Path(model_url).name)
if not model_path.exists():
    model_path.write_bytes(get(model_url).content)

# Create ONNXRuntime inference session
model = InferenceSession(model_path)

@compile(
    sandbox=Sandbox().pip_install("onnxruntime"),
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(
            session=model,
            model_path=model_path
        )
    ]
)
def robust_video_matting(
    image: Annotated[
        Image.Image,
        Parameter.Generic(description="Input image.")
    ],
    *,
    guidance: Annotated[
        ndarray,
        Parameter.Generic(description="Matting guidance tensor.")
    ]=None
) -> tuple[
    Annotated[Image.Image, Parameter.Generic(description="Mask image.")],
    Annotated[ndarray, Parameter.Generic(description="Updated matting guidance tensor.")]
]:
    """
    Perform portrait matting with Robust Video Matting.
    """
    # Convert image to tensor
    image_tensor = to_tensor(image)[None]
    # Populate recurrent state tensors
    shapes = [
        (1,16,135,240),
        (1,20,68,120),
        (1,40,34,60),
        (1,64,17,30)
    ]
    total_elements = sum(prod(s) for s in shapes)
    if guidance is None:
        r1 = zeros(shapes[0])
        r2 = zeros(shapes[1])
        r3 = zeros(shapes[2])
        r4 = zeros(shapes[3])
    else:
        assert guidance.size == total_elements, "Guidance has incorrect size"
        splits = cumsum([0] + [prod(s) for s in shapes])
        r_flat = [guidance[splits[i]:splits[i+1]] for i in range(4)]
        r1 = tensor(r_flat[0]).view(shapes[0])
        r2 = tensor(r_flat[1]).view(shapes[1])
        r3 = tensor(r_flat[2]).view(shapes[2])
        r4 = tensor(r_flat[3]).view(shapes[3])
    # Run the model
    _, mask_tensor, r1, r2, r3, r4 = model.run(None, {
        "image": image_tensor.numpy(),
        "r1i": r1.numpy(),
        "r2i": r2.numpy(),
        "r3i": r3.numpy(),
        "r4i": r4.numpy()
    })
    # Convert mask to image
    mask = to_pil_image(mask_tensor.squeeze())
    # Flatten and concatenate guidance tensors
    guidance_tensors = [r1, r2, r3, r4]
    guidance = concatenate([r.flatten() for r in guidance_tensors])
    # Return
    return mask, guidance