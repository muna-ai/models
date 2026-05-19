#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["ai-edge-litert", "huggingface_hub", "muna", "rich", "torchvision"]
# ///

from ai_edge_litert.interpreter import Interpreter
from huggingface_hub import hf_hub_download
from muna import compile, Parameter, Sandbox
from muna.beta import TFLiteInterpreterMetadata
from numpy import array, ndarray
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
from torchvision.transforms.v2 import functional as F
from typing import Annotated

KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

class Keypoint(BaseModel):
    x: float = Field("Normalized X position.")
    y: float = Field("Normalized Y position.")
    label: str = Field(description="Keypoint label.")
    confidence: float = Field("Normalized keypoint confidence score.")

class Pose(BaseModel):
    x: float = Field(description="Normalized minimum point X coordinate.")
    y: float = Field(description="Normalized minimum point Y coordinate.")
    width: float = Field(description="Normalized width.")
    height: float = Field(description="Normalized height.")
    confidence: float = Field(description="Pose confidence score in range [0, 1].")
    keypoints: list[Keypoint] = Field(description="Pose keypoints.")

model_path = hf_hub_download("muna-ai/natml-hub", "movenet-multipose-lightning-256-fp16-cle.tflite")
model = Interpreter(model_path)
model.allocate_tensors()
input_idx = model.get_input_details()[0]["index"]
output_idx = model.get_output_details()[0]["index"]

@compile(
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("ai-edge-litert", "huggingface_hub"),
    metadata=[
        TFLiteInterpreterMetadata(
            interpreter=model,
            model_path=model_path,
            options=["xnnpack_force_fp16"],
        )
    ]
)
def movenet_multipose(
    image: Annotated[
        Image.Image,
        Parameter.Generic(description="Input image.")
    ],
    min_confidence: Annotated[float, Parameter.Numeric(
        description="Minimum detection confidence.",
        min=0.,
        max=1.
    )]=0.3
) -> Annotated[
    list[Pose],
    Parameter.BoundingBoxes(description="Detected poses.")
]:
    """
    Detect poses in an image with MoveNet Multipose.
    """
    # Preprocess image
    image = F.resize(image, [256, 256])
    image = image.convert("RGB")
    image_tensor = array(image, dtype="uint8")[None]
    # Run model
    model.set_tensor(input_idx, image_tensor)
    model.invoke()
    logits = model.get_tensor(output_idx) # (1,6,56)
    # Parse poses
    valid_pose_mask = logits[0,:,55] >= min_confidence
    valid_pose_data = logits[0,valid_pose_mask] # (N,56)
    poses = [_create_pose(row) for row in valid_pose_data]
    # Return
    return poses

def _create_pose(data: ndarray) -> Pose:
    """
    Create a `Pose` object from raw pose tensor data.
    """
    keypoints=[Keypoint(
        x=data[3 * idx + 1].item(),
        y=data[3 * idx].item(),
        label=keypoint,
        confidence=data[3 * idx + 2].item()
    ) for idx, keypoint in enumerate(KEYPOINTS)]
    pose = Pose(
        x=data[52].item(),
        y=data[51].item(),
        width=data[54].item(),
        height=data[53].item(),
        confidence=data[55].item(),
        keypoints=keypoints
    )
    return pose

def _visualize_poses(
    image: Image.Image,
    poses: list[Pose]
) -> Image.Image:
    """
    Render poses on an image.
    """
    from PIL import ImageDraw
    from torch import tensor
    from torchvision.utils import draw_bounding_boxes
    KEYPOINT_SKELETON = [
        ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"),
        ("right_eye", "right_ear"), ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle")
    ]
    KEYPOINT_COLOR_MAP = {
        "nose": "red", "left_eye": "blue", "right_eye": "blue", "left_ear": "purple",
        "right_ear": "purple", "left_shoulder": "orange", "right_shoulder": "orange",
        "left_elbow": "yellow", "right_elbow": "yellow", "left_wrist": "cyan",
        "right_wrist": "cyan", "left_hip": "magenta", "right_hip": "magenta",
        "left_knee": "pink", "right_knee": "pink", "left_ankle": "brown",
        "right_ankle": "brown"
    }
    image = image.convert("RGB")
    image_width, image_height = image.size
    # Draw bounding boxes for each pose
    image_tensor = F.pil_to_tensor(image)
    boxes_xyxy = tensor([[
        pose.x * image_width,
        pose.y * image_height,
        (pose.x + pose.width) * image_width,
        (pose.y + pose.height) * image_height
    ] for pose in poses])
    labels = [f"{pose.confidence:.2f}" for pose in poses]
    result_tensor = draw_bounding_boxes(
        image_tensor,
        boxes=boxes_xyxy,
        labels=labels,
        width=4,
        font="Arial",
        font_size=int(0.02 * image_width)
    )
    # Convert back to PIL for keypoint drawing
    result_image = F.to_pil_image(result_tensor)
    draw = ImageDraw.Draw(result_image)
    # Draw keypoints and skeleton for each pose
    for pose in poses:
        keypoints_by_label = { kp.label: kp for kp in pose.keypoints }
        # Draw skeleton connections
        for start_label, end_label in KEYPOINT_SKELETON:
            start_kp = keypoints_by_label[start_label]
            end_kp = keypoints_by_label[end_label]
            start = (start_kp.x * image_width, start_kp.y * image_height)
            end = (end_kp.x * image_width, end_kp.y * image_height)
            draw.line([start, end], fill="lime", width=3)
        # Draw keypoints
        for keypoint in pose.keypoints:
            x = keypoint.x * image_width
            y = keypoint.y * image_height
            radius = 6
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=KEYPOINT_COLOR_MAP[keypoint.label],
                outline="black",
                width=2
            )
    # Return
    return result_image

if __name__ == "__main__":
    from rich import print_json
    # Load image
    image_path = Path(__file__).parent / "demo" / "runner.jpg"
    image = Image.open(image_path)
    # Predict
    poses = movenet_multipose(image)
    # Print poses
    print(f"Detected {len(poses)} poses:")
    print_json(data=[pose.model_dump() for pose in poses])
    # Show annotated image
    annotated_image = _visualize_poses(image, poses)
    annotated_image.show()