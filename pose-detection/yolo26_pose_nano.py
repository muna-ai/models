#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna", "torchvision", "ultralytics"]
# ///

from muna import compile, Parameter, Sandbox
from muna.beta import TorchToOnnxRuntimeInferenceMetadata
from PIL import Image
from pydantic import BaseModel, Field
from torch import inference_mode, randn, tensor, Tensor
from torch.nn import Module
from torchvision.ops import box_convert
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes
from typing import Annotated
from ultralytics import YOLO

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

class Keypoint(BaseModel):
    x: float = Field(description="Normalized x-coordinate of the keypoint.")
    y: float = Field(description="Normalized y-coordinate of the keypoint.")
    label: str = Field(description="Keypoint label.")
    confidence: float = Field(description="Normalized keypoint confidence score (sigmoid of visibility logit).")

class Pose(BaseModel):
    center_x: float = Field(description="Normalized bounding box center x coordinate.")
    center_y: float = Field(description="Normalized bounding box center y coordinate.")
    width: float = Field(description="Normalized bounding box width.")
    height: float = Field(description="Normalized bounding box height.")
    label: str = Field(description="Detection label.")
    confidence: float = Field(description="Detection confidence score.")
    keypoints: list[Keypoint] = Field(description="Pose keypoints.")

# Create the YOLO model
yolo = YOLO("yolo26n-pose.pt")
model: Module = yolo.model.eval()
labels: dict[int, str] = model.names

# Dry run the model for export
INPUT_SIZE = 640
model_args = [randn(1, 3, INPUT_SIZE, INPUT_SIZE)]
model(*model_args)

@compile(
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("ultralytics")
        .pip_install("opencv-python-headless"),
    metadata=[
        TorchToOnnxRuntimeInferenceMetadata(model=model, model_args=model_args)
    ]
)
@inference_mode()
def yolo26_nano_pose(
    image: Annotated[
        Image.Image,
        Parameter.Generic(description="Input image.")
    ],
    *,
    min_confidence: Annotated[float, Parameter.Numeric(
        description="Minimum detection confidence.",
        min=0.,
        max=1.
    )]=0.25
) -> Annotated[
    list[Pose],
    Parameter.BoundingBoxes(description="Detected poses.")
]:
    """
    Detect poses in an image with YOLO26 Pose (nano).
    """
    image_tensor, xy_scale_factors = _preprocess_image(image, input_size=INPUT_SIZE)
    model_outputs: Tensor = model(image_tensor[None])[0]    # (1,300,6+nk)
    predictions = model_outputs[0]                          # (300,6+nk)
    # Filter by score
    scores = predictions[:,4]
    mask = scores >= min_confidence
    predictions = predictions[mask]
    # Check if any detections remain
    if len(predictions) == 0:
        return []
    # Per-row layout: [x1, y1, x2, y2, score, class_idx, kpt0_x, kpt0_y, kpt0_v, ...]
    boxes_xyxy = predictions[:,:4] * xy_scale_factors.repeat(2)
    boxes_cxcywh = box_convert(boxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh")
    filtered_scores = predictions[:,4]
    class_indices = predictions[:,5]
    keypoint_scale_factors = tensor([float(xy_scale_factors[0]), float(xy_scale_factors[1]), 1.])
    keypoints = predictions[:,6:].reshape(-1, len(KEYPOINT_NAMES), 3) * keypoint_scale_factors
    # Create pose objects
    poses = [
        _create_pose(box, kps, confidence, int(class_idx.item()))
        for box, kps, confidence, class_idx
        in zip(boxes_cxcywh, keypoints, filtered_scores, class_indices)
    ]
    # Return
    return poses

def _preprocess_image(
    image: Image.Image,
    *,
    input_size: int
) -> tuple[Tensor, Tensor]:
    """
    Preprocess an image for inference by downscaling and padding it to have a square aspect.
    """
    # Compute scaled size and padding
    image_width, image_height = image.size
    ratio = min(input_size / image_width, input_size / image_height)
    scaled_width = int(image_width * ratio)
    scaled_height = int(image_height * ratio)
    image_padding = [0, 0, input_size - scaled_width, input_size - scaled_height]
    # Downscale and pad image
    image = image.convert("RGB")
    image = F.resize(image, [scaled_height, scaled_width])
    image = F.pad(image, image_padding, fill=114)
    # Create tensors
    image_tensor = F.to_tensor(image)
    xy_scale_factors = tensor([scaled_width, scaled_height]).reciprocal()
    # Return
    return image_tensor, xy_scale_factors

def _create_pose(
    box: Tensor,
    keypoints: Tensor,
    confidence: Tensor,
    class_index: int
) -> Pose:
    """
    Create a `Pose` object from raw pose tensor data.
    """
    keypoint_data = [Keypoint(
        x=row[0].item(),
        y=row[1].item(),
        confidence=row[2].item(),
        label=KEYPOINT_NAMES[idx]
    ) for idx, row in enumerate(keypoints)]
    pose = Pose(
        center_x=box[0].item(),
        center_y=box[1].item(),
        width=box[2].item(),
        height=box[3].item(),
        label=labels[class_index],
        confidence=confidence.item(),
        keypoints=keypoint_data
    )
    return pose

def _visualize_poses(
    image: Image.Image,
    detections: list[Pose]
) -> Image.Image:
    """
    Render poses on an image.
    """
    from PIL import ImageDraw
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
    # Draw bounding boxes
    image = image.convert("RGB")
    image_tensor = F.to_tensor(image)
    boxes_cxcywh = tensor([[
        detection.center_x * image.width,
        detection.center_y * image.height,
        detection.width * image.width,
        detection.height * image.height
    ] for detection in detections])
    boxes_xyxy = box_convert(
        boxes_cxcywh,
        in_fmt="cxcywh",
        out_fmt="xyxy"
    )
    box_labels = [detection.label for detection in detections]
    result_tensor = draw_bounding_boxes(
        image_tensor,
        boxes=boxes_xyxy,
        labels=box_labels,
        width=8,
        font="Arial",
        font_size=int(0.015 * image.width)
    )
    # Convert back to PIL for keypoint drawing
    result_image = F.to_pil_image(result_tensor)
    draw = ImageDraw.Draw(result_image)
    # Draw keypoints and skeleton for each detection
    for detection in detections:
        keypoints = detection.keypoints
        # Draw skeleton connections (joints)
        for start_label, end_label in KEYPOINT_SKELETON:
            start_kp = next(kp for kp in keypoints if kp.label == start_label)
            end_kp = next(kp for kp in keypoints if kp.label == end_label)
            start_x = start_kp.x * image.width
            start_y = start_kp.y * image.height
            end_x = end_kp.x * image.width
            end_y = end_kp.y * image.height
            draw.line([(start_x, start_y), (end_x, end_y)], fill="lime", width=3)
        # Draw keypoints
        for keypoint in keypoints:
            x = keypoint.x * image.width
            y = keypoint.y * image.height
            radius = 6
            draw.ellipse(
                [x-radius, y-radius, x+radius, y+radius],
                fill=KEYPOINT_COLOR_MAP[keypoint.label],
                outline="black",
                width=2
            )
    # Return
    return result_image

if __name__ == "__main__":
    from pathlib import Path
    from rich import print_json
    # Detect poses
    image_path = Path(__file__).parent / "demo" / "runner.jpg"
    image = Image.open(image_path)
    poses = yolo26_nano_pose(image)
    # Print detections
    print(f"Detected {len(poses)} poses:")
    print_json(data=[pose.model_dump() for pose in poses])
    # Show annotated image
    annotated_image = _visualize_poses(image, poses)
    annotated_image.show()