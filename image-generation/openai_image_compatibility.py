#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna", "requests"]
# ///

from io import BytesIO
from muna import compile, BatchConfig, Parameter
from muna.beta import Annotations
from PIL import Image
from requests import get
from typing import Annotated

image_response = get(
    "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
    headers={ "User-Agent": "py2cpp-image-compatibility-test/1.0" },
)
image = Image.open(BytesIO(image_response.content))

@compile()
def fake_image_model(
    prompt: Annotated[list[str], Parameter.Generic(
        description="Text descriptions of the desired images.",
        batch=BatchConfig(mode="dynamic", capacity=4)
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
        max=4
    )]=1,
) -> Annotated[
    list[Image.Image],
    Parameter.Generic(description="Generated images.")
]:
    """
    Example that compiles a model compatible with the OpenAI Images API.
    """
    resized_image = image.resize((width, height))
    return [resized_image] * num_images

if __name__ == "__main__":
    results = fake_image_model("image of a dog")
    results[0].show()