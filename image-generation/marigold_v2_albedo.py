#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# NOTE: In order to run this model in Python, you must clone Marigold v2 into the current working directory.
# Run `git clone https://github.com/huawei-bayerlab/marigold-v2.git && git -C marigold-v2 checkout 18466672fb8661152434a0efe1184939164cda19`
# then run this script with `MARIGOLD_V2_LOAD_WEIGHTS=1`. Compiling only needs the pipeline topology.

# NOTE: Marigold v2 runs a 20B-parameter DiT, so we compile it only for Linux x86_64 with CUDA.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "muna",
#     "accelerate==1.13.0",
#     "bitsandbytes==0.49.2",
#     "diffusers==0.38.0",
#     "huggingface_hub==1.8.0",
#     "numpy==2.2.5",
#     "omegaconf==2.3.0",
#     "peft==0.18.1",
#     "safetensors==0.8.0",
#     "torch==2.10.0",
#     "torchvision==0.25.0",
#     "transformers==5.4.0"
# ]
# ///

from accelerate import init_empty_weights
from diffusers import AutoencoderKLQwenImage, DiffusionPipeline, QwenImageTransformer2DModel
from muna import compile, Parameter, Sandbox
from muna.beta import Annotations, DiffusersToSGLangInferenceMetadata
from os import environ
from pathlib import Path
from PIL import Image
import sys
import torch
from torch import bfloat16, Generator, Tensor
from torchvision.transforms.functional import resize, to_pil_image, to_tensor
from typing import Annotated, Callable

# Pinned upstream code and checkpoints. Depth is parity-qualified at these
# revisions; the albedo adapter needs its own real-weight parity run. The adapter is one trained Marigold v2 checkpoint:
# a LoRA over the NF4 base DiT plus a fine-tuned VAE decoder.
MARIGOLD_COMMIT = "18466672fb8661152434a0efe1184939164cda19"
BASE_CHECKPOINT = "Qwen/Qwen-Image-Edit-2509"
BASE_REVISION = "d3968ef930e841f4c73640fb8afa3b306a78167e"
ADAPTER_CHECKPOINT = "huawei-bayerlab/marigold-v2-0"
ADAPTER_REVISION = "cdf9810fb690886391a63aec012b5f501064fb0d"
ADAPTER_SUBFOLDER = "albedo"
PROMPT_PREFIX = "qwen_edit_2509_qwen_albedo_rgb_dummy512"

# Upstream `scripts/infer.py` resolves assets under `<repo>/assets` at import time.
REPO_ROOT = Path.cwd() / "marigold-v2"
ASSETS_DIR = REPO_ROOT / "assets"
BASE_CHECKPOINT_DIR = ASSETS_DIR / "checkpoints/Qwen-Image-Edit-2509"
ADAPTER_CHECKPOINT_DIR = ASSETS_DIR / "checkpoints/Marigold-V2"

# Upstream graph constants (`scripts/infer.py` albedo modality)
LATENT_MULTIPLE = 16
PREDICTION_KEY = "albedo_pred"
MAX_PROCESSING_RESOLUTION = 2048
DEFAULT_SEED = 2025

class MarigoldV2AlbedoPipeline(DiffusionPipeline):
    """
    One-step albedo adapter over Qwen-Image-Edit-2509.

    Compilation only instantiates the component topology. Set
    MARIGOLD_V2_LOAD_WEIGHTS=1 to run the pinned upstream reference graph.
    """
    model_cpu_offload_seq = "vae->transformer"

    def __init__(
        self,
        transformer: QwenImageTransformer2DModel,
        vae: AutoencoderKLQwenImage,
        graph: Callable[[dict], dict] | None=None,
        adapter_checkpoint: str=ADAPTER_CHECKPOINT,
        adapter_revision: str=ADAPTER_REVISION,
        adapter_subfolder: str=ADAPTER_SUBFOLDER,
        prompt_prefix: str=PROMPT_PREFIX
    ):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae)
        self.register_to_config(
            adapter_checkpoint=adapter_checkpoint,
            adapter_revision=adapter_revision,
            adapter_subfolder=adapter_subfolder,
            prompt_prefix=prompt_prefix
        )
        self.graph = graph

    @torch.no_grad()
    def __call__(
        self,
        image: Tensor,
        *,
        generator: Generator
    ) -> Tensor:
        """
        Return display-ready sRGB `(1, 3, H, W)` at the processing size.
        """
        if self.graph is None:
            raise RuntimeError("Reference inference requires MARIGOLD_V2_LOAD_WEIGHTS=1")
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("Expected RGB image (1, 3, H, W)")
        if image.shape[-2] % 16 or image.shape[-1] % 16:
            raise ValueError("Processing dimensions must be multiples of 16")
        device = self.vae.device
        batch = { "rgb_norm": image.float().cpu() * 2.0 - 1.0, "out": { } }
        with torch.random.fork_rng(devices=[device]), torch.autocast("cuda", dtype=bfloat16):
            torch.manual_seed(generator.initial_seed())
            batch = self.graph(batch)
        # scripts/infer.py selects FolderRGBAlbedoPrediction(output_color_space="srgb"):
        # its gamma 2.2 conversion and bf16 rounding are already in this result.
        return batch["out"][PREDICTION_KEY].float().clamp(0.0, 1.0).cpu()

def _processing_size(
    width: int,
    height: int,
    processing_resolution: int
) -> tuple[int, int]:
    """
    Bound the longest side by `processing_resolution` (if nonzero), then round
    both sides up to the latent multiple.
    """
    if processing_resolution > 0:
        scale = min(1.0, processing_resolution / max(width, height))
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
    m = LATENT_MULTIPLE
    return (width + m - 1) // m * m, (height + m - 1) // m * m

def _load_pipeline() -> MarigoldV2AlbedoPipeline:
    """
    Build the upstream albedo inference graph with real weights, exactly as
    upstream `scripts/infer.py` does, and wrap its components in the pipeline.
    """
    from huggingface_hub import snapshot_download
    snapshot_download(
        BASE_CHECKPOINT,
        revision=BASE_REVISION,
        local_dir=BASE_CHECKPOINT_DIR,
        allow_patterns=["transformer/**", "vae/**"]
    )
    snapshot_download(
        ADAPTER_CHECKPOINT,
        revision=ADAPTER_REVISION,
        local_dir=ADAPTER_CHECKPOINT_DIR,
        allow_patterns=[
            f"{ADAPTER_SUBFOLDER}/trainables.safetensors",
            f"qwen_text_embeddings/{PROMPT_PREFIX}_prompt_embeds.pt",
            f"qwen_text_embeddings/{PROMPT_PREFIX}_prompt_mask.pt"
        ]
    )
    sys.path.insert(0, str(REPO_ROOT))
    from accelerate import Accelerator
    from evaluation.depth.evaluate_pipeline import _freeze_network_components
    from marigoldv2.core.builder import build_transforms, register_experiment_modules
    from marigoldv2.core.registry import REGISTRY
    from marigoldv2.network.change_network_mode import set_to_eval
    from marigoldv2.script.train.util import make_load_trainables_hook
    from marigoldv2.util.config_resolvers import recursive_load_config
    from omegaconf import OmegaConf
    from scripts.infer import _build_config
    checkpoint_dir = ADAPTER_CHECKPOINT_DIR / ADAPTER_SUBFOLDER
    config_path = Path("/tmp/marigold-v2-albedo.yaml")
    raw_config = _build_config(
        image_dir=Path("/tmp/input"),
        output_dir=Path("/tmp/output"),
        modality="albedo",
        checkpoint=str(checkpoint_dir),
        width=None,
        height=None,
        seed=DEFAULT_SEED
    )
    OmegaConf.save(config=OmegaConf.create(raw_config), f=str(config_path))
    cfg = recursive_load_config(str(config_path))
    cfg.paths = cfg.paths[cfg.paths.use_paths_from]
    accelerator = Accelerator(mixed_precision="bf16")
    REGISTRY["accelerator"] = accelerator
    REGISTRY["cfg"] = cfg
    cfg["device"] = str(accelerator.device)
    register_experiment_modules(cfg)
    build_transforms(cfg.network_components, registry_category="network_components")({ })
    graph = build_transforms(cfg.network_graph, registry_category="network_graph")
    # `set_to_eval` switches every graph step to test mode through this key,
    # as upstream `evaluation/depth/evaluate_pipeline.py` registers it.
    REGISTRY["built_network_graphs"] = { "train": graph }
    load_trainables = make_load_trainables_hook(
        REGISTRY,
        accelerator,
        filename="trainables.safetensors"
    )
    load_trainables([], str(checkpoint_dir))
    _freeze_network_components()
    set_to_eval()
    components = REGISTRY["network_components"]
    return MarigoldV2AlbedoPipeline(
        transformer=components["Diffuser"],
        vae=components["VAE"],
        graph=graph
    )

def _create_pipeline() -> MarigoldV2AlbedoPipeline:
    """
    Build the pipeline topology without weights. This is all the compiler needs:
    the lowering reads the component configs and the pinned adapter from the
    pipeline config.
    """
    transformer_config = QwenImageTransformer2DModel.load_config(
        BASE_CHECKPOINT,
        subfolder="transformer",
        revision=BASE_REVISION
    )
    vae_config = AutoencoderKLQwenImage.load_config(
        BASE_CHECKPOINT,
        subfolder="vae",
        revision=BASE_REVISION
    )
    with init_empty_weights():
        transformer = QwenImageTransformer2DModel.from_config(transformer_config)
        vae = AutoencoderKLQwenImage.from_config(vae_config)
    return MarigoldV2AlbedoPipeline(transformer=transformer, vae=vae)

pipe = (
    _load_pipeline()
    if environ.get("MARIGOLD_V2_LOAD_WEIGHTS") == "1"
    else _create_pipeline()
)

@compile(
    targets=["x86_64-unknown-linux-gnu"],
    sandbox=Sandbox()
        .pip_install("torch==2.10.0", "torchvision==0.25.0", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("accelerate==1.13.0", "diffusers==0.38.0", "transformers==5.4.0"),
    metadata=[
        DiffusersToSGLangInferenceMetadata(
            pipeline=pipe,
            compute_architecture="sm_100"
        )
    ]
)
def marigold_v2_albedo(
    images: Annotated[
        list[Image.Image],
        Annotations.ChatImages(description="Input images. Marigold requires exactly one image.")
    ],
    *,
    prompt: Annotated[
        str,
        Annotations.ImagePrompt(description="Accepted for OpenAI Images compatibility (ignored).")
    ] = "",
    width: Annotated[int, Annotations.ImageWidth(
        description="Output width, or 0 to preserve the input width.",
        min=0,
        max=4096
    )]=0,
    height: Annotated[int, Annotations.ImageHeight(
        description="Output height, or 0 to preserve the input height.",
        min=0,
        max=4096
    )]=0,
    num_images: Annotated[int, Annotations.ImageCount(
        description="Number of posterior samples per prompt.",
        min=1,
        max=4
    )]=1,
    processing_resolution: Annotated[int, Parameter.Numeric(
        description="Maximum processing side (rounded up to a multiple of 16).",
        min=16,
        max=2048
    )]=2048,
    seed: Annotated[int, Parameter.Numeric(
        description="Random seed for the VAE posterior sample.",
        min=0,
        max=2_147_483_647
    )]=2025
) -> Annotated[
    list[Image.Image],
    Parameter.Generic(description="Albedo images.")
]:
    """
    Estimate albedo from an image with Marigold v2.
    """
    if len(images) != 1:
        raise ValueError("Marigold albedo requires exactly one input image")
    image = images[0]
    input_width, input_height = image.size
    output_width = width if width > 0 else input_width
    output_height = height if height > 0 else input_height
    processing_width, processing_height = _processing_size(
        input_width,
        input_height,
        processing_resolution
    )
    image_tensor = to_tensor(image.convert("RGB"))
    image_tensor = resize(image_tensor, [processing_height, processing_width])
    images = [
        _generate_albedo(image_tensor, index + seed, output_width, output_height)
        for index in range(num_images)
    ]
    return images

def _generate_albedo(
    image_tensor,
    seed,
    width,
    height
) -> Image.Image:
    generator = Generator(device="cuda").manual_seed(seed)
    albedo = pipe(image=image_tensor[None], generator=generator)[0]
    albedo = resize(albedo, [height, width])
    return to_pil_image(albedo.clamp(0.0, 1.0))