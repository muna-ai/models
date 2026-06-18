#
#   Muna
#   Copyright 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub", "muna", "onnxruntime", "sounddevice"]
# ///

from huggingface_hub import hf_hub_download
from json import load as json_load
from muna import compile, Parameter, Sandbox
from muna.beta import OnnxRuntimeInferenceSessionMetadata
from muna.beta.annotations import Annotations
from numpy import array, float32, int64, max as np_max, ndarray, savez
from numpy.random import randn
from numpy.lib.npyio import NpzFile
from numpy import load as np_load
from onnxruntime import InferenceSession
from pathlib import Path
from typing import get_args, Annotated, Literal

# Language codes supported by SuperTonic 2
GenerationLanguage = Literal["en", "ko", "es", "pt", "fr"]

# Voice styles available (5 female, 5 male)
GenerationVoice = Literal[
    "F1", "F2", "F3", "F4", "F5",
    "M1", "M2", "M3", "M4", "M5"
]

# Download model files from HuggingFace
dp_model_path = hf_hub_download(
    repo_id="Supertone/supertonic-2",
    filename="onnx/duration_predictor.onnx"
)
text_enc_model_path = hf_hub_download(
    repo_id="Supertone/supertonic-2",
    filename="onnx/text_encoder.onnx"
)
vector_est_model_path = hf_hub_download(
    repo_id="Supertone/supertonic-2",
    filename="onnx/vector_estimator.onnx"
)
vocoder_model_path = hf_hub_download(
    repo_id="Supertone/supertonic-2",
    filename="onnx/vocoder.onnx"
)

# Download configuration files
config_path = hf_hub_download(
    repo_id="Supertone/supertonic-2",
    filename="onnx/tts.json"
)
unicode_indexer_path = hf_hub_download(
    repo_id="Supertone/supertonic-2",
    filename="onnx/unicode_indexer.json"
)

# Load ONNX models
dp_session = InferenceSession(dp_model_path)
text_enc_session = InferenceSession(text_enc_model_path)
vector_est_session = InferenceSession(vector_est_model_path)
vocoder_session = InferenceSession(vocoder_model_path)

# Load configuration
with open(config_path, "r") as f:
    config = json_load(f)

SAMPLE_RATE = config["ae"]["sample_rate"]
BASE_CHUNK_SIZE = config["ae"]["base_chunk_size"]
CHUNK_COMPRESS_FACTOR = config["ttl"]["chunk_compress_factor"]
LATENT_DIM = config["ttl"]["latent_dim"]

# Pre-generate noise tensor at module load time to avoid capturing random state during tracing
# Max duration ~30 seconds gives max_latent_len ~431 with chunk_size=3072
MAX_LATENT_LEN = 500  # Buffer for longer outputs
PREGENERATED_NOISE: ndarray = randn(1, LATENT_DIM * CHUNK_COMPRESS_FACTOR, MAX_LATENT_LEN).astype(float32)

# Load unicode indexer
with open(unicode_indexer_path, "r") as f:
    UNICODE_INDEXER = json_load(f)

# Download and create NPZ file for all voice styles
def _download_voice_style(voice_name: str) -> tuple[ndarray, ndarray]:
    voice_path = hf_hub_download(
        repo_id="Supertone/supertonic-2",
        filename=f"voice_styles/{voice_name}.json"
    )
    with open(voice_path, "r") as f:
        style_data = json_load(f)
    style_ttl = array(style_data["style_ttl"]["data"], dtype=float32).reshape(*style_data["style_ttl"]["dims"])
    style_dp = array(style_data["style_dp"]["data"], dtype=float32).reshape(*style_data["style_dp"]["dims"])
    return style_ttl, style_dp

voices_path = Path("supertonic2_voices.npz")
if not voices_path.exists():
    all_styles = {}
    for voice_name in get_args(GenerationVoice):
        ttl, dp = _download_voice_style(voice_name)
        all_styles[f"{voice_name}_ttl"] = ttl
        all_styles[f"{voice_name}_dp"] = dp
    savez(voices_path, **all_styles)
voices: NpzFile = np_load(voices_path)

# Symbol replacements for text preprocessing (list of pairs for iteration)
SYMBOL_REPLACEMENTS: list[tuple[str, str]] = [
    ("\u2013", "-"), ("\u2011", "-"), ("\u2014", "-"), ("\u00af", " "), ("_", " "),
    ("\u201c", '"'), ("\u201d", '"'), ("\u2018", "'"), ("\u2019", "'"), ("\u00b4", "'"),
    ("`", "'"), ("[", " "), ("]", " "), ("|", " "), ("/", " "), ("#", " "), ("→", " "), ("←", " "),
]

@compile(
    sandbox=Sandbox().pip_install("huggingface_hub", "onnxruntime"),
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(session=dp_session, model_path=dp_model_path),
        OnnxRuntimeInferenceSessionMetadata(session=text_enc_session, model_path=text_enc_model_path),
        OnnxRuntimeInferenceSessionMetadata(session=vector_est_session, model_path=vector_est_model_path),
        OnnxRuntimeInferenceSessionMetadata(session=vocoder_session, model_path=vocoder_model_path),
    ]
)
def supertonic_2(
    text: Annotated[
        str,
        Parameter.Generic(description="Text to generate speech from.")
    ],
    *,
    voice: Annotated[
        GenerationVoice,
        Annotations.AudioVoice(description="Generation voice (F1-F5 for female, M1-M5 for male).")
    ],
    language: Annotated[
        GenerationLanguage,
        Parameter.Generic(description="Generation language code.")
    ]="en",
    speed: Annotated[float, Annotations.AudioSpeed(
        description="Speech speed multiplier.",
        min=0.7,
        max=2.0
    )]=1.05,
    total_steps: Annotated[int, Parameter.Numeric(
        description="Number of diffusion steps (higher = better quality, slower).",
        min=1,
        max=20
    )]=5
) -> Annotated[ndarray, Parameter.Audio(
    description="Linear PCM audio samples with shape (F,) and sample rate 22050Hz.",
    sample_rate=SAMPLE_RATE
)]:
    """
    Generate speech from text with SuperTonic 2.
    """
    # Get voice style vectors
    style_ttl = voices[f"{voice}_ttl"]
    style_dp = voices[f"{voice}_dp"]
    # Preprocess and encode text
    processed_text = _preprocess_text(text, language)
    text_ids, text_mask = _encode_text([processed_text])
    # Predict duration
    dur_outputs = dp_session.run(None, {
        "text_ids": text_ids,
        "style_dp": style_dp,
        "text_mask": text_mask
    })
    duration = dur_outputs[0] / speed
    # Encode text with style
    text_emb_outputs = text_enc_session.run(None, {
        "text_ids": text_ids,
        "style_ttl": style_ttl,
        "text_mask": text_mask
    })
    text_emb = text_emb_outputs[0]
    # Sample noisy latent
    xt, latent_mask = _sample_noisy_latent(duration)
    # Iterative denoising (batch size is always 1)
    total_step_np = array([total_steps], dtype=float32)
    for step in range(total_steps):
        current_step = array([step], dtype=float32)
        xt_outputs = vector_est_session.run(None, {
            "noisy_latent": xt,
            "text_emb": text_emb,
            "style_ttl": style_ttl,
            "text_mask": text_mask,
            "latent_mask": latent_mask,
            "current_step": current_step,
            "total_step": total_step_np
        })
        xt = xt_outputs[0]
    # Generate waveform
    wav_outputs = vocoder_session.run(None, {"latent": xt})
    wav = wav_outputs[0]
    # Return squeezed audio
    return wav.squeeze()

def _preprocess_text(text: str, lang: str) -> str:
    """
    Preprocess text with normalization and language tags.
    """
    # Symbol replacements (unicode normalization removed - handled by model's text processor)
    for old, new in SYMBOL_REPLACEMENTS:
        text = text.replace(old, new)
    # Clean whitespace
    text = " ".join(text.split()).strip()
    # Add period if needed
    if len(text) > 0 and text[-1] not in ".!?;:,'\"')]}>":
        text = text + "."
    # Add language tags
    return f"<{lang}>{text}</{lang}>"

def _encode_text(text_list: list[str]) -> tuple[ndarray, ndarray]:
    """
    Convert text to unicode indices and create attention mask.
    """
    # Get lengths
    text_lengths = array([len(t) for t in text_list], dtype=int64)
    max_len = int(np_max(text_lengths))
    # Build text_ids as flat list then reshape (avoids 2D element-by-element assignment)
    # Start with first text's row to establish type, then use + for subsequent
    first_text = text_list[0]
    first_row: list[int] = [UNICODE_INDEXER[ord(first_text[j])] for j in range(len(first_text))]
    first_padding: list[int] = [0 for _ in range(max_len - len(first_text))]
    text_ids_flat: list[int] = first_row + first_padding
    # Process remaining texts (if any)
    for idx in range(1, len(text_list)):
        text = text_list[idx]
        row: list[int] = [UNICODE_INDEXER[ord(text[j])] for j in range(len(text))]
        padding: list[int] = [0 for _ in range(max_len - len(text))]
        text_ids_flat = text_ids_flat + row
        text_ids_flat = text_ids_flat + padding
    text_ids = array(text_ids_flat, dtype=int64).reshape(len(text_list), max_len)
    # Create mask
    ids = array([i for i in range(max_len)], dtype=int64)
    mask = (ids < text_lengths.reshape(-1, 1)).astype(float32)
    text_mask = mask.reshape(-1, 1, max_len)
    return text_ids, text_mask

def _sample_noisy_latent(duration: ndarray) -> tuple[ndarray, ndarray]:
    """
    Sample noisy latent representation for diffusion.
    """
    bsz = len(duration)
    wav_len_max = np_max(duration) * SAMPLE_RATE
    wav_lengths = (duration * SAMPLE_RATE).astype(int64)
    chunk_size = BASE_CHUNK_SIZE * CHUNK_COMPRESS_FACTOR
    latent_len = int((wav_len_max + chunk_size - 1) / chunk_size)
    latent_dim = LATENT_DIM * CHUNK_COMPRESS_FACTOR
    # Use pre-generated noise (slice to required latent_len)
    noisy_latent = PREGENERATED_NOISE[:, :, :latent_len]
    # Create latent mask
    latent_size = BASE_CHUNK_SIZE * CHUNK_COMPRESS_FACTOR
    latent_lengths = ((wav_lengths + latent_size - 1) // latent_size).astype(int64)
    max_latent_len = int(np_max(latent_lengths))
    ids = array([i for i in range(max_latent_len)], dtype=int64)
    mask = (ids < latent_lengths.reshape(-1, 1)).astype(float32)
    latent_mask = mask.reshape(-1, 1, max_latent_len)
    # Apply mask
    noisy_latent = noisy_latent * latent_mask
    return noisy_latent, latent_mask

if __name__ == "__main__":
    import sounddevice as sd
    # Generate audio
    audio = supertonic_2(
        text="I like one of my options more.",
        voice="F2"
    )
    print(f"Generated audio shape: {audio.shape}")
    print(f"Duration: {len(audio) / SAMPLE_RATE:.2f} seconds")
    # Playback
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()