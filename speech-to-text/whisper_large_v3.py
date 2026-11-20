#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = ["accelerate", "muna", "numpy", "torch", "transformers==5.15.0"]
# ///

from __future__ import annotations
from accelerate import init_empty_weights
from muna import compile, BatchConfig, Parameter, Sandbox
from muna.beta import Annotations, TorchToSGLangInferenceMetadata
from numpy import ndarray
from os import environ
from transformers import AutoConfig, AutoProcessor, WhisperForConditionalGeneration
from typing import Annotated

# Load Whisper large-v3
CHECKPOINT = "openai/whisper-large-v3"
config = AutoConfig.from_pretrained(CHECKPOINT)
processor = AutoProcessor.from_pretrained(CHECKPOINT)
with init_empty_weights(): # compilation only needs skeleton
    model = WhisperForConditionalGeneration._from_config(config)

# Engine limits
# Each window request owns its 1500 audio placeholder rows plus the decoder's
# learned-position table (448 rows): the engine's per-request context.
ENCODER_TOKENS = config.max_source_positions                # 1500
DECODER_POSITIONS = config.max_target_positions             # 448
MAX_TOTAL_TOKENS = ENCODER_TOKENS + DECODER_POSITIONS       # 1948
MAX_RUNNING_REQUESTS = 64                                   # Concurrent 30 s windows (~312 MB of KV each)
SAMPLE_RATE = processor.feature_extractor.sampling_rate     # 16 kHz

@compile(
    targets=["x86_64-unknown-linux-gnu"],   # Linux x64 + CUDA only
    sandbox=Sandbox()
        .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("accelerate", "numpy", "transformers==5.15.0")
        .env({
            "HF_TOKEN": environ.get("HF_TOKEN"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1"
        }),
    metadata=[
        TorchToSGLangInferenceMetadata(
            model=model,
            compute_architecture="sm_90",   # Hopper
            max_running_requests=MAX_RUNNING_REQUESTS,
            max_total_tokens=MAX_TOTAL_TOKENS
        )
    ]
)
def whisper_large_v3(
    audio: Annotated[ndarray, Parameter.Audio(
        description="Mono audio to transcribe.",
        sample_rate=SAMPLE_RATE,
        batch=BatchConfig(mode="continuous")
    )],
    *,
    language: Annotated[
        str,
        Annotations.TranscriptionLanguage(description="Spoken language as an ISO 639-1 code.")
    ]="en",
) -> Annotated[
    str,
    Parameter.Generic(description="Transcribed text.")
]:
    """
    Transcribe speech with OpenAI Whisper large-v3.
    """
    # Long-form input: the whole recording, unpadded past 30 s
    inputs = processor(
        audio.reshape(-1),
        sampling_rate=SAMPLE_RATE,
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        return_tensors="pt",
    )
    sequences = model.generate(
        inputs["input_features"],
        attention_mask=inputs["attention_mask"],
        return_timestamps=True,
        language=language,
        task="transcribe",
        prompt_ids=None,
        condition_on_prev_tokens=False,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=1.35,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    return processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()
