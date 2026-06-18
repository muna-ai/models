#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub", "muna", "onnxruntime", "sounddevice==0.5.2"]
# ///

from huggingface_hub import hf_hub_download
from json import loads
from muna import compile, Parameter, Sandbox
from muna.beta import Annotations, OnnxRuntimeInferenceSessionMetadata
from numpy import array, float32, fromfile, int64, load, ndarray, ones_like, savez
from onnxruntime import InferenceSession
from pathlib import Path
from typing import get_args, Annotated, Literal

# Language codes
GenerationLanguage = Literal[
    "en-US", "en-GB", "es-ES", "fr-FR", "hi-IN",
    "it-IT", "ja-JP", "pt-BR", "zh-CN"
]

# Generation voices
GenerationVoice = Literal[
    # American English
    "af", "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    # British English
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    # Spanish
    "ef_dora", "em_alex", "em_santa",
    # French
    "ff_siwis",
    # Hindi
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    # Italian
    "if_sara", "im_nicola",
    # Japanese
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    # Portuguese
    "pf_dora", "pm_alex", "pm_santa",
    # Chinese
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
]

# Download the Kokoro model
kokoro_model_path = hf_hub_download(
    repo_id="onnx-community/Kokoro-82M-v1.0-ONNX",
    filename="onnx/model.onnx"
)
kokoro = InferenceSession(kokoro_model_path)
sample_rate = 24_000

# Download the tokenizer vocabulary
tokenizer_path = hf_hub_download(
    repo_id="onnx-community/Kokoro-82M-v1.0-ONNX",
    filename="tokenizer.json"
)
tokenizer = loads(Path(tokenizer_path).read_text())["model"]["vocab"]

# Download ByT5-based grapheme to phoneme converter
phonemizer_path = hf_hub_download(
    repo_id="OpenVoiceOS/g2p-mbyt5-12l-ipa-childes-espeak-onnx",
    filename="fdemelo_g2p-mbyt5-12l-ipa-childes-espeak.onnx"
)
phonemizer = InferenceSession(phonemizer_path)

# Download and create an NPZ file of all Kokoro voices
def _load_kokoro_voice(voice_name: str) -> ndarray:
    voice_path = hf_hub_download(
        repo_id="onnx-community/Kokoro-82M-v1.0-ONNX",
        filename=f"voices/{voice_name}.bin"
    )
    return fromfile(voice_path, dtype=float32).reshape(-1, 1, 256)

kokoro_voices_path = Path("kokoro_voices.npz")
if not kokoro_voices_path.exists():
    kokoro_voices = { voice_name: _load_kokoro_voice(voice_name) for voice_name in get_args(GenerationVoice) }
    savez(kokoro_voices_path, **kokoro_voices)
voices = load(kokoro_voices_path)

@compile(
    sandbox=Sandbox().pip_install("huggingface_hub", "numpy", "onnxruntime"),
    metadata=[
        OnnxRuntimeInferenceSessionMetadata(session=kokoro, model_path=kokoro_model_path),
        OnnxRuntimeInferenceSessionMetadata(session=phonemizer, model_path=phonemizer_path),
    ]
)
def kokoro_tts(
    text: Annotated[
        str,
        Parameter.Generic(description="Text to generate speech from")
    ],
    *,
    voice: Annotated[
        GenerationVoice,
        Annotations.AudioVoice(description="Generation voice.")
    ],
    language: Annotated[
        GenerationLanguage,
        Parameter.Generic(description="Generation language.")
    ]="en-US",
    speed: Annotated[float, Annotations.AudioSpeed(
        description="Voice speed multiplier.",
        min=0.5,
        max=2.0
    )]=1.0
) -> Annotated[ndarray, Parameter.Audio(
    description="Linear PCM audio samples with shape (F,) and sample rate 24KHz.",
    sample_rate=sample_rate
)]:
    """
    Perform text-to-speech with Kokoro TTS.
    """
    # Tokenize input text
    ipa_text = _convert_to_ipa(text, language=language)
    all_tokens = [tokenizer[c] for c in ipa_text if c in tokenizer]
    # Truncate to fit context length (512 total, leaving room for 2 padding tokens)
    tokens = all_tokens[:510]
    tokens_with_padding = [0] + tokens + [0]
    # Run inference
    padded_input_ids = array(tokens_with_padding, dtype=int64)[None]
    ref_s = voices[voice][len(tokens_with_padding)]
    speed_spec = array([speed], dtype=float32)
    outputs = kokoro.run(None, {
        "input_ids": padded_input_ids,
        "style": ref_s,
        "speed": speed_spec
    })
    # Return
    return outputs[0].squeeze()

def _convert_to_ipa(
    text: str,
    *,
    language: GenerationLanguage
) -> str:
    """
    Convert a string to an IPA string with ByT5-G2P.
    """
    # Tokenize input (ByT5-G2P expects angle brackets around language code)
    prompt = f"<{language}>: {text}"
    num_special_tokens = 3
    input_ids = array([list(prompt.encode())]) + num_special_tokens
    attention_mask = ones_like(input_ids)
    # Generate autoregressively
    eos_token_id = 1
    max_length = 510
    decoder_input_ids = [0]
    for idx in range(max_length):
        logits = phonemizer.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": array(decoder_input_ids)[None]
        })[0]
        last_token_logits = logits[:,-1,:]
        next_id = int(last_token_logits.argmax(-1).item())
        decoder_input_ids.append(next_id)
        if next_id == eos_token_id:
            break
    # Decode to IPA
    output_ids = [token_id - num_special_tokens for token_id in decoder_input_ids]
    phoneme_data = bytes([token_id for token_id in output_ids if 0 <= token_id < 256])
    phonemes = phoneme_data.decode()
    # Return
    return phonemes

if __name__ == "__main__":
    import sounddevice as sd
    # Generate audio
    audio = kokoro_tts(
        text="What a time to be alive",
        voice="af_jessica",
        language="en-US",
        speed=1.0
    )
    # Playback
    sd.play(audio, samplerate=sample_rate)
    sd.wait()