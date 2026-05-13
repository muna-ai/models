#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate", "flash-attn", "jinja2", "muna", "nvidia-modelopt",
#   "tiktoken", "torch", "transformers>=5.7"
# ]
# ///

from accelerate import init_empty_weights
from contextlib import contextmanager
from muna import compile, BatchConfig, Parameter, Sandbox
from muna.beta import Annotations, DFlashSpeculativeDecoding, SGLangInferenceMetadata
from muna.beta.openai import (
    ChatCompletion, ChatCompletionChunk, DeltaMessage,
    Message, StreamChoice
)
from time import time
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import ContinuousBatchingConfig, GenerationConfig
from transformers.generation.continuous_batching import RequestStatus
from transformers.modeling_utils import PreTrainedModel
from typing import Annotated, Iterator
from uuid import uuid4

# Patch transformers since model was written for 4.x but we need >5.0 for continuous batching
from transformers.utils import import_utils as _t_import_utils
if not hasattr(_t_import_utils, "is_torch_fx_available"):
    _t_import_utils.is_torch_fx_available = lambda: False

# Helpers for loading the model in transformers v5
@contextmanager
def suppress_init_weights():
    saved = PreTrainedModel.init_weights
    PreTrainedModel.init_weights = lambda self, *a, **kw: None
    try:
        yield
    finally:
        PreTrainedModel.init_weights = saved

@contextmanager
def force_eager_attn(cfg):
    def _walk(c):
        if hasattr(c, "_attn_implementation"):
            c._attn_implementation = "eager"
        for sub in vars(c).values():
            if hasattr(sub, "_attn_implementation"):
                _walk(sub)
    _walk(cfg)
    yield cfg

# Load the Kimi K2.5 model
# We instantiate the model on the meta device to skip a ~500GB download
CHECKPOINT = "nvidia/Kimi-K2.5-NVFP4"
config = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
with force_eager_attn(config), suppress_init_weights(), init_empty_weights():
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        attn_implementation="eager",
    )

# Load the DFlash draft model
# Same as above, instantiate on the meta device
DRAFT_CHECKPOINT = "z-lab/Kimi-K2.5-DFlash"
draft_config = AutoConfig.from_pretrained(DRAFT_CHECKPOINT, trust_remote_code=True)
with force_eager_attn(draft_config), suppress_init_weights(), init_empty_weights():
    draft_model = AutoModelForCausalLM.from_config(
        draft_config,
        trust_remote_code=True,
        attn_implementation="eager",
    )

# Create the continuous batching manager
generation_config = GenerationConfig(
    max_new_tokens=2048,
    eos_token_id=[
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>")
    ],
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
)
batching_config = ContinuousBatchingConfig(
    per_request_processors=True,
    use_cuda_graph=True,
    max_memory_percent=0.9,
)
manager = model.init_continuous_batching(
    generation_config=generation_config,
    continuous_batching_config=batching_config,
)

@compile(
    targets=["x86_64-unknown-linux-gnu"], # Linux x64 + CUDA only
    sandbox=Sandbox()
        .pip_install("accelerate", "nvidia-modelopt", "tiktoken", "torch", "transformers>=5.7"),
    metadata=[
        SGLangInferenceMetadata(
            model=model,
            speculative_decoding=DFlashSpeculativeDecoding(
                draft_model=draft_model,
                num_draft_tokens=8, # number of draft tokens to generate per step
                node_budget=0       # >1 enables DDTree on DFlash
            ),
            max_running_requests=4,
            max_total_tokens=32_768
        )
    ]
)
def kimi_k2_5(
    messages: Annotated[
        list[Message],
        Parameter.Generic(
            description="Messages comprising the conversation so far.",
            batch=BatchConfig(mode="continuous")
        )
    ],
    *,
    max_output_tokens: Annotated[int, Annotations.MaxOutputTokens(
        description="Maximum number of tokens in the response.",
        min=1,
        max=16384
    )]=2048,
    temperature: Annotated[float, Annotations.SamplingTemperature(
        description="Sampling temperature.",
        min=0.0,
        max=2.0
    )]=0.7,
    top_p: Annotated[float, Annotations.SamplingProbability(
        description="Nucleus sampling probability.",
        min=0.0,
        max=1.0
    )]=0.95,
) -> Iterator[ChatCompletionChunk]:
    """
    Stream chat completions from Kimi K2.5 (NVFP4).
    """
    # Tokenize message history
    input_ids = tokenizer.apply_chat_template(
        [{ "role": m.role, "content": m.content } for m in messages],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    completion_id = f"chatcmpl-{uuid4()}"
    created = int(time())
    prompt_tokens = len(input_ids)
    # Submit the request to the shared batching manager. Other concurrent calls
    # to this predictor add their own requests in parallel; the manager merges
    # them all into the next forward step.
    manager.add_request(
        input_ids=input_ids,
        request_id=completion_id,
        streaming=True,
        max_new_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    # First chunk announces the assistant role with no content, mirroring the
    # OpenAI streaming protocol.
    yield ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=CHECKPOINT,
        choices=[StreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant", content=""),
            finish_reason=None,
        )],
    )
    # Stream tokens from the manager
    completion_tokens = 0
    seen = 0
    for chunk in manager.request_id_iter(request_id=completion_id):
        new_token_ids = chunk.generated_tokens[seen:]
        seen = len(chunk.generated_tokens)
        finished = chunk.status == RequestStatus.FINISHED
        # Check for empty chunk
        if not new_token_ids:
            # Usually signifies a status change
            if not finished:
                continue
            # Yield end of stream
            else:
                yield ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=CHECKPOINT,
                    choices=[StreamChoice(
                        index=0,
                        delta=DeltaMessage(content=""),
                        finish_reason="stop",
                    )],
                    usage=ChatCompletion.Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
                break
        # Decode
        token_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        completion_tokens += len(new_token_ids)
        finish_reason = "stop" if finished else None
        # Create usage
        usage = ChatCompletion.Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ) if finished else None
        # Yield chunk
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=CHECKPOINT,
            choices=[StreamChoice(
                index=0,
                delta=DeltaMessage(content=token_text),
                finish_reason=finish_reason,
            )],
            usage=usage,
        )
        # Handle finish with content
        if finished:
            break

if __name__ == "__main__":
    chat_messages = [
        Message(role="system", content="You are Kimi, a helpful AI assistant."),
        Message(role="user", content="What is the capital of France?"),
    ]
    for chunk in kimi_k2_5(chat_messages):
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
