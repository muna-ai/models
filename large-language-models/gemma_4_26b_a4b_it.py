#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = ["accelerate", "muna", "torch", "transformers>=5.12"]
# ///

from accelerate import init_empty_weights
from muna import compile, BatchConfig, Parameter, Sandbox
from muna.beta import (
    Annotations, KVRoutingMetadata, SpeculativeDecodingConfig, TorchToSGLangInferenceMetadata
)
from muna.beta.openai import (
    ChatCompletion, ChatCompletionChunk, DeltaMessage, Message, StreamChoice
)
from time import time
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import ContinuousBatchingConfig, GenerationConfig
from transformers.generation.continuous_batching import RequestStatus
from typing import Annotated, Iterator
from uuid import uuid4

# Load the Gemma 4 model
# We instantiate the model on the meta device to skip a ~52GB download
CHECKPOINT = "google/gemma-4-26B-A4B-it"
config = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Load the draft model
DRAFT_CHECKPOINT = "z-lab/gemma-4-26B-A4B-it-DFlash"
draft_config = AutoConfig.from_pretrained(DRAFT_CHECKPOINT, trust_remote_code=True)
with init_empty_weights():
    draft_model = AutoModelForCausalLM.from_config(draft_config)

# Create the continuous batching manager
generation_config = GenerationConfig(
    max_new_tokens=2048,
    eos_token_id=config.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
)
batching_config = ContinuousBatchingConfig(
    per_request_processors=True,
    max_memory_percent=0.9,
)
manager = model.init_continuous_batching(
    generation_config=generation_config,
    continuous_batching_config=batching_config,
)

# Define a tokenization function
# This function is both used for both inference and KV-aware routing
def _tokenize(messages) -> list[int]:
    return tokenizer.apply_chat_template(
        [{ "role": m.role, "content": m.content } for m in messages],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )

@compile(
    targets=["x86_64-unknown-linux-gnu"],   # Linux x64 + CUDA only
    sandbox=Sandbox()
        .pip_install("accelerate", "nvidia-modelopt", "tiktoken", "torch", "transformers>=5.7"),
    metadata=[
        TorchToSGLangInferenceMetadata(
            model=model,
            compute_architecture="sm_100",  # Blackwell only
            speculative_decoding=SpeculativeDecodingConfig(
                draft_model=draft_model,
                num_draft_tokens=8,         # Draft tokens per step
            ),
            max_running_requests=8,
            max_total_tokens=32_768
        ),
        KVRoutingMetadata(tokenize=_tokenize)
    ]
)
def gemma_4_26b_a4b_it(
    messages: Annotated[list[Message], Parameter.Generic(
        description="Messages comprising the conversation so far.",
        batch=BatchConfig(mode="continuous")
    )],
    *,
    max_output_tokens: Annotated[int, Annotations.MaxOutputTokens(
        description="Maximum number of tokens in the response.",
        min=1,
        max=32768
    )]=32768,
    temperature: Annotated[float, Annotations.Temperature(
        description="Sampling temperature.",
        min=0.0,
        max=2.0
    )]=0.7,
    top_p: Annotated[float, Annotations.TopP(
        description="Nucleus sampling probability.",
        min=0.0,
        max=1.0
    )]=0.95,
) -> Iterator[ChatCompletionChunk]:
    """
    Stream chat completions from Gemma 4 26B.
    """
    # Tokenize message history
    input_ids = _tokenize(messages)
    completion_id = f"chatcmpl-{uuid4()}"
    created = int(time())
    prompt_tokens = len(input_ids)
    # Submit the request to the shared batching manager
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
    cached_tokens = 0
    seen = 0
    for chunk in manager.request_id_iter(request_id=completion_id):
        new_token_ids = chunk.generated_tokens[seen:]
        seen = len(chunk.generated_tokens)
        finished = chunk.status == RequestStatus.FINISHED
        finish_reason_final = "length" if seen >= max_output_tokens else "stop"
        cached_tokens = getattr(chunk, "cached_tokens", 0)
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
                        finish_reason=finish_reason_final,
                    )],
                    usage=ChatCompletion.Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        prompt_tokens_details=ChatCompletion.Usage.PromptTokensDetails(
                            cached_tokens=cached_tokens,
                        ),
                    ),
                )
                break
        # Decode
        token_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        completion_tokens += len(new_token_ids)
        finish_reason = finish_reason_final if finished else None
        # Create usage
        usage = ChatCompletion.Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=ChatCompletion.Usage.PromptTokensDetails(
                cached_tokens=cached_tokens,
            ),
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