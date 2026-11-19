#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate", "muna", "sglang==0.5.21",
#   "tokenizers==0.22.2", "torch", "transformers==5.15.0"
# ]
# ///

# `sglang` is used for its streaming tool-call parser only (the `qwen` Hermes
# parser). PyPI ships Linux-only wheels for it; on macOS install it from the
# tag without the Rust extensions (which the parsers never import):
#
#   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
#   SGLANG_BUILD_RUST_EXTS=none pip install --no-deps --no-build-isolation \
#     "git+https://github.com/sgl-project/sglang.git@v0.5.21#subdirectory=python"
#   pip install msgspec orjson pybase64 pyzmq starlette IPython \
#     openai==2.6.1 partial_json_parser xgrammar

from __future__ import annotations
from accelerate import init_empty_weights
from muna import compile, BatchConfig, Sandbox
from muna.beta import (
    Annotations, KVRoutingMetadata, SpeculativeDecodingConfig,
    TorchToSGLangInferenceMetadata
)
from muna.beta.openai import (
    ChatCompletion, ChatCompletionChunk, ChoiceDeltaToolCall,
    DeltaMessage, Message, StreamChoice
)
from os import environ
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from time import time
from tokenizers.decoders import DecodeStream
from torch.nn import Module
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM
from transformers.generation import ContinuousBatchingConfig, GenerationConfig
from transformers.generation.continuous_batching import RequestStatus
from typing import Annotated, Iterator
from uuid import uuid4

# Load Qwen3-4B-Instruct-2507
CHECKPOINT = "Qwen/Qwen3-4B-Instruct-2507"
config = AutoConfig.from_pretrained(CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
with init_empty_weights():
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation="eager",
    )

# Load the DFlash draft model config.
# Transformers has no `DFlashDraftModel` class and the checkpoint's remote code
# is not needed to lower it, so the drafter is not instantiated here. The
# compiler only needs a module carrying the config.
DRAFT_CHECKPOINT = "z-lab/Qwen3-4B-DFlash-b16"
draft_config = AutoConfig.from_pretrained(DRAFT_CHECKPOINT)

class DFlashDraftStub(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

draft_model = DFlashDraftStub(draft_config)

# Resolve the stop tokens. A reply ends with `<|im_end|>`; `<|endoftext|>`
# is the generation config's second end-of-sequence token.
IM_END_TOKEN = tokenizer.convert_tokens_to_ids("<|im_end|>")
END_OF_TEXT_TOKEN = tokenizer.convert_tokens_to_ids("<|endoftext|>")
STOP_TOKENS = [IM_END_TOKEN, END_OF_TEXT_TOKEN]

# Engine limits
MAX_TOTAL_TOKENS = 40_960       # Per-request context (the drafter's `max_position_embeddings`)
NUM_DRAFT_TOKENS = 16           # DFlash block size

# Create the continuous batching manager
generation_config = GenerationConfig(
    max_new_tokens=2048,
    eos_token_id=STOP_TOKENS,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
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

# Define a tokenization function
# This function is used for both inference and KV-aware routing
def _tokenize(
    messages: list[Message],
    tools: list[dict] | None = None
) -> list[int]:
    # Convert `developer` prompt into `system` prompt
    prompt = [
        { "role": "system", "content": message["content"] }
        if message["role"] == "developer" else message
        for message in messages
    ]
    # Render the prompt
    text = tokenizer.apply_chat_template(
        prompt,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    # Tokenize
    return tokenizer.encode(text, add_special_tokens=False)

@compile(
    targets=["x86_64-unknown-linux-gnu"],   # Linux x64 + CUDA only
    sandbox=Sandbox()
        .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("accelerate", "transformers==5.15.0", "tokenizers==0.22.2")
        .pip_install("sglang==0.5.21", flags="--no-deps") # parsers only, skip CUDA dependency tree
        .pip_install(
            "aiohttp", "IPython", "msgspec", "openai", "orjson",
            "partial_json_parser", "pybase64", "pyzmq", "starlette", "xgrammar"
        )
        .env({
            "HF_TOKEN": environ.get("HF_TOKEN"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1"
        }),
    metadata=[
        TorchToSGLangInferenceMetadata(
            model=model,
            compute_architecture="sm_90",   # Compile for Hopper
            tensor_parallelism=1,           # Run on 1xH100
            speculative_decoding=SpeculativeDecodingConfig(
                draft_model=draft_model,
                num_draft_tokens=NUM_DRAFT_TOKENS,
            ),
            max_running_requests=64,        # DFlash batch cap
            max_total_tokens=MAX_TOTAL_TOKENS
        ),
        KVRoutingMetadata(tokenize=_tokenize)
    ]
)
def qwen_3_4b(
    messages: Annotated[list[Message], Annotations.ChatMessages(
        description="Messages comprising the conversation so far.",
        context_length=MAX_TOTAL_TOKENS,
        batch=BatchConfig(mode="continuous")
    )],
    *,
    tools: Annotated[
        list[dict],
        Annotations.ChatTools(description="Tools the model may call.")
    ]=None,
    max_output_tokens: Annotated[int, Annotations.MaxOutputTokens(
        description="Maximum number of tokens in the response.",
        min=1,
        max=MAX_TOTAL_TOKENS
    )]=16384,
    temperature: Annotated[float, Annotations.Temperature(
        description="Sampling temperature.",
        min=0.0,
        max=2.0
    )]=0.7,
    top_p: Annotated[float, Annotations.TopP(
        description="Nucleus sampling probability.",
        min=0.0,
        max=1.0
    )]=0.8,
    top_k: Annotated[int, Annotations.TopK(
        description="Only sample from the top K options for each token.",
        min=1
    )]=20,
    seed: Annotated[
        int,
        Annotations.Seed(description="Seed for reproducible sampling.")
    ]=None,
) -> Iterator[ChatCompletionChunk]:
    """
    Stream chat completions from Qwen3-4B-Instruct-2507.
    """
    # Submit the request to the shared batching manager
    input_ids = _tokenize(messages, tools)
    # Check input token count
    # Speculative steps need `NUM_DRAFT_TOKENS - 1` slots of slack past the
    # last committed token, the same rule the scheduler applies at admission.
    if len(input_ids) + NUM_DRAFT_TOKENS - 1 >= MAX_TOTAL_TOKENS:
        raise ValueError(
            f"The input ({len(input_ids)} tokens) is longer than the model's "
            f"context length ({MAX_TOTAL_TOKENS} tokens)."
        )
    completion_id = f"chatcmpl-{uuid4()}"
    created = int(time())
    prompt_tokens = len(input_ids)
    manager.add_request(
        input_ids=input_ids,
        request_id=completion_id,
        streaming=True,
        max_new_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    # Stream the completion
    try:
        # Create the function call parser. Instruct-2507 is a non-thinking
        # model, so there is no reasoning to split out.
        decoder = DecodeStream(skip_special_tokens=False)
        functions = FunctionCallParser(_tool_specs(tools), "qwen")
        # Initialize decode state
        completion_tokens = 0
        tool_calls = 0
        tool_call_id = ""
        error = ""
        role_sent = False
        # Consume generated tokens
        for output in manager.request_id_iter(request_id=completion_id):
            if output.status == RequestStatus.FAILED:
                error = output.error or "The request was aborted."
                break
            if not role_sent:
                # First chunk announces the assistant role with no content (OpenAI protocol)
                yield _chunk(completion_id, created, DeltaMessage(role="assistant", content=""))
                role_sent = True
            finished = output.status == RequestStatus.FINISHED
            # Decode the new tokens
            normal_text = ""
            new_tokens = output.generated_tokens[completion_tokens:]
            # Trim the matched stop token before decoding
            if finished and new_tokens and new_tokens[-1] in STOP_TOKENS:
                new_tokens = new_tokens[:-1]
            for token_id in new_tokens:
                piece = decoder.step(tokenizer.backend_tokenizer, token_id)
                if piece is not None:
                    normal_text += piece
            completion_tokens = len(output.generated_tokens)
            if not normal_text and not finished:
                continue
            # Split tool calls out of the normal text
            calls = []
            if tools:
                normal_text, calls = functions.parse_stream_chunk(normal_text)
                if finished:
                    tail, more = functions.parse_stream_end()
                    normal_text = normal_text + tail
                    calls = calls + more
            for call in calls:
                if call.name:
                    tool_call_id = f"call_{uuid4()}"
                    tool_calls += 1
                tool_call = ChoiceDeltaToolCall(
                    index=tool_calls - 1,
                    id=tool_call_id if call.name else None,
                    type="function" if call.name else None,
                    function=ChoiceDeltaToolCall.Function(
                        name=call.name,
                        arguments=call.parameters
                    )
                )
                yield _chunk(completion_id, created, DeltaMessage(tool_calls=[tool_call]))
            if normal_text:
                yield _chunk(completion_id, created, DeltaMessage(content=normal_text))
            if finished:
                finish_reason = _finish_reason(
                    completion_tokens=completion_tokens,
                    max_output_tokens=max_output_tokens,
                    tool_calls=tool_calls
                )
                usage = ChatCompletion.Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    prompt_tokens_details=ChatCompletion.Usage.PromptTokensDetails(
                        cached_tokens=getattr(output, "cached_tokens", 0),
                    ),
                )
                yield _chunk(completion_id, created, DeltaMessage(content=""), finish_reason, usage)
        if error:
            raise RuntimeError(error)
    finally:
        # Release the engine request so it stops holding KV and a batch slot.
        # Cancelling a request that already finished is a no-op.
        manager.cancel_request(request_id=completion_id)

def _tool_specs(tools: list[dict] | None) -> list[Tool]:
    """
    Convert OpenAI tool definitions into SGLang tool specs.
    """
    if not tools:
        return []
    return [Tool(type=tool["type"], function=tool["function"]) for tool in tools]

def _chunk(
    completion_id: str,
    created: int,
    delta: DeltaMessage,
    finish_reason: str | None = None,
    usage: ChatCompletion.Usage | None = None
) -> ChatCompletionChunk:
    """
    Construct a single-choice streaming chunk.
    """
    return ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=CHECKPOINT,
        choices=[StreamChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason,
        )],
        usage=usage,
    )

def _finish_reason(
    completion_tokens: int,
    max_output_tokens: int,
    tool_calls: int
) -> str:
    """
    Compute the finish reason for a completed generation.
    """
    if completion_tokens >= max_output_tokens: return "length"
    if tool_calls > 0: return "tool_calls"
    return "stop"
