#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate", "muna", "sglang==0.5.19", "tokenizers==0.22.2", "torch",
#   "transformers==5.12.1"
# ]
# ///

# `sglang` is used for its streaming reasoning / tool-call parsers only. PyPI
# ships Linux-only wheels for it; on macOS install it from the tag without the
# Rust extensions (which the parsers never import):
#
#   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
#   SGLANG_BUILD_RUST_EXTS=none pip install --no-deps --no-build-isolation \
#     "git+https://github.com/sgl-project/sglang.git@v0.5.19#subdirectory=python"
#   pip install orjson pybase64 requests pillow starlette torchvision IPython \
#     openai==2.6.1 partial_json_parser xgrammar

from __future__ import annotations
from accelerate import init_empty_weights
from muna import compile, BatchConfig, Parameter, Sandbox
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
from sglang.srt.parser.reasoning_parser import ReasoningParser
from time import time
from tokenizers.decoders import DecodeStream
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import ContinuousBatchingConfig, GenerationConfig
from transformers.generation.continuous_batching import RequestStatus
from typing import Annotated, Iterator
from uuid import uuid4

# Load the Gemma 4 model
CHECKPOINT = "nvidia/Gemma-4-26B-A4B-NVFP4"
config = AutoConfig.from_pretrained(CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Load the DFlash draft model (block 16; the engine pins the block size to it)
DRAFT_CHECKPOINT = "z-lab/gemma-4-26B-A4B-it-DFlash"
draft_config = AutoConfig.from_pretrained(DRAFT_CHECKPOINT)
with init_empty_weights():
    draft_model = AutoModelForCausalLM.from_config(draft_config)

# Resolve the reasoning and turn marker tokens.
# Gemma 4 renders reasoning as a named channel, `<|channel>thought\n...<channel|>`,
# and tool calls as `<|tool_call>call:NAME{...}<tool_call|>`.
THINK_OPEN = tokenizer.convert_tokens_to_ids("<|channel>")
THINK_CLOSE = tokenizer.convert_tokens_to_ids("<channel|>")
TURN_CLOSE = tokenizer.convert_tokens_to_ids("<turn|>")
TOOL_RESPONSE_OPEN = tokenizer.convert_tokens_to_ids("<|tool_response>")
STOP_TOKENS = [tokenizer.eos_token_id, TURN_CLOSE, TOOL_RESPONSE_OPEN]

# Create the continuous batching manager
generation_config = GenerationConfig(
    max_new_tokens=2048,
    eos_token_id=STOP_TOKENS,
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

# Define a tokenization function
# This function is both used for both inference and KV-aware routing
def _tokenize(
    messages: list[Message],
    tools: list[dict] | None = None
) -> list[int]:
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )

@compile(
    targets=["x86_64-unknown-linux-gnu"],   # Linux x64 + CUDA only
    sandbox=Sandbox()
        .pip_install("torch==2.13.0", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("accelerate", "transformers==5.12.1", "tokenizers==0.22.2")
        # Parsers only: skip sglang's CUDA dependency tree, then add the pure
        # packages its import chain needs (measured on Linux).
        .pip_install("sglang==0.5.19", flags="--no-deps") # parsers only, skip CUDA dependency tree
        .pip_install("aiohttp", "IPython", "openai", "orjson", "partial_json_parser", "pybase64", "xgrammar")
        .env({
            "HF_TOKEN": environ.get("HF_TOKEN"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1"
        }),
    metadata=[
        TorchToSGLangInferenceMetadata(
            model=model,
            compute_architecture="sm_100",  # Compile for Blackwell
            tensor_parallelism=1,
            speculative_decoding=SpeculativeDecodingConfig(
                draft_model=draft_model,
                num_draft_tokens=16,        # DFlash block size
            ),
            max_running_requests=8,
            max_total_tokens=131_072
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
    tools: Annotated[
        list[dict],
        Annotations.ChatTools(description="Tools the model may call.")
    ]=None,
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
    Stream chat completions from Gemma 4 26B A4B (NVFP4).
    """
    # Submit the request to the shared batching manager. Other concurrent calls
    # to this predictor add their own requests in parallel; the manager merges
    # them all into the next forward step.
    input_ids = _tokenize(messages, tools)
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
    )
    # Stream the completion
    try:
        # First chunk announces the assistant role with no content (match OpenAI protocol)
        yield _chunk(completion_id, created, DeltaMessage(role="assistant", content=""))
        # Create parsers for reasoning and function calling
        decoder = DecodeStream(skip_special_tokens=False)
        reasoning = ReasoningParser(
            "gemma4",
            stream_reasoning=True,
            force_reasoning=_starts_in_reasoning(input_ids)
        )
        functions = FunctionCallParser(_tool_specs(tools), "gemma4")
        # Initialize decode state
        completion_tokens = 0
        reasoning_tokens = 0
        pending_tokens = 0
        tool_calls = 0
        tool_call_id = ""
        held_newlines = ""
        trim_content = False
        error = ""
        # Consume generated tokens
        for output in manager.request_id_iter(request_id=completion_id):
            if output.status == RequestStatus.FAILED:
                error = output.error or "The request was aborted."
                break
            finished = output.status == RequestStatus.FINISHED
            # Decode the new tokens and split the text into reasoning and normal text
            segments = []
            text = ""
            new_tokens = output.generated_tokens[completion_tokens:]
            # Trim the matched stop token before decoding
            if finished and new_tokens and new_tokens[-1] in STOP_TOKENS:
                new_tokens = new_tokens[:-1]
            for token_id in new_tokens:
                pending_tokens += 1
                piece = decoder.step(tokenizer.backend_tokenizer, token_id)
                if piece is not None:
                    text += piece
            if text:
                reasoning_text, normal_text = reasoning.parse_stream_chunk(text)
                segments.append((reasoning_text, normal_text, pending_tokens))
                pending_tokens = 0
            completion_tokens = len(output.generated_tokens)
            if finished:
                reasoning_text, normal_text = reasoning.parse_stream_end()
                segments.append((reasoning_text, normal_text, 0))
            # Render the segments as OpenAI chunks
            for index, segment in enumerate(segments):
                reasoning_text, normal_text, span = segment
                if reasoning_text:
                    reasoning_tokens += span
                    text = held_newlines + reasoning_text
                    kept = text.rstrip("\n")
                    held_newlines = text[len(kept):]
                    trim_content = True
                    if kept:
                        yield _chunk(completion_id, created, DeltaMessage(reasoning_content=kept))
                if not normal_text and not finished:
                    continue
                calls = []
                if tools:
                    normal_text, calls = functions.parse_stream_chunk(normal_text)
                    if finished and index == len(segments) - 1:
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
                    held_newlines = ""
                    if trim_content:
                        normal_text = normal_text.lstrip("\n")
                        if normal_text:
                            trim_content = False
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
                    completion_tokens_details=ChatCompletion.Usage.CompletionTokensDetails(
                        reasoning_tokens=reasoning_tokens,
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

def _starts_in_reasoning(input_ids: list[int]) -> bool:
    """
    The Gemma 4 template closes the generation prompt with an empty
    `<|channel>thought\n<channel|>` block when thinking is disabled, and with
    an open `<|channel>thought\n` when thinking is enabled after a tool
    response (trailing text, so the *last* token is not the marker). Scan back
    to the most recent marker instead of checking the final token.
    """
    for token in reversed(input_ids):
        if token == THINK_OPEN:
            return True
        if token == THINK_CLOSE:
            return False
    return False

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