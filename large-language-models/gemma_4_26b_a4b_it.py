#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = ["accelerate", "muna", "torch", "transformers>=5.12"]
# ///

from __future__ import annotations
from accelerate import init_empty_weights
from enum import IntEnum
from json import dumps
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
from pydantic import BaseModel
from time import time
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import ContinuousBatchingConfig, GenerationConfig
from transformers.generation.continuous_batching import RequestStatus
from typing import Annotated, Iterator
from uuid import uuid4

# Load the Gemma 4 model
# The NVIDIA export quantizes the routed experts to NVFP4 (everything else BF16)
# and is otherwise identical to `google/gemma-4-26B-A4B-it`, tokenizer and chat
# template included. We instantiate on the meta device to skip the download.
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

# Resolve the reasoning, tool call, and turn marker tokens.
# Gemma 4 renders reasoning as a named channel, `<|channel>thought\n...<channel|>`,
# and tool calls as `<|tool_call>call:NAME{...}<tool_call|>`. A turn ends with
# `<turn|>`, or with `<|tool_response>` when the model expects tool results.
THINK_OPEN = tokenizer.convert_tokens_to_ids("<|channel>")
THINK_CLOSE = tokenizer.convert_tokens_to_ids("<channel|>")
TOOL_OPEN = tokenizer.convert_tokens_to_ids("<|tool_call>")
TOOL_CLOSE = tokenizer.convert_tokens_to_ids("<tool_call|>")
TURN_CLOSE = tokenizer.convert_tokens_to_ids("<turn|>")
TOOL_RESPONSE_OPEN = tokenizer.convert_tokens_to_ids("<|tool_response>")

# Response template for one tool call body. Gemma 4 emits tool call arguments
# as JSON with unquoted keys and `<|"|>`-delimited strings (see the
# `response_template` shipped in the checkpoint's `tokenizer_config.json`).
TOOL_CALL_TEMPLATE = {
    "start_anchor": "<|turn>model\n",
    "fields": {
        "tool_calls": {
            "open_pattern": r"call:(?P<name>\w+)",
            "close": "<tool_call|>",
            "repeats": True,
            "content": "json",
            "content_args": {
                "string_delims": [["<|\"|>", "<|\"|>"]],
                "unquoted_keys": True
            },
            "transform": {
                "type": "function",
                "function": { "name": "{name}", "arguments": "{content}" }
            }
        }
    }
}

# Create the continuous batching manager
generation_config = GenerationConfig(
    max_new_tokens=2048,
    eos_token_id=[tokenizer.eos_token_id, TURN_CLOSE, TOOL_RESPONSE_OPEN],
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
    tag="@google/gemma-4-26b-a4b-it",
    access="public",
    targets=["x86_64-unknown-linux-gnu"],   # Linux x64 + CUDA only
    sandbox=Sandbox()
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("accelerate", "transformers>=5.12")
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
        # Compose the token pipeline: raw tokens, reasoning, and tools
        starts_in_reasoning = _starts_in_reasoning(input_ids)
        events = _create_token_stream(completion_id)
        events = _split_token_stream(
            events,
            open_id=THINK_OPEN,
            close_id=THINK_CLOSE,
            out_kind=_EventKind.REASONING,
            buffered=False,
            initial=starts_in_reasoning
        )
        events = _split_token_stream(
            events,
            open_id=TOOL_OPEN,
            close_id=TOOL_CLOSE,
            out_kind=_EventKind.TOOL_CALL,
            buffered=True,
            initial=False
        )
        # Render events as OpenAI chunks
        reasoning_tokens = 0
        tool_calls = 0
        held_newlines = ""
        trim_content = False
        # The channel name (`thought\n`) follows the `<|channel>` marker inside the
        # reasoning span. Collect it until the newline and drop it. When the prompt
        # itself ends in an open channel, the template already wrote the name.
        channel_header = None if starts_in_reasoning else ""
        for event in events:
            match event.kind:
                case _EventKind.REASONING:
                    reasoning_tokens += len(event.token_ids)
                    text = tokenizer.decode(event.token_ids, skip_special_tokens=True)
                    if channel_header is not None:
                        channel_header += text
                        if "\n" not in channel_header:
                            continue
                        text = channel_header.split("\n", 1)[1]
                        channel_header = None
                    text = held_newlines + text
                    kept = text.rstrip("\n")
                    held_newlines = text[len(kept):]
                    trim_content = True
                    if kept:
                        yield _chunk(completion_id, created, DeltaMessage(reasoning_content=kept))
                case _EventKind.TOKENS:
                    held_newlines = ""
                    channel_header = ""
                    text = tokenizer.decode(event.token_ids, skip_special_tokens=True)
                    if trim_content:
                        text = text.lstrip("\n")
                        if text:
                            trim_content = False
                    if text:
                        yield _chunk(completion_id, created, DeltaMessage(content=text))
                case _EventKind.TOOL_CALL:
                    held_newlines = ""
                    channel_header = ""
                    # Keep special tokens: the `<|"|>` string delimiters are part
                    # of the argument syntax. The splitter consumed the close
                    # marker, so restore it for the parser.
                    text = tokenizer.decode(event.token_ids, skip_special_tokens=False)
                    message = tokenizer.parse_response(
                        text + "<tool_call|>",
                        TOOL_CALL_TEMPLATE,
                        prefix="",
                        tools=tools
                    )
                    function = message["tool_calls"][0]["function"]
                    tool_call = ChoiceDeltaToolCall(
                        index=tool_calls,
                        id=f"call_{uuid4()}",
                        type="function",
                        function=ChoiceDeltaToolCall.Function(
                            name=function["name"],
                            arguments=dumps(function["arguments"])
                        )
                    )
                    tool_calls += 1
                    yield _chunk(completion_id, created, DeltaMessage(tool_calls=[tool_call]))
                case _EventKind.FINISHED:
                    held_newlines = ""
                    finish_reason = _finish_reason(
                        completion_tokens=event.completion_tokens,
                        max_output_tokens=max_output_tokens,
                        tool_calls=tool_calls
                    )
                    usage = ChatCompletion.Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=event.completion_tokens,
                        total_tokens=prompt_tokens + event.completion_tokens,
                        prompt_tokens_details=ChatCompletion.Usage.PromptTokensDetails(
                            cached_tokens=event.cached_tokens,
                        ),
                        completion_tokens_details=ChatCompletion.Usage.CompletionTokensDetails(
                            reasoning_tokens=reasoning_tokens,
                        ),
                    )
                    yield _chunk(completion_id, created, DeltaMessage(content=""), finish_reason, usage)
    finally:
        # Release the engine request so it stops holding KV and a batch slot.
        # Cancelling a request that already finished is a no-op.
        manager.cancel_request(request_id=completion_id)

def _create_token_stream(request_id: str) -> Iterator[_Event]:
    """
    Transform the batching manager's chunk stream into a token stream.
    """
    seen = 0
    for chunk in manager.request_id_iter(request_id=request_id):
        if chunk.status == RequestStatus.FAILED:
            raise RuntimeError(chunk.error)
        new_token_ids = chunk.generated_tokens[seen:]
        seen = len(chunk.generated_tokens)
        if new_token_ids:
            yield _Event(kind=_EventKind.TOKENS, token_ids=new_token_ids)
        if chunk.status == RequestStatus.FINISHED:
            yield _Event(
                kind=_EventKind.FINISHED,
                completion_tokens=seen,
                cached_tokens=getattr(chunk, "cached_tokens", 0)
            )
            return

def _split_token_stream(
    upstream: Iterator[_Event],
    open_id: int,
    close_id: int,
    out_kind: int,
    buffered: bool,
    initial: bool
) -> Iterator[_Event]:
    """
    Map token spans between `open_id` and `close_id` and relabel them as `out_kind`.
    All other tokens are passthrough, while marker tokens are consumed.
    When `buffered`, held tokens are emitted as one event at the close marker instead of streamed.
    """
    inside = initial
    buffer = [0][:0]
    for event in upstream:
        if event.kind == _EventKind.FINISHED and inside and buffered and buffer:
            yield _Event(kind=out_kind, token_ids=buffer)
            yield event
            continue
        if event.kind != _EventKind.TOKENS:
            yield event
            continue
        ids = event.token_ids
        while ids:
            marker = close_id if inside else open_id
            crossed = marker in ids
            boundary = ids.index(marker) if crossed else len(ids)
            span = ids[:boundary]
            ids = ids[boundary + 1:]
            if not inside:
                if span:
                    yield _Event(kind=_EventKind.TOKENS, token_ids=span)
            elif buffered:
                buffer = buffer + span
                if crossed:
                    yield _Event(kind=out_kind, token_ids=buffer)
                    buffer.clear()
            else:
                if span:
                    yield _Event(kind=out_kind, token_ids=span)
            if crossed:
                inside = not inside

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

class _EventKind(IntEnum):
    """
    Kind of an event flowing through the token pipeline.
    """
    TOKENS = 0      # unclaimed tokens, destined for `content`
    REASONING = 1   # tokens inside `<|channel>`...`<channel|>`
    TOOL_CALL = 2   # inner tokens of one complete `<|tool_call>` block
    FINISHED = 3    # terminal event, carries stream totals

class _Event(BaseModel):
    """
    One event in the token pipeline.
    """
    kind: int
    token_ids: list[int] = []
    completion_tokens: int = 0  # FINISHED only: total generated, markers included
    cached_tokens: int = 0      # FINISHED only: prefix-cache hits
